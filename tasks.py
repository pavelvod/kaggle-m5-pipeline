import importlib
import inspect
import warnings
from multiprocessing import Pool

import time
from math import ceil

import luigi

import base
from base import FeatureTask

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
from utils import sizeof_fmt, reduce_mem_usage, merge_by_concat, seed_everything
from functools import partial
import lightgbm as lgb
from base import Task

# import mlflow.lightgbm

# mlflow.lightgbm.autolog()

from config import input_path, TARGET, END_TRAIN, \
    START_TRAIN, MAIN_INDEX, remove_features, P_HORIZON, STORES_IDS, \
    SEED, N_CORES, ROLS_SPLIT, i_split, j_split, lgb_params, SHIFT_DAY, split_features


def make_lag_roll(LAG_DAY, base_test):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id', 'd', TARGET]]
    col_name = 'rolling_mean_tmp_' + str(shift_day) + '_' + str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES, len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df


class GetTmpBaseFeatures(Task):

    def run(self):
        ########################### Load Data
        #################################################################################
        print('Load Main Data')

        # Here are reafing all our data 
        # without any limitations and dtype modification
        train_df = pd.read_csv(input_path / 'sales_train_validation.csv')
        prices_df = pd.read_csv(input_path / 'sell_prices.csv')
        calendar_df = pd.read_csv(input_path / 'calendar.csv')

        ########################### Make Grid
        #################################################################################
        print('Create Grid')

        # drop unused days
        cols = []
        cols += [col for col in train_df.columns if not col.startswith('d_')]
        cols += [f'd_{d}' for d in np.arange(START_TRAIN, END_TRAIN + 1000) if f'd_{d}' in train_df]
        train_df = train_df.loc[:, cols]

        # We can tranform horizontal representation 
        # to vertical "view"
        # Our "index" will be 'id','item_id','dept_id','cat_id','store_id','state_id'
        # and labels are 'd_' coulmns

        index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        grid_df = pd.melt(train_df,
                          id_vars=index_columns,
                          var_name='d',
                          value_name=TARGET)

        # If we look on train_df we se that 
        # we don't have a lot of traning rows
        # but each day can provide more train data
        print('Train rows:', len(train_df), len(grid_df))

        # To be able to make predictions
        # we need to add "test set" to our grid
        add_grid = pd.DataFrame()
        for i in range(1, 29):
            temp_df = train_df[index_columns]
            temp_df = temp_df.drop_duplicates()
            temp_df['d'] = 'd_' + str(END_TRAIN + i)
            temp_df[TARGET] = np.nan
            add_grid = pd.concat([add_grid, temp_df])

        grid_df = pd.concat([grid_df, add_grid])
        grid_df = grid_df.reset_index(drop=True)

        # Remove some temoprary DFs
        del temp_df, add_grid, train_df

        # Let's check our memory usage
        print("{:>20}: {:>8}".format('Original grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

        # We can free some memory 
        # by converting "strings" to categorical
        # it will not affect merging and 
        # we will not lose any valuable data
        for col in index_columns:
            grid_df[col] = grid_df[col].astype('category')

        # Let's check again memory usage
        print("{:>20}: {:>8}".format('Reduced grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

        ########################### Product Release date
        #################################################################################
        print('Release week')

        # It seems that leadings zero values
        # in each train_df item row
        # are not real 0 sales but mean
        # absence for the item in the store
        # we can safe some memory by removing
        # such zeros

        # Prices are set by week
        # so it we will have not very accurate release week 
        release_df = prices_df.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
        release_df.columns = ['store_id', 'item_id', 'release']

        # Now we can merge release_df
        grid_df = merge_by_concat(grid_df, release_df, ['store_id', 'item_id'])
        del release_df

        # We want to remove some "zeros" rows
        # from grid_df 
        # to do it we need wm_yr_wk column
        # let's merge partly calendar_df to have it
        grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk', 'd']], ['d'])

        # Now we can cutoff some rows 
        # and safe memory 
        grid_df = grid_df[grid_df['wm_yr_wk'] >= grid_df['release']]
        grid_df = grid_df.reset_index(drop=True)

        # Let's check our memory usage
        print("{:>20}: {:>8}".format('Original grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

        grid_df['release'] = grid_df['release'] - grid_df['release'].min()
        grid_df['release'] = grid_df['release'].astype(np.int16)

        # Let's check again memory usage
        print("{:>20}: {:>8}".format('Reduced grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

        print('Size:', grid_df.shape)

        self.save(grid_df)


class GetIndex(Task):
    """
    this task configure which rows will be used
    all other features will be left-joined
    """

    def requires(self):
        return {
            'data1': GetTmpBaseFeatures()
        }

    def run(self):
        grid_df = self.input()['data1'].load()
        grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

        self.save(grid_df.loc[:, MAIN_INDEX + split_features + [TARGET]])


class GetBaseFeatures(Task):

    def requires(self):
        return {
            'data1': GetTmpBaseFeatures()
        }

    def run(self):
        grid_df = self.input()['data1'].load()
        grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

        del grid_df['wm_yr_wk']
        self.save(grid_df)


class GetStoreFeatures(Task):
    store_id = luigi.Parameter()

    def requires(self):
        reqs: dict = {}
        reqs['index'] = GetIndex()
        reqs['features'] = {}
        for x in ['tasks']:
            module = importlib.import_module(x)
            for name, cls in inspect.getmembers(module):
                if inspect.isclass(cls):
                    if (base.FeatureTask in cls.mro()) and cls != base.FeatureTask:
                        reqs['features'][cls.__name__] = cls(store_id=self.store_id)
        return reqs

    def run(self):
        main_df = self.load('index')
        main_df = main_df.loc[main_df['store_id'] == self.store_id, :]

        for name, inp in self.input()['features'].items():
            features_df = pd.read_pickle(inp.path)
            main_df = main_df.merge(features_df, on=MAIN_INDEX, validate='one_to_one')
            for col in features_df.columns:
                if col not in MAIN_INDEX:
                    print(f'{name}.{col} was added')

        features = [col for col in list(main_df) if col not in remove_features]
        for feature in features:
            print(feature)
        main_df = main_df[MAIN_INDEX + [TARGET] + features]
        main_df = main_df[main_df['d'] >= START_TRAIN].reset_index(drop=True)
        self.save(main_df)


class SaveTrainForStore(base.LGBMBinarySetTask):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            'StoreData': GetStoreFeatures(store_id=self.store_id)
        }

    def run(self):
        grid_df = self.load('StoreData')
        drop_columns = [col for col in grid_df.columns if col.startswith('__')]
        features_columns = set(grid_df.columns) - set(MAIN_INDEX + [TARGET] + drop_columns)

        train_mask = grid_df['d'] <= END_TRAIN
        train_data = lgb.Dataset(grid_df[train_mask][features_columns],
                                 label=grid_df[train_mask][TARGET]
                                 )
        self.save(train_data)


class SaveValidForStore(base.LGBMBinarySetTask):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            'StoreData': GetStoreFeatures(store_id=self.store_id),
            'TrainForStore': SaveTrainForStore(store_id=self.store_id),
        }

    def run(self):
        grid_df = self.input()['StoreData'].load()
        drop_columns = [col for col in grid_df.columns if col.startswith('__')]
        features_columns = set(grid_df.columns) - set(MAIN_INDEX + [TARGET] + drop_columns)
        train_mask = grid_df['d'] <= END_TRAIN
        valid_mask = train_mask & (grid_df['d'] > (END_TRAIN - P_HORIZON))
        train_data = self.load('TrainForStore')
        valid_data = lgb.Dataset(grid_df[valid_mask][features_columns],
                                 label=grid_df[valid_mask][TARGET],
                                 #                                  weight = grid_df[valid_mask]['__weight'],
                                 reference=train_data)
        self.save(valid_data)


class SaveTestForStore(Task):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            'StoreData': GetStoreFeatures(store_id=self.store_id)
        }

    def run(self):
        grid_df = self.load('StoreData')
        drop_columns = [col for col in grid_df.columns if col.startswith('__')]
        features_columns = set(grid_df.columns) - set(MAIN_INDEX + [TARGET] + drop_columns)
        preds_mask = grid_df['d'] > (END_TRAIN - 100)
        grid_df = grid_df[preds_mask].reset_index(drop=True)
        keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
        grid_df = grid_df[keep_cols]
        self.save(grid_df)


class TrainModelForStore(base.ModelTask):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            'TrainForStore': SaveTrainForStore(store_id=self.store_id),
            'ValidForStore': SaveValidForStore(store_id=self.store_id)
        }

    def run(self):
        train_data = self.load('TrainForStore')
        valid_data = self.load('ValidForStore', reference=train_data)

        seed_everything(SEED)
        estimator = lgb.train(lgb_params,
                              train_data,
                              valid_sets=[train_data, valid_data],
                              verbose_eval=100
                              )
        self.save(estimator)


class GetBaseTest(Task):

    def requires(self):
        return {
            f'StoreData_{store_id}': SaveTestForStore(store_id=store_id) for store_id in STORES_IDS
        }

    def run(self):
        base_test = pd.DataFrame()

        for store_id in STORES_IDS:
            temp_df = self.load(f'StoreData_{store_id}')
            temp_df['store_id'] = store_id
            base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)

        self.save(base_test)


class MakePredictions(Task):

    def requires(self):
        reqs = {'BaseTest': GetBaseTest()}

        for store_id in STORES_IDS:
            reqs[f'StoreData'] = GetStoreFeatures(store_id=store_id)
            reqs[f'ModelForStore_{store_id}'] = TrainModelForStore(store_id=store_id)
        return reqs

    def run(self):
        df = self.load('StoreData')
        drop_columns = [col for col in df.columns if col.startswith('__')]
        features_columns = set(df.columns) - set(MAIN_INDEX + [TARGET] + drop_columns)
        del df

        base_test = self.load('BaseTest')
        all_preds = pd.DataFrame()

        main_time = time.time()

        for PREDICT_DAY in range(1, 29):
            print('Predict | Day:', PREDICT_DAY)
            start_time = time.time()

            grid_df = base_test.copy()
            make_lag_roll_ = partial(make_lag_roll, base_test=base_test)
            grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll_, ROLS_SPLIT)], axis=1)

            for store_id in STORES_IDS:
                estimator = self.load(f'ModelForStore_{store_id}')
                day_mask = base_test['d'] == (END_TRAIN + PREDICT_DAY)
                store_mask = base_test['store_id'] == store_id
                mask = (day_mask) & (store_mask)
                base_test[TARGET][mask] = estimator.predict(grid_df[mask][features_columns])

            temp_df = base_test[day_mask][['id', TARGET]]
            temp_df.columns = ['id', 'F' + str(PREDICT_DAY)]
            if 'id' in list(all_preds):
                all_preds = all_preds.merge(temp_df, on=['id'], how='left')
            else:
                all_preds = temp_df.copy()

            print('#' * 10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F' + str(PREDICT_DAY)].sum()))
            del temp_df

        all_preds = all_preds.reset_index(drop=True)
        self.save(all_preds)


class MakePredictionsCSV(base.CSVTask):
    task_namespace = 'simple'

    def requires(self):
        return {
            f'MergePredictions': MakePredictions()
        }

    def run(self):
        all_preds = pd.read_pickle(self.requires()['MergePredictions'].output().path)
        submission = pd.read_csv(input_path / 'sample_submission.csv')[['id']]
        submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
        self.save(submission)


class RunPipeline(luigi.WrapperTask):

    def requires(self):
        yield MakePredictionsCSV()


class GetPriceFeatures(Task):

    def requires(self):
        return {
            'data1': GetTmpBaseFeatures()
        }

    def run(self):
        grid_df = self.input()['data1'].load()
        prices_df = pd.read_csv(input_path / 'sell_prices.csv')
        calendar_df = pd.read_csv(input_path / 'calendar.csv')

        print('Prices')

        # We can do some basic aggregations
        prices_df['price_max'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
        prices_df['price_min'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
        prices_df['price_std'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
        prices_df['price_mean'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')

        # and do price normalization (min/max scaling)
        prices_df['price_norm'] = prices_df['sell_price'] / prices_df['price_max']

        # Some items are can be inflation dependent
        # and some items are very "stable"
        prices_df['price_nunique'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')
        prices_df['item_nunique'] = prices_df.groupby(['store_id', 'sell_price'])['item_id'].transform('nunique')

        # I would like some "rolling" aggregations
        # but would like months and years as "window"
        calendar_prices = calendar_df[['wm_yr_wk', 'month', 'year']]
        calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
        prices_df = prices_df.merge(calendar_prices[['wm_yr_wk', 'month', 'year']], on=['wm_yr_wk'], how='left')
        del calendar_prices

        # Now we can add price "momentum" (some sort of)
        # Shifted by week
        # by month mean
        # by year mean
        prices_df['price_momentum'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id'])[
            'sell_price'].transform(lambda x: x.shift(1))
        prices_df['price_momentum_m'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'month'])[
            'sell_price'].transform('mean')
        prices_df['price_momentum_y'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'year'])[
            'sell_price'].transform('mean')

        del prices_df['month'], prices_df['year']

        # Merge Prices
        original_columns = list(grid_df)
        grid_df = grid_df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        keep_columns = [col for col in list(grid_df) if col not in original_columns]
        grid_df = grid_df[MAIN_INDEX + keep_columns]
        grid_df = reduce_mem_usage(grid_df)

        print('Size:', grid_df.shape)

        self.save(grid_df)


class GetCalenderFeatures(Task):

    def requires(self):
        return {
            'data1': GetTmpBaseFeatures()
        }

    def run(self):
        grid_df = self.input()['data1'].load()
        calendar_df = pd.read_csv(input_path / 'calendar.csv')
        grid_df = grid_df[MAIN_INDEX]

        # Merge calendar partly
        icols = ['date',
                 'd',
                 'event_name_1',
                 'event_type_1',
                 'event_name_2',
                 'event_type_2',
                 'snap_CA',
                 'snap_TX',
                 'snap_WI']

        grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')

        # Minify data
        # 'snap_' columns we can convert to bool or int8
        icols = ['event_name_1',
                 'event_type_1',
                 'event_name_2',
                 'event_type_2',
                 'snap_CA',
                 'snap_TX',
                 'snap_WI']
        for col in icols:
            grid_df[col] = grid_df[col].astype('category')

        # Convert to DateTime
        grid_df['date'] = pd.to_datetime(grid_df['date'])

        # Make some features from date
        grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
        grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
        grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
        grid_df['tm_y'] = grid_df['date'].dt.year
        grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
        grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x / 7)).astype(np.int8)

        grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
        grid_df['tm_w_end'] = (grid_df['tm_dw'] >= 5).astype(np.int8)

        # Remove date
        grid_df['__date'] = grid_df['date'].copy()
        del grid_df['date']

        print('Size:', grid_df.shape)
        self.save(grid_df)


class CombineBaseData(Task):

    def requires(self):
        return {
            'data2': GetPriceFeatures(),
            'data3': GetCalenderFeatures(),
            'data4': GetBaseFeatures()
        }

    def run(self):
        # Now we have 3 sets of features
        data = pd.concat([self.load('data4'),
                          self.load('data2').iloc[:, 2:],
                          self.load('data3').iloc[:, 2:]],
                         axis=1)

        data = reduce_mem_usage(data)

        # Let's check again memory usage
        print("{:>20}: {:>8}".format('Full Grid', sizeof_fmt(data.memory_usage(index=True).sum())))
        print('Size:', data.shape)
        self.save(data)


class CombineBaseDataForStore(Task):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            'data': CombineBaseData()
        }

    def run(self):
        data = (self.load('data')
                    .loc[lambda dt: dt['store_id'] == self.store_id, :]
                    )

        self.save(data)


class MeanEncodings(FeatureTask):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            'CombineData': CombineBaseDataForStore(store_id=self.store_id)
        }

    def run(self):
        grid_df = self.load('CombineData')

        grid_df[TARGET][grid_df['d'] > (1913 - 28)] = np.nan
        base_cols = list(grid_df)

        icols = [
            ['state_id'],
            ['store_id'],
            ['cat_id'],
            ['dept_id'],
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        ]

        for col in icols:
            print('Encoding', col)
            col_name = '_' + '_'.join(col) + '_'
            grid_df['enc' + col_name + 'mean'] = grid_df.groupby(col)[TARGET].transform('mean').astype(np.float16)
            grid_df['enc' + col_name + 'std'] = grid_df.groupby(col)[TARGET].transform('std').astype(np.float16)

        keep_cols = [col for col in list(grid_df) if col not in base_cols]
        grid_df = grid_df[['id', 'd'] + keep_cols]

        self.save(grid_df)


class SlidingShiftFeatures(FeatureTask):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            'CombineData': CombineBaseDataForStore(store_id=self.store_id)
        }

    def run(self):
        grid_df = self.load('CombineData')[MAIN_INDEX + [TARGET]]
        # Rollings
        # with sliding shift
        for d_shift in i_split:
            print('Shifting period:', d_shift)
            for d_window in j_split:
                col_name = 'rolling_mean_tmp_' + str(d_shift) + '_' + str(d_window)
                grid_df[col_name] = grid_df.groupby(['id'])[TARGET].transform(
                    lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)
        self.save(grid_df)


class BaseFeatures(FeatureTask):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            'CombineData': CombineBaseDataForStore(store_id=self.store_id)
        }

    def run(self):
        grid_df = self.load('CombineData')
        self.save(grid_df)


class RollingFeatures(FeatureTask):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            'CombineData': CombineBaseDataForStore(store_id=self.store_id)
        }

    def run(self):
        grid_df = self.load('CombineData')[MAIN_INDEX + [TARGET]]

        for i in [7, 14, 30, 60, 180]:
            print('Rolling period:', i)
            grid_df['rolling_mean_' + str(i)] = grid_df.groupby(['id'])[TARGET].transform(
                lambda x: x.shift(SHIFT_DAY).rolling(i).mean()).astype(np.float16)
            grid_df['rolling_std_' + str(i)] = grid_df.groupby(['id'])[TARGET].transform(
                lambda x: x.shift(SHIFT_DAY).rolling(i).std()).astype(np.float16)

        self.save(grid_df)


class ShiftFeatures(FeatureTask):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            'CombineData': CombineBaseDataForStore(store_id=self.store_id)
        }

    def run(self):
        grid_df = self.load('CombineData')[MAIN_INDEX + [TARGET]]
        LAG_DAYS = [col for col in range(SHIFT_DAY, SHIFT_DAY + 15)]
        grid_df = grid_df.assign(**{
            '{}_lag_{}'.format(col, l): grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l)).astype(np.float16)
            for l in LAG_DAYS
            for col in [TARGET]
        })
        self.save(grid_df)


if __name__ == '__main__':
    luigi.build([RunPipeline()], local_scheduler=False, workers=1)
