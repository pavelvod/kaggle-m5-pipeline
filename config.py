from pathlib import Path

import psutil

root_path = Path(r'your/path/to/project/folder')
input_path = root_path / 'input'
cache_path = root_path / 'cache'

if not cache_path.exists():
    cache_path.mkdir(parents=True)

TARGET = 'sales'  # Our main target
END_TRAIN = 1913  # Last day in train set
TRAIN_DAYS = 28 * 20 - 1
START_TRAIN = 0  # END_TRAIN - TRAIN_DAYS
MAIN_INDEX = ['id', 'd']  # We can identify item by these columns
remove_features = ['id', 'state_id', 'store_id',
                   'date', 'wm_yr_wk', 'd', TARGET]
P_HORIZON = 28
SHIFT_DAY = 28
STORES_IDS = ['CA_1', 'CA_2', 'CA_3', 'CA_4',
              'TX_1', 'TX_2', 'TX_3',
              'WI_1', 'WI_2', 'WI_3'
              ]
SEED = 42
N_CORES = psutil.cpu_count()

ROLS_SPLIT = []
split_features = ['store_id']
i_split = [1, 7, 14]
j_split = [7, 14, 30, 60]

for i in i_split:
    for j in j_split:
        ROLS_SPLIT.append([i, j])

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'subsample': 0.5,
    'subsample_freq': 1,
    'learning_rate': 0.03,
    'num_leaves': 2 ** 11 - 1,
    'min_data_in_leaf': 2 ** 12 - 1,
    'feature_fraction': 0.5,
    'max_bin': 100,
    'n_estimators': 1400,
    'boost_from_average': False,
    'verbose': -1,
}
