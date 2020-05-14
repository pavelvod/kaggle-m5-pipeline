import pickle
import lightgbm as lgb
import luigi
import pandas as pd
import inspect
import hashlib

from config import cache_path, TARGET, split_features


class BaseTarget(luigi.Target):
    def exists(self):
        pass


class LocalTarget(BaseTarget):
    def __init__(self, task):
        self.path = cache_path / f'{task.task_id}.pickle'

    def exists(self):
        return self.path.exists()

    def save(self, df):
        # df = reduce_mem_usage(df)
        df.to_pickle(self.path)
        return

    def load(self):
        return pd.read_pickle(self.path)


class CSVTarget(BaseTarget):
    def __init__(self, task):
        self.path = cache_path / f'{task.task_id}.csv'

    def exists(self):
        return self.path.exists()

    def save(self, df):
        # df = reduce_mem_usage(df)
        df.to_csv(self.path, index=False)
        return

    def load(self):
        return pd.read_csv(self.path)


class ModelTarget(BaseTarget):
    def __init__(self, task):
        self.path = cache_path / f'{task.task_id}.model'

    def exists(self):
        return self.path.exists()

    def save(self, estimator):
        pickle.dump(estimator, open(self.path, 'wb'))
        return

    def load(self):
        return pickle.load(open(self.path, 'rb'))


class LGBMBinarySetTarget(BaseTarget):
    def __init__(self, task):
        self.path = cache_path / f'{task.task_id}.data'

    def exists(self):
        return self.path.exists()

    def save(self, df):
        # df = reduce_mem_usage(df)
        df.save_binary(str(self.path))
        return

    def load(self, *args, **kwargs):
        return lgb.Dataset(str(self.path), *args, **kwargs)


class BaseTask(luigi.Task):

    def output(self):
        return luigi.LocalTarget(cache_path / f'{self.task_id}.pickle')

    def save(self, df, *args, **kwargs):
        self.output().save(df, *args, **kwargs)
        return

    def load(self, task_name, *args, **kwargs):
        return self.input()[task_name].load(*args, **kwargs)


class Task(BaseTask):
    # src_hash = luigi.Parameter(default=hashlib.md5(str(inspect.getsource(self)).encode()).hexdigest())

    def output(self):
        return LocalTarget(self)


class ModelTask(Task):
    def output(self):
        return ModelTarget(self)


class LGBMBinarySetTask(Task):
    def output(self):
        return LGBMBinarySetTarget(self)


class CSVTask(Task):
    def output(self):
        return CSVTarget(self)


class FeatureTask(Task):
    def save(self, df, *args, **kwargs):
        drop_columns = split_features + [TARGET]
        cols2drop = []
        for col in drop_columns:
            if col in df.columns:
                cols2drop.append(col)

        if len(cols2drop) > 0:
            df = df.drop(columns=[TARGET])

        self.output().save(df, *args, **kwargs)
        return
