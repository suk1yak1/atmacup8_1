import csv
import inspect
import os
import time
import warnings
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from utils import Logger

logger = Logger(name="feature", log_level="INFO", file_log=False)

warnings.filterwarnings("ignore")


@contextmanager
def timer(name):
    t0 = time.time()
    logger.info(f"[{name}] start")
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")


@dataclass
class Feature(BaseEstimator, TransformerMixin, metaclass=ABCMeta):

    dir = "."
    prefix: str = ""
    suffix: str = ""
    description: str = ""

    def __post_init__(self):
        self.name = self.__class__.__name__
        self.train_path = Path(self.dir) / f"{self.name}_train.pkl"
        self.test_path = Path(self.dir) / f"{self.name}_test.pkl"
        self.prefix = self.prefix + "-" if self.prefix else ""
        self.suffix = "-" + self.suffix if self.suffix else ""
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()

    def __cal__(self, X: pd.DataFrame):
        return self.transform(X)

    @abstractmethod
    def create_feature(self, train, test):
        raise NotImplementedError

    def run(self, train, test):
        with timer(self.name):
            self.create_feature(train, test)
            self.train.columns = self.prefix + self.train.columns + self.suffix
            self.test.columns = self.prefix + self.test.columns + self.suffix
            create_memo(
                self.name,
                ",\n".join(self.train.columns.values.tolist()),
                self.description,
            )
            logger.info(f"made: {self.train.columns.values.tolist()}")
        return self

    def save(self):
        self.train.to_pickle(str(self.train_path))
        self.test.to_pickle(str(self.test_path))

    def load(self):
        self.train = pd.read_pickle(str(self.train_path))
        self.test = pd.read_pickle(str(self.test_path))


def create_memo(class_name, col_name, description):
    file_path = Feature.dir + "/_features_memo.csv"
    if not os.path.isfile(file_path):
        with open(file_path, "w"):
            pass

    with open(file_path, "r+") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        col = [line for line in lines if line.split(",")[0] == class_name]
        if len(col) != 0:
            return

        writer = csv.writer(f)
        writer.writerow([class_name, col_name, description])


def get_features(namespace):
    for _, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, train, test, overwrite):

    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            logger.info(f"[{f.name}] was skipped")
        else:
            f.run(train, test).save()
