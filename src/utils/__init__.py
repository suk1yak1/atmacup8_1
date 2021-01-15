import copy
import datetime
import os
from logging import Formatter, StreamHandler, basicConfig, getLogger

import numpy as np
import pandas as pd
import psutil
from sklearn.base import BaseEstimator, TransformerMixin


def get_dt_now() -> str:
    return datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9))
    ).strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(
        self, name: str = __file__, log_level: str = "INFO", file_log: bool = True
    ) -> None:

        format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        if file_log:
            basicConfig(
                filename=f"log/{name}_{get_dt_now()}.log",
                level=log_level,
                format=format,
            )
        else:
            basicConfig(
                level=log_level, format=format,
            )
        self.logger = getLogger(name)
        formatter = Formatter(format)

        # stdout
        handler = StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.process = psutil.Process(os.getpid())

    def debug(self, msg: str) -> None:
        self.logger.debug(f"[{self.get_use_mem_gb():.1f}GB] {msg}")

    def info(self, msg: str) -> None:
        self.logger.info(f"[{self.get_use_mem_gb():.1f}GB] {msg}")

    def warn(self, msg: str) -> None:
        self.logger.warning(f"[{self.get_use_mem_gb():.1f}GB] {msg}")

    def error(self, msg: str) -> None:
        self.logger.error(f"[{self.get_use_mem_gb():.1f}GB] {msg}")

    def critical(self, msg: str) -> None:
        self.logger.critical(f"[{self.get_use_mem_gb():.1f}GB] {msg}")

    def get_use_mem_gb(self) -> float:
        return self.process.memory_info()[0] / 2.0 ** 30


class CVFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cv, transformer, as_frame: bool = False):
        self.cv = cv
        self.transformer = transformer
        self.as_frame = as_frame

    def fit(self, X, y, cv_target=None, groups=None):
        self._transformers = []
        self._oofs = []
        # self._train = X
        if cv_target is None:
            cv = list(self.cv.split(X, y, groups))
        else:
            cv = list(self.cv.split(X, cv_target, groups))
        for i, (train_idx, val_idx) in enumerate(cv):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            _transformer = copy.copy(self.transformer)
            _transformer.fit(X_train, y_train)
            self._transformers.append(_transformer)
            _oof = _transformer.transform(X_val)
            if i == 0:
                if len(_oof.shape) == 1:
                    self.oof = np.zeros((len(X),))
                else:
                    self.oof = np.zeros((len(X), _oof.shape[1]))
                if self.as_frame:
                    if type(_oof) == pd.Series:
                        self._columns = [_oof.name]
                    else:
                        self._columns = _oof.columns
            self.oof[val_idx] = _oof
        if self.as_frame:
            self.oof = pd.DataFrame(self.oof, columns=self._columns, index=X.index)
        return self

    def transform(self, X, mode=None):
        # if self._train.equals(X):
        if mode == "train":
            return self.oof
        else:
            preds = []
            for transformer in self._transformers:
                preds.append(transformer.transform(X))
            pred = np.mean(preds, axis=0)
            if self.as_frame:
                pred = pd.DataFrame(pred, columns=self._columns, index=X.index)
            return pred
