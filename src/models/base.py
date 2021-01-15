import warnings
from abc import ABCMeta, abstractmethod
from inspect import signature
from logging import DEBUG, StreamHandler, getLogger

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.utils import multiclass as skum

from .split import StratifiedGroupKFold

_logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
_logger.setLevel(DEBUG)
_logger.addHandler(handler)
_logger.propagate = False

warnings.filterwarnings("ignore")


class BaseModel(BaseEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        cv,
        eval_metric,
        groups=None,
        params=None,
        fit_params=None,
        type_of_target="auto",
        target_log_flg=False,
        cv_target=None,
        logger=None,
        seeds=None,
    ):
        try:
            self.logger = logger.getChild(self.__class__.__name__)
        except BaseException:
            self.logger = _logger

        self.cv = cv
        self.eval_metric = eval_metric
        self.groups = groups

        self.params = params
        self.fit_params = fit_params
        self.target_log_flg = target_log_flg
        self.cv_target = cv_target

        self.type_of_target = type_of_target
        if seeds is not None:
            assert type(seeds) == list
        self.seeds = seeds

    def fit(self, X, y):

        if type(X) == pd.DataFrame:
            self.feature_names = X.columns

        if self.type_of_target == "auto":
            self.type_of_target = skum.type_of_target(y)

        if self.type_of_target in ("multiclass"):
            self.num_class = int(max(y)) + 1

        self.models = []
        self.oof_fold_scores = []

        if (self.type_of_target == "regression") & self.target_log_flg:
            y_log = np.log1p(y)
            self.oof = np.zeros((len(X),))
            self._cross_validate(X, y_log)
            self.oof_score = self.eval_metric(y, self.oof)
        else:
            if self.type_of_target in ["multiclass"]:
                self.oof = np.zeros((len(X), self.num_class))
            else:
                self.oof = np.zeros((len(X),))

            self._cross_validate(X, y)

            if self.type_of_target in ["multiclass"]:
                self.oof_score = self.eval_metric(y, np.argmax(self.oof, axis=1))
            else:
                self.oof_score = self.eval_metric(y, self.oof)

        self.logger.info(f"oof {self.eval_metric.__name__}: {self.oof_score}")

        return self

    def predict(self, X):

        if self.type_of_target in ("multiclass"):
            self.pred = np.zeros((len(X), self.num_class))
        else:
            self.pred = np.zeros((len(X),))

        for model in self.models:
            if (self.type_of_target == "regression") & self.target_log_flg:
                self.pred += np.expm1(self._predict(model, X)) / len(self.models)
            else:
                self.pred += self._predict(model, X) / len(self.models)

        return self.pred

    def _predict(self, model, X):
        if self.type_of_target in ("binary", "multiclass"):
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
            elif hasattr(model, "decision_function"):
                self.logger.warning(
                    (
                        f"Since {type(model)} does not have"
                        "predict_proba method,decision_function"
                        " is used for the prediction instead."
                    )
                )
                proba = model.decision_function(X)
            else:
                raise RuntimeError(
                    (
                        "Estimator in classification problem should have either"
                        " predict_proba or decision_function"
                    )
                )
            if proba.ndim == 1:
                return proba
            else:
                return proba[:, 1] if proba.shape[1] == 2 else proba
        else:
            return model.predict(X)

    def _cross_validate(self, X, y):

        if self.cv_target is None:
            cv = list(self.cv.split(X, y, self.groups))
        else:
            cv = list(self.cv.split(X, self.cv_target, self.groups))

        for i, (train_idx, val_idx) in enumerate(cv):
            if type(X) == pd.DataFrame:
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            else:
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

            if self.seeds is None:
                model = self._get_model(self.params)
                model = self._fit(model, X_train, y_train, X_val, y_val)

                _oof = self._predict(model, X_val)
                if (self.type_of_target == "regression") & self.target_log_flg:
                    _oof = np.expm1(_oof)
                self.oof[val_idx] = _oof

                self.models.append(model)
            else:
                if self.type_of_target in ("multiclass"):
                    _oof = np.zeros((len(X_val), self.num_class))
                else:
                    _oof = np.zeros(len(X_val),)
                for seed in self.seeds:
                    model = self._get_model(self.params)
                    model.set_params(random_state=seed)
                    self.logger.info(f"set seed {seed}")
                    model = self._fit(model, X_train, y_train, X_val, y_val)
                    _oof_seed = self._predict(model, X_val)

                    if (self.type_of_target == "regression") & self.target_log_flg:
                        _oof_seed = np.expm1(_oof_seed)
                        _oof_seed_score = self.eval_metric(np.expm1(y_val), _oof_seed)
                    else:
                        _oof_seed_score = self.eval_metric(y_val, _oof_seed)

                    self.logger.info(
                        f"{i+1} fold {seed} seed {self.eval_metric.__name__}: {_oof_seed_score}"
                    )
                    _oof += _oof_seed / len(self.seeds)
                    self.models.append(model)

                self.oof[val_idx] = _oof

            if (self.type_of_target == "regression") & self.target_log_flg:
                _oof_fold_score = self.eval_metric(np.expm1(y_val), _oof)
            elif self.type_of_target in ["multiclass"]:
                _oof_fold_score = self.eval_metric(y_val, np.argmax(_oof, axis=1))
            else:
                _oof_fold_score = self.eval_metric(y_val, _oof)

            self.oof_fold_scores.append(_oof_fold_score)
            self.logger.info(
                f"{i+1} fold {self.eval_metric.__name__}: {_oof_fold_score}"
            )

    def _fit(self, model, X_train, y_train, X_val, y_val):
        if signature(model.fit).parameters.keys() == {"X", "y"}:
            model.fit(X_train, y_train)
        elif self.fit_params is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, self.fit_params)
        return model

    @abstractmethod
    def _get_model(self, params):
        raise NotImplementedError


def get_validator(validator_name, n_splits=5, shuffle=True, random_state=42):

    if validator_name == "kfold":
        validator = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    elif validator_name == "stratified":
        validator = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
    elif validator_name == "group":
        validator = GroupKFold(n_splits=n_splits)
    elif validator_name == "stratifiedgroup":
        validator = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
    elif validator_name == "timeseries":
        validator = TimeSeriesSplit(n_splits=n_splits)
    else:
        raise ValueError(
            (
                "Invalid validator:"
                "you must provide in [kfold, stratified , group , stratifiedgroup, timeseries]"
            )
        )
    return validator
