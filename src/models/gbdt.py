import warnings

import lightgbm as lgb
import pandas as pd

from .base import BaseModel

warnings.filterwarnings("ignore")


class GBDTModel(BaseModel):
    def get_feature_importance(self):

        fti = pd.DataFrame(0, index=self.feature_names, columns=["importance"])
        for model in self.models:
            fti["importance"] += model.feature_importances_ / len(self.models)
        fti.sort_values("importance", ascending=False, inplace=True)
        return fti


class LgbModel(GBDTModel):
    def _get_model(self, params):
        if self.type_of_target in ("binary", "multiclass"):
            if params is not None:
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMClassifier()
        else:
            if params is not None:
                model = lgb.LGBMRegressor(**params)
            else:
                model = lgb.LGBMRegressor()

        return model

    def _fit(self, model, X_train, y_train, X_val, y_val):
        model.fit(
            X_train,
            y_train,
            **self.fit_params,
            eval_set=[(X_train, y_train), (X_val, y_val)],
        )
        return model
