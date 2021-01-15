import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from models.base import get_validator
from models.gbdt import LgbModel
from preprocessing import *
from utils import CVFeatureTransformer, Logger
from utils.feature import get_features


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, np.clip(y_pred, 0, 100))


def main():
    logger = Logger(name="train", log_level="INFO", file_log=True)
    logger.info("start")

    train = pd.concat(
        [
            pd.read_pickle(f"../features/{cls.name}_train.pkl")
            for cls in get_features(globals())
        ],
        axis=1,
    )

    logger.info("loaded train features")

    test = pd.concat(
        [
            pd.read_pickle(f"../features/{cls.name}_test.pkl")
            for cls in get_features(globals())
        ],
        axis=1,
    )
    logger.info("loaded test features")

    target = pd.read_pickle("../input/target.pkl")

    categorical_cols = [
        "species",
        "owner",
        "countryof_origin",
        "mill",
        "ico_number",
        "company",
        "region",
        "producer",
        "bag_weight",
        "in_country_partner",
        "harvest_year",
        "grading_date",
        "owner1",
        "variety",
        "processing_method",
        "color",
        "expiration",
        "unit_of_measurement",
    ]

    cv = get_validator("stratified", n_splits=10, random_state=42)

    target_bin = pd.qcut(
        target["total_cup_points"], 50, duplicates="drop", labels=False
    )

    # target encording

    for target_col in target.columns:
        ce_te = ce.TargetEncoder(
            cols=categorical_cols,
            handle_missing="return_nan",
            handle_unknown="return_nan",
        )
        cv_ft = CVFeatureTransformer(cv=cv, transformer=ce_te, as_frame=True)
        cv_ft.fit(
            train.loc[:, categorical_cols], target[target_col], cv_target=target_bin,
        )
        train_te = cv_ft.transform(train.loc[:, categorical_cols]).add_prefix(
            f"te-{target_col}-"
        )
        train = pd.concat([train, train_te], axis=1)
        test_te = cv_ft.transform(test.loc[:, categorical_cols]).add_prefix(
            f"te-{target_col}-"
        )
        test = pd.concat([test, test_te], axis=1)

    logger.info("finish target encording")

    params = {
        "objective": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 33,
        "cat_smooth": 5,
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "random_state": 42,
        "importance_type": "gain",
    }

    fit_params = {
        "early_stopping_rounds": 20,
        "verbose": 500,
        "categorical_feature": categorical_cols,
    }

    fti = pd.DataFrame(index=train.columns, columns=target.columns)
    oof = pd.DataFrame(index=train.index, columns=target.columns)
    pred = pd.DataFrame(index=test.index, columns=target.columns)
    oof_scores = []
    for target_col in target.columns:
        logger.info(f"start train {target_col}")

        model = LgbModel(
            cv=cv,
            eval_metric=mae,
            params=params,
            fit_params=fit_params,
            type_of_target="regression",
            logger=logger,
            cv_target=target_bin,
        )
        model = model.fit(train, target[target_col])
        fti[target_col] = model.get_feature_importance()
        logger.info(model.oof_score)
        logger.info(model.oof_fold_scores)
        oof[target_col] = np.clip(model.oof, 0, 100)
        pred[target_col] = np.clip(model.predict(test), 0, 100)
        oof_scores.append(model.oof_score)

    score = np.mean(oof_scores)
    logger.info(target.columns)
    logger.info(f"oof scores: {oof_scores}")
    logger.info(f"mean mae score: {score}")

    oof.to_csv(f"../output/oof_{score}.csv")
    pred.to_csv(f"../output/sub_{score}.csv", index=False)
    fti.to_csv(f"../output/fti_{score}.csv")

    logger.info("finish")


if __name__ == "__main__":
    main()
