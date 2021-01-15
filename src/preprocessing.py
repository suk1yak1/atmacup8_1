from dataclasses import dataclass

import category_encoders as ce
import pandas as pd

from utils import Logger
from utils.feature import Feature, generate_features

Feature.dir = "../features"


@dataclass
class OriginalFeature(Feature):
    def create_feature(self, train, test):
        original_features = [
            "numberof_bags",
            "quakers",
            "moisture",
            "category_one_defects",
            "category_two_defects",
            "altitude_low_meters",
            "altitude_high_meters",
            "altitude_mean_meters",
        ]
        for self_df, df in zip([self.train, self.test], [train, test]):
            self_df[original_features] = df.loc[:, original_features]
        self.description = "そのままの特徴量"


@dataclass
class CategoricalFeature(Feature):
    def create_feature(self, train, test):
        categorical_features = [
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
        ce_oe = ce.OrdinalEncoder(
            categorical_features,
            handle_unknown="return_nan",
            handle_missing="return_nan",
        )
        ce_oe.fit(train.loc[:, categorical_features])

        for self_df, df in zip([self.train, self.test], [train, test]):
            self_df[categorical_features] = ce_oe.transform(
                df.loc[:, categorical_features]
            )
        self.description = "OrdinalEncoder"


def preprocessing(overwrite=False):
    logger = Logger(name="preprocessing", log_level="INFO", file_log=False)
    logger.info("start")

    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

    target_cols = list(set(train.columns) - set(test.columns))

    target = train.loc[:, target_cols]
    target.to_pickle("../input/target.pkl")

    logger.info("save target")

    generate_features(globals(), train, test, overwrite)

    logger.info("finish")


if __name__ == "__main__":
    preprocessing()
