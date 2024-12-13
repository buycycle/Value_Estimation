import pandas as pd
import numpy as np

from typing import Tuple, List

from abc import ABC, abstractmethod

from src.price import predict_price_interval, predict_price
from quantile_forest import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor


class PricePredictionStrategy(ABC):
    @abstractmethod
    def predict_price(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
        pass


class GenericStrategy(PricePredictionStrategy):
    def __init__(self, model, data_transform_pipeline, logger):
        self.model = model
        self.data_transform_pipeline = data_transform_pipeline
        self.logger = logger

    def predict_price(
        self, X: pd.DataFrame, quantiles: List[float] = [0.025, 0.5, 0.975]
    ) -> Tuple[str, np.ndarray, np.ndarray, str]:
        if isinstance(
            self.model, (RandomForestQuantileRegressor, ExtraTreesQuantileRegressor)
        ):
            return predict_price_interval(
                X,
                self.model,
                self.logger,
                quantiles,
            )
        else:
            return predict_price(
                X,
                self.model,
                self.logger,
            )
