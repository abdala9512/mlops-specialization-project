
import pandas as pd
import numpy as np
from mlmodels.base_model import BaseModel
from typing import Tuple, Callable, List

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    RFE, SelectKBest, SelectFromModel, chi2, f_classif
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score
)

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureSelector:


    def __init__(self,
     dataset: pd.DataFrame,
     model: BaseModel,
     target: str = None
    ) -> None:
     """FeatureSelector initizalization"""
     self.model = model
     self.data = dataset
     self.target = target
     self.X = self.data.drop(self.target, axis=1)
     self.Y = self.data[target]
     (self.X_train,
      self.X_test, 
      self.Y_train, 
      self.Y_test) = self.__split_data()

    
    def run_fs_report(self):
        pass
    
    def __select_kbest_features(self, scorer: Callable, k: int) -> np.ndarray:
        """Select K best features
        """

        selector = SelectKBest(scorer, k=k)
        selector.fit_transform(self.X, self.y)

        feature_idx = selector.get_support()

        return self.data.drop(self.target).columns[feature_idx]
    
    def __recursive_feature_elimination(self, top: int):
        """Recursive Feature Elimination method"""
        X_train_scaled, X_test_scaled = self.__scale_data()

        rfe = RFE(self.model, top)
        rfe.fit(self.X_train, self.Y_train)
        feature_idx = rfe.get_support()

        return self.data.drop(self.target,axis=1).columns[]


    def __calculate_classification_metrics(
        self,
        y_real, 
        features
        ) -> Tuple:
        """Calculate performance metrics of classification algorithm
        """

        
        model_predictions = self.model.predict(features)
        
        
        roc  = roc_auc_score(y_real, model_predictions)
        acc  = accuracy_score(y_real, model_predictions)
        prec = precision_score(y_real, model_predictions)
        rec  = recall_score(y_real, model_predictions)
        f1   = f1_score(y_real, model_predictions)
        
        return acc, roc, prec, rec, f1

    def __split_data(self) -> tuple:
        """Split into train and test the data"""
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, 
            self.Y, 
            test_size = 0.2,
            stratify=self.Y, 
            random_state = 123
         )

        return X_train, X_test, Y_train, Y_test

    def __scale_data(self) -> Tuple:
        """Data Standardization"""

        scaler = StandardScaler()

        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        return X_train_scaled, X_test_scaled