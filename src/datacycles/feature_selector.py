
import pandas as pd
from mlmodels.base_model import BaseModel
from typing import Tuple

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


class FeatureSelector:


    def __init__(self, model: BaseModel) -> None:
        pass

    
    def run_fs_report(self):
        pass


    def __calculate_classification_metrics(
        self,
        model, 
        y_real, 
        features
        ) -> Tuple:
        """Calculate performance metrics of classification algorithm
        """

        
        model_predictions = model.predict(features)
        
        
        roc  = roc_auc_score(y_real, model_predictions)
        acc  = accuracy_score(y_real, model_predictions)
        prec = precision_score(y_real, model_predictions)
        rec  = recall_score(y_real, model_predictions)
        f1   = f1_score(y_real, model_predictions)
        
        return acc, roc, prec, rec, f1