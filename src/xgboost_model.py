"""Premade XGBClassifier training function is located here."""
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from typing import List

MODEL_VALIDATION_SET_SIZE = 0.20
MODEL_VALIDATION_SPLIT_RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50
VERBOSITY_ROUNDS = 50

def model_params():
    """Returns the parameters for the XGBClassifier."""
    params = {
        'eta': 0.01, 
        'gamma': 0.1, 
        'max_depth': 6, 
        'subsample': 0.2, 
        'colsample_bytree': 1, 
        'objective': 'binary:logistic', 
        'base_score': 0.5, 
        'eval_metric': 'aucpr', 
        'seed': 42, 
        'min_child_weight': 2, 
        'reg_alpha': 2, 
        'importance_type': 'gain', 
        'n_estimators': 221
        }
    return params


def fit_model(features, targets, params=None):
    """Fits a model of XGB with given features and targets."""
    if not params:
        params = model_params()
   
    # Generates the actual train set and a validation set to determine the number of trees
    X_train, X_validation, y_train, y_validation = train_test_split(
        features,
        targets,
        test_size=MODEL_VALIDATION_SET_SIZE,
        random_state=MODEL_VALIDATION_SPLIT_RANDOM_STATE,
        stratify=targets,
    )

    # FIRST MODEL #
    # Parametrize a watch list to follow the training performance
    watch_list = [(X_train, y_train), (X_validation, y_validation)]

    # Fit first a XGBoost booster to determine the best number of rounds...
    # Defining first model with params and training balance.
    first_model = XGBClassifier(
        **params,
        scale_pos_weight=(
            y_train.value_counts().sort_index()[0]
            / y_train.value_counts().sort_index()[1]
        ),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    
    # Fitting first model with 2 eval set elements which the first one is training itself.
    print("\n	Training Performance		Validation Performance")
    print("	--------------------		----------------------")
    first_model = first_model.fit(
        X_train, y_train, eval_set=watch_list, verbose=VERBOSITY_ROUNDS
    )

    # Best iteration spot
    params["n_estimators"] = first_model.best_iteration
    print("Working with parameters:", params)

    final_model = XGBClassifier(
        **params,
        scale_pos_weight=(
            targets.value_counts().sort_index()[0]
            / targets.value_counts().sort_index()[1]
        ),
    )
    final_model = final_model.fit(features, targets)

    print("Done !")
    return final_model, params

# functions to split train and test 
def oot_train_test_split(df, 
                    features_to_drop:List[str] = None,
                    top_features:List[str] = None, 
                    oot=False):

    
    ref_date_col = "year"
        
    if top_features:
        train_df = df[top_features]
    else:
        train_df = df
            
    # train_df[ref_date_col] = pd.to_datetime(train_df[ref_date_col], errors="coerce")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.rename(columns={"target": "TARGET"})
    
    if features_to_drop: 
        train_df.drop(features_to_drop, axis=1, inplace=True)
    else: 
        train_df = train_df
    
    # train test separation
    if oot:
        train = train_df[train_df[ref_date_col] < oot]
        test = train_df[train_df[ref_date_col] >= oot]
        X_train = train.drop(['TARGET'], axis=1)
        y_train = train['TARGET']
        X_test = test.drop(['TARGET'], axis=1)
        y_test = test['TARGET']
    else:
        X = train_df.drop(['TARGET'], axis=1)
        y = train_df['TARGET']
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.25,
                                                            stratify=y,
                                                            random_state=42)
        

    print(f"Training vs Validation size: {X_train.shape[0]} - {X_test.shape[0]}")
    print("Training set Class distribution:")
    print((y_train.value_counts()*100.0/len(y_train)).round(1))
    print("Test set Class distribution:")
    print((y_test.value_counts()*100.0/len(y_test)).round(1))
   
    return X_train, X_test, y_train, y_test

# functions to get dummies for categorical variables
def get_dummies_cols(X_train, X_test, cols:List[str]):
    train_dummies = pd.get_dummies(X_train[cols], drop_first=True)
    test_dummies = pd.get_dummies(X_test[cols], drop_first=True)

    # Ensure the test set has the same dummy variables as the training set
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    
    X_train = pd.concat([X_train.drop(cols, axis=1), train_dummies], axis=1)
    X_test = pd.concat([X_test.drop(cols, axis=1), test_dummies], axis=1)
    return X_train, X_test


