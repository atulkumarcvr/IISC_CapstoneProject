#will have all the common functunalities that the project is going to use 
#importing important libraries
import os
import sys
import pickle

import pandas as pd
import numpy as np
import dill

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomeException(e,sys)
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # key = list(models.keys())[i]
            param = params[list(models.keys())[i]]
            
            
            gridsearch = GridSearchCV(model,param,cv=3,error_score='raise')
            gridsearch.fit(X_train, y_train)
            
            #once we get the best parameters we train the model using them
            model.set_params(**gridsearch.best_params_)
            model.fit(X_train,y_train)
            
            # model.fit(X_train, y_train) #training the model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except (FileNotFoundError, EOFError) as e:
        print(f"Error: {e}")
        raise
    except pickle.UnpicklingError as e:
        print("Error unpickling file. Ensure compatibility:", e)
        raise