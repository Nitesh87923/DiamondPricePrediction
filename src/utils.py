import os,sys
import pickle
import numpy as np
import pandas as pd
from src. logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)        
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
          model=list(models.values())[i]
          model.fit(x_train,y_train)
          y_pred=model.predict(x_test)
          model_score=r2_score(y_test,y_pred)
          report[list(models.keys())[i]]=model_score
        return report
    except Exception as e:
        logging.info('Exception during model training')
        raise CustomException(e,sys)  
        
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception while loading pickle file')
        raise CustomException(e,sys)     
        
