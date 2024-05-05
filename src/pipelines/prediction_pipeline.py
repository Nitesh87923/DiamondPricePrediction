import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)

            return pred
        except Exception as e:
           logging.info('Exception for current input')   
           raise CustomException(e,sys)

class CustomData:
    def __init__(
            self,
            carat:float,
            depth:float,
            table:float,
            x:float,
            y:float,
            z:float,
            cut:str,
            color:str,
            clarity:str):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.clarity=clarity
        self.color=color
    def get_data_as_df(self):
        try:
            custon_data_dict={
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'clarity':[self.clarity],
                'color':[self.color]

            }  
            df=pd.DataFrame(custon_data_dict)
            return df 
        except Exception as e:
           logging.info('Exception in custom data')   
           raise CustomException(e,sys)   
                