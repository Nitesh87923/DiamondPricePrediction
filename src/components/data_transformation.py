from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import os,sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessing_ob_file=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation_obj(self):
        try:
            logging.info('Data Transformation Started') 
            numerical_col=['carat','depth','table','x','y','z']
            categorical_col=['cut','color','clarity']  
            
            cut_cat=['Fair','Good','Very Good','Premium','Ideal']
            clarity_cat=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            color_cat=['D','E','F','G','H','I','J']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalEncoder',OrdinalEncoder(categories=[cut_cat,color_cat,clarity_cat])),
                    ('scaler',StandardScaler()),
                    
                ]
            )
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_col),
                ('cat_pipeline',cat_pipeline,categorical_col)
            ])

            return preprocessor
            logging.info('Pipeline completed')
        except Exception as e:
            logging.info('Error in Data transformation')
            raise CustomException(e,sys)




    def initiate_data_transformation(self,train_path,test_path):
      try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)
        
        logging.info('train,test files read')
        logging.info(f'Train dataframe Head : \n {train_df.head().to_string()}')
        logging.info(f'Teat dataframe Head : \n {test_df.head().to_string()}')
        logging.info('Obtaining preprocessor obj')

        preprocessor_obj=self.get_data_transformation_obj()
        target_col='price'
        drop_columns=[target_col,'id']
        input_train_df=train_df.drop(columns=drop_columns,axis=1)
        target_train_df=train_df[target_col]
        input_test_df=test_df.drop(columns=drop_columns,axis=1)
        target_test_df=test_df[target_col]
        input_train_processed_arr=preprocessor_obj.fit_transform(input_train_df)
        input_test_processed_arr=preprocessor_obj.transform(input_test_df)
        logging.info('Done with preprocessing')

        train_arr=np.c_[input_train_processed_arr,np.array(target_train_df)]
        test_arr=np.c_[input_test_processed_arr,np.array(target_test_df)]

        save_object(
            file_path=self.data_transformation_config.preprocessing_ob_file,
            obj=preprocessor_obj
        )
        logging.info('Preprocessor obj pickle file created')
        return(
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessing_ob_file
        )
      except Exception as e:
          logging.info('Exception in initiate data transformation')
          raise CustomException(e,sys)


