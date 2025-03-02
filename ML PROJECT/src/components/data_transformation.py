# import sys 
# from dataclasses import dataclass

# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder,StandardScaler

# from src.exception import CustomException
# from src.logger import logging
# import os
# from src.utils import save_object
# @dataclass

# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.__data_transformation_config=DataTransformationConfig()
#     def get_data_transformer_object(self):# it will convert catogorical features into numericals or to perfom standard scaler for this making pickel file
#         '''tis function is responsible for data transformation'''
#         try:
#             numerical_columns=["writing_score","reading_score"]
#             categorical_feature=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            
#             num_pipeline=Pipeline(# handling missing values and this pipeline will run on taining datset and test set fi tansform and transfom
#                 steps=[
#                 ("imputer",SimpleImputer(strategy="median")),# for missing data
#                 ("scaler",StandardScaler())
#                 ]
#             )   
#             cat_pipeline=Pipeline(# catogorical pipeline
#                 steps=[
#                 ("imputer",SimpleImputer(strategy="most_frequent")),# for missing data
#                 ("one_hot_encoder",OneHotEncoder()),
#                 ("scaler",StandardScaler(with_mean=False))
#                 ]                                       
                
#             )
#             logging.info("Numerical columns standard scaling completed")
#             logging.info("Categorical columns encoding completed")
#             # combining the pipelines numerical and catogorical
            
#             preprocessor=ColumnTransformer(
#                 [
#                 ("num_pipeline",num_pipeline,numerical_columns),
#                 ("cat_pipeline",cat_pipeline,categorical_feature)
#                 ]
#             )
            
#             return preprocessor
        
#         except Exception as e:
#             raise CustomException(e,sys)
#         # starting our data transformation techniques
#     def initiate_data_transformation(self,train_path,test_path):
#         try:
#             train_df=pd.read_csv(train_path)
#             test_df=pd.read_csv(test_path)
            
#             logging.info("read train and test data completely")
#             logging.info("obtaining preprocessing object")
            
#             preprocessing_obj=self.get_data_transformer_object()
#             target_column_name="math_score" 
#             numerical_coloumns=["writing_score","reading_score"]
            
#             input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
#             target_feature_train_df=train_df[target_column_name]

#             input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
#             target_feature_test_df=test_df[target_column_name]

#             logging.info(
#                 f"Applying preprocessing object on training dataframe and testing dataframe."
#             )

#             input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

#             train_arr = np.c_[
#                 input_feature_train_arr, np.array(target_feature_train_df)
#             ]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             logging.info(f"Saved preprocessing object.")

#             save_object(

#                 file_path=self.__data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj

#             )

#             return (
#                 train_arr,
#                 test_arr,
#                 self.__data_transformation_config.preprocessor_obj_file_path,
#             )
#         except Exception as e:
#             raise CustomException(e,sys)
                
import sys 
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.__data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """Function responsible for data transformation"""
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            # Numerical Pipeline
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical Pipeline
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))  # Prevents error with sparse matrices
            ])

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            # Combine Pipelines
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]  # Fixed spelling

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")
            

            save_object(
                file_path=self.__data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                # "artifacts/test1.pkl", obj= preprocessing_obj
            )
            

            return (
                train_arr,
                test_arr,
                self.__data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
     