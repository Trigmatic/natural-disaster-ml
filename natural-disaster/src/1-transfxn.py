#################################
# Transformation Pipeline
#################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import  ColumnTransformer

class TransformationPipeline:
    """This class is used for data preprocessing 
    and transformations.
    """

    def __init__(self):
        """Define parameters."""
        pass

    def num_pipeline(self, X_train, X_test):
        """Transformation pipeline of data with 
        only numerical variables.

        Parameters
        ----------
        X_train: training feature matrix
        X_test: test feature matrix

        Returns
        -------
        Transformation pipeline and transformed data in array
        """
        # Create pipeline
        num_pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='median')),
                ('p_transf', PowerTransformer(standardize=False)),
                ('std_scaler', StandardScaler()),
            ]
        )

        # Original numerical feature names 
        feat_nm = list(X_train.select_dtypes('number'))

        # Fit transform the training set and transform the test set
        X_train_scaled = num_pipeline.fit_transform(X_train)
        X_test_scaled = num_pipeline.transform(X_test)
        return X_train_scaled, X_test_scaled, feat_nm
    
    def cat_encoder(self, X_train, X_test):
        """Transformation pipeline of categorical variables.

        Parameters
        ----------
        X_train: training feature matrix
        X_test: test feature matrix

        Returns
        ------
        Transformation pipeline and transformed data in array
        """
        # Create pipeline
        cat_pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
            ]
        )

        # fit transform the training set and transform the test set
        X_train_scaled = cat_pipeline.fit_transform(X_train)
        X_test_scaled = cat_pipeline.transform(X_test)

        # feature names for output features
        raw_cat_attribs = list(X_train.select_dtypes('O'))
        feat_nm = list(cat_pipeline.get_feature_names_out(raw_cat_attribs))
        return X_train_scaled.toarray(), X_test_scaled.toarray(), feat_nm

    def preprocessing(self, X_train, X_test):
        """Transformation pipeline of data with 
        both numerical and categorical variables.

        Parameters
        ----------
        X_train: training feature matrix
        X_test: test feature matrix

        Returns
        -------
        Transformed data in array
        """
        # numerical transformation pipepline
        num_train, num_test, num_attribs = self.num_pipeline(
            X_train.select_dtypes('number'), 
            X_test.select_dtypes('number'),
        )

        # categorical transformation pipepline
        cat_train, cat_test, cat_attribs = self.cat_encoder(
            X_train.select_dtypes('O'), 
            X_test.select_dtypes('O'),
        )

        # transformed training and test sets
        X_train_scaled = np.concatenate((num_train, cat_train), axis=1)
        X_test_scaled = np.concatenate((num_test, cat_test), axis=1)

        # feature names
        feat_nm = num_attribs + cat_attribs
        return X_train_scaled, X_test_scaled, feat_nm


    def preprocessing_2(self, X_train, X_test):
        """This preprocessing uses scikit-learn ColumnTransformer class."""
        # Create pipelines
        num_pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='median')),
                ('p_transf', PowerTransformer(standardize=False)),
                ('std_scaler', StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
            ]
        )

        cat_attribs = list(X_train.select_dtypes('O'))
        num_attribs = list(X_train.select_dtypes('number'))
        combo_pipeline = ColumnTransformer(
            [
                ('num', num_pipeline, num_attribs),
                ('cat', cat_pipeline, cat_attribs),
        
            ]
        )

        # fit transform the training set and transform the test set
        X_train_scaled = combo_pipeline.fit_transform(X_train)
        X_test_scaled = combo_pipeline.transform(X_test)
        feat_nms = list(combo_pipeline.get_feature_names_out())
        return X_train_scaled, X_test_scaled, feat_nms
