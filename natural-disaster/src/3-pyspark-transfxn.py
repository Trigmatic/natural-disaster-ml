#################################
# PySpark transformations
#################################

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from sklearn.metrics import mean_squared_error, r2_score

class TransformationPipeline:
    """This class is used for transformation pipelines in PySpark."""

    def __init__(self, label_col):
        """Define parameters."""
        self.label_col = label_col

    def one_val_imputer(self, df, cols, impute_with):
        """Impute column(s) with one specific value.

        Parameters
        ----------
        df: spark dataframe
        num_cols: list of column name(s)
        impute_with: imputation value

        Returns
        --------
        Dataframe with imputed column(s) 
        """
        df = df.fillna(impute_with, subset=cols)
        return df
    
    def df_to_numeric(self, df, dont_cols):
        """Convert numerical columns to float type."""
        cols = [x for x in df.columns if x not in dont_cols]
        for col in cols:
            df = df.withColumn(col, df[col].cast(FloatType()))
        return df
        
    def preprocessing(self, trainDF, validDF, testDF):
        """Data preprocessing steps involving the following transformations:

        1. one-hot encode categorical variables
        2. impute missing values in numerical variables
        3. standardize numerical variables

        Parameters
        ----------
        trainDF: training data set
        validDF: test data set
        testDF: test data set
        label_col: column name for the labels or target variable

        Returns
        -------
        Transformed training and test data sets with the assembler vector
        """
        # extract numerical and categorical column names
        cat_cols = [
            field for (field, dataType) 
            in trainDF.dtypes if dataType=="string"
            ]
        num_cols = [
            field for (field, dataType) in 
            trainDF.dtypes if ((dataType == "double") & \
                (field != self.label_col))
            ]

        # create output columns
        index_output_cols = [x + "Index" for x in cat_cols]
        ohe_output_cols = [x + "OHE" for x in cat_cols]
        # num_output_cols = [x + "scaled" for x in num_cols]

        # strinf indexer for categorical variables
        s_indexer = StringIndexer(
            inputCols=cat_cols, 
            outputCols=index_output_cols, 
            handleInvalid="skip"
        )

        # one-hot code categorical columns
        cat_encoder = OneHotEncoder(
            inputCols=index_output_cols, 
            outputCols=ohe_output_cols
        )

        # impute missing values in numerical columns
        num_imputer = Imputer(inputCols = num_cols, outputCols = num_cols)

        # vector assembler
        assembler_inputs = ohe_output_cols + num_cols
        assembler = VectorAssembler(
            inputCols=assembler_inputs, 
            outputCol="unscaled_features"
        )

        # features scaling using StandardScaler
        scaler = StandardScaler(
            inputCol=assembler.getOutputCol(), 
            outputCol="features"
        )
        
        # create pipeline
        stages = [s_indexer, cat_encoder, num_imputer, assembler, scaler]
        pipeline = Pipeline(stages = stages)
        pipelineModel = pipeline.fit(trainDF)

        # preprocess training and test data
        trainDF_scaled = pipelineModel.transform(trainDF)
        validDF_scaled = pipelineModel.transform(validDF)
        testDF_scaled = pipelineModel.transform(testDF)
        return assembler, trainDF_scaled, validDF_scaled, testDF_scaled

    def eval_metrics(self, model_pred, model_nm):
        """Print regression evaluation metrics.

        Parameters
        -----------
        model_pred: model prediction
        model_nm: name of the model
        label_col: column name for the labels or target variable

        Returns
        -----------
        Print metrics
        """
        eval = RegressionEvaluator(
            predictionCol="prediction", 
            labelCol=self.label_col,
            metricName="rmse"
        )

        rmse = eval.evaluate(model_pred)
        mse = eval.evaluate(model_pred, {eval.metricName: "mse"})
        mae = eval.evaluate(model_pred, {eval.metricName: "mae"})
        r2 = eval.evaluate(model_pred, {eval.metricName: "r2"})
        result = {'MAE = {}'.format(
            np.round(mae,3)
            ), 
            'MSE = {}'.format(np.round(mse,3)
        ),
        'RMSE = {}'.format(np.round(rmse,3)), 
        'R^2 = {}'.format(np.round(r2,3)
        )
        }
        print(f"Performance metrics for {str(model_nm)}")
        print('-'*40)
        print(result)

    def diagnostic_plot(self, y_test, y_pred):
        """Diagnostic plot.

        Parameters
        ----------
        y_test: ground truth
        y_train: model prediction

        Returns
        -------
        Matplolib figure
        """
        # compute residual and metrics
        residual = (y_test - y_pred)
        r2 = np.round(r2_score(y_test, y_pred), 3)
        rm = np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)

        # plot figures
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        ax1.scatter(y_pred, residual, color ='b')
        ax1.set_xlim([-0.1, 15])
        ax1.hlines(y=0, xmin=-0.1, xmax=15, lw=2, color='k')
        ax1.set_xlabel('Predicted values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs. Predicted values')
        ax2.scatter(y_pred, y_test, color='b')
        ax2.plot([-0.3, 15], [-0.3, 15], color='k')
        ax2.set_xlim([-0.3, 15])
        ax2.set_ylim([-0.3, 15])
        ax2.text(6, 2, r'$R^2 = {},~ RMSE = {}$'.format(
            str(r2), 
            str(rm)
            ), 
            fontsize=20
        )
        ax2.set_xlabel('Predicted values')
        ax2.set_ylabel('True values')
        ax2.set_title('True values vs. Predicted values')
    
