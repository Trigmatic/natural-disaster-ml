#########################################
# Supervised Regression Machine Learning
#########################################

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error

class RegressionML:
    """This class is used for training 
    supervised regression ML models.
    """

    def __init__(self):
        """Parameter initialization."""
        pass

    def eval_metric_cv(
        self, 
        model, 
        X_train, 
        y_train, 
        cv_fold, 
        model_nm=None
    ):
        """Cross-validation on the training set.

        Parameters
        ----------
        model: supervised regression ML model
        X_train (array): feature matrix of the training set
        y_train (1d array): target variable
        cv_fold (int): number of cross-validation fold

        Returns
        -------
        Performance metrics on the cross-validation training set
        """
        # fit the training set
        model.fit(X_train, y_train)

        # make prediction on k-fold cross validation set
        y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv_fold)

        # print results
        print(f'{str(cv_fold)}-Fold cross-validation results for {str(model_nm)}')
        print('-' * 45)
        print(self.error_metrics(y_train, y_pred_cv))
        print('-' * 45)
        
    def eval_metrics(self, y_pred, y_true, subset=None, model_nm=None):
        """Prediction on the data.

        Parameters
        ----------
        y_pred (1d array): predicted label
        y_true (1d array): ground truth
        subset (str): subset of data
        model_nm (str): model name

        Returns
        -------
        Performance metrics
        """
        # Print results
        print(f'Prediction on the {subset} for {model_nm}')
        print('-' * 45)
        print(self.error_metrics(y_true, y_pred))
        print('-' * 45)

    def error_metrics(self, y_true, y_pred):
        """Print out error metrics.

        Parameters
        ----------
        y_pred (1d array): predicted label
        y_true (1d array)): ground truth

        Returns
        -------
        Performance metrics
        """
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        errors = {
            'MAE' : np.round(mae,3),
            'RMSE' : np.round(rmse,3),
            'R^2' : np.round(r2,3),
        }
        return errors
        
    def diagnostic_plot(self, y_pred, y_true, ylim=None):
        """Diagnostic plot.
        
        Parameters
        ----------
        y_pred (1d array): predicted label
        y_true (1d array): ground truth
        ylim (1d array): y-axis limit

        Returns
        -------
        Matplolib figure
        """
        # compute residual and metrics
        residual = (y_true - y_pred)
        r2 = np.round(r2_score(y_true, y_pred), 3)
        rm = np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
        
        # plot figures
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
        ax1.scatter(y_pred, residual, color='b')
        ax1.set_xlim([-0.1, 20])
        ax1.set_ylim(ylim)
        ax1.hlines(y=0, xmin=-0.1, xmax=20, lw=2, color='k')
        ax1.set_xlabel('Predicted values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs. Predicted values')
        ax2.scatter(y_pred, y_true, color='b')
        ax2.plot([-0.3, 20], [-0.3, 20], color='k')
        ax2.set_xlim([-0.3, 20])
        ax2.set_ylim([-0.3, 20])
        ax2.text(2, 14, f'$R^2 = {str(r2)},~ RMSE = {str(rm)}$', fontsize=20)
        ax2.set_xlabel('Predicted values')
        ax2.set_ylabel('True values')
        ax2.set_title('True values vs. Predicted values')

    def plot_mae_rsme_svr(self, X_train, y_train, cv_fold):
        """Plot of cross-validation MAE and RMSE for SVR.

        Parameters
        ----------
        X_train (array): feature matrix of the training set
        y_train (1d array): target variable
        cv_fold (int): number of cross-validation fold

        Returns
        -------
        matplolib figure of MAE & RMSE
        """
        C_list = [2**x for x in range(-2,11,2)]
        gamma_list = [2**x for x in range(-7,-1,2)]
        mae_list = [
            pd.Series(0.0, 
            index=range(len(C_list))
            ) for _ in range(len(gamma_list)
            )
        ]
        rmse_list = [
            pd.Series(0.0,
            index=range(len(C_list))
            ) for _ in range(len(gamma_list)
            )
        ]
        axes_labels = ['2^-2', '2^0', '2^2', '2^4', '2^6', '2^8', '2^10']
        gamma_labels = ['2^-7', '2^-5', '2^-3']
        plt.rcParams.update({'font.size': 15})
        _, (ax1, ax2) = plt.subplots(1,2, figsize = (18,6))

        for i, val1 in enumerate(gamma_list):
            for j, val2 in enumerate(C_list):
                model = SVR(C = val2, gamma = val1, kernel = 'rbf')
                model.fit(X_train, y_train)
                y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv_fold)
                mae_list[i][j] = mean_absolute_error(y_train, y_pred_cv)
                rmse_list[i][j] = mean_squared_error(y_train, y_pred_cv, squared=False)
            mae_list[i].plot(
                label="gamma="+str(gamma_labels[i]), 
                marker = "o", 
                linestyle="-", ax=ax1
            )
            rmse_list[i].plot(
                label="gamma="+str(gamma_labels[i]), 
                marker="o", 
                linestyle="-", 
                ax=ax2
            )

        ax1.set_xlabel("C", fontsize = 15)
        ax1.set_ylabel("MAE", fontsize = 15)
        ax1.set_title(f"{cv_fold}-Fold Cross-Validation with RBF kernel SVR", 
            fontsize=15)
        ax1.set_xticklabels(axes_labels)
        ax1.set_xticks(range(len(C_list)))
        ax1.legend(loc = 'best')
        ax2.set_xlabel("C", fontsize = 15)
        ax2.set_ylabel("RSME", fontsize = 15)
        ax2.set_title(f"{cv_fold}-Fold Cross-Validation with RBF kernel SVR", 
            fontsize=15)
        ax2.set_xticks(range(len(C_list)))
        ax2.set_xticklabels(axes_labels)
        ax2.legend(loc = 'best')
        plt.show()
