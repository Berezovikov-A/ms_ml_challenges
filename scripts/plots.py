from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats


def show_distribution(var_data: pd.Series):

    '''
    Function to show summary stats and distribution for a column
    '''

    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    fig, ax = plt.subplots(2, 1, figsize = (15,6))

    sns.histplot(var_data, kde=True, ax=ax[0])
    ax[0].set_ylabel('Frequency')

    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    sns.boxplot(x=var_data, ax=ax[1])

    ax[1].set_xlabel('Value')

    # Add a title to the Figure
    fig.suptitle(var_data.name +': Minimum:{:.2f} | Mean:{:.2f} | Median:{:.2f} | Mode:{:.2f} | Maximum:{:.2f}'.format(min_val,
                                                                                                                        mean_val,
                                                                                                                        med_val,
                                                                                                                        mod_val,
                                                                                                                        max_val))

def show_box(var_data: pd.DataFrame):
    '''
    This function accepts a Pandas dataframe with a label and a feature and shows a box plot to show their relationship
    '''
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data = var_data, y = var_data.columns[0], x = var_data.columns[1], ax=ax)

def show_corr_matrix(var_data: pd.DataFrame):
    '''
    This function accepts a Pandas dataframe and returns a correlation matris of its columns
    '''

    correlation_var = var_data.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlation_var, annot=True, fmt='.1f', vmin=-1, vmax=1, center=0, square=True, cmap='rocket', ax=ax)

def show_correlation(var_data: pd.DataFrame):
    '''
    this function accepts a Pandas dataframe 2 columns and plots a scatter plot to show the correlation
    '''
    col = var_data.columns

    fig, ax = plt.subplots(figsize=(7,7))
    sns.regplot(data=var_data, x=col[1], y=col[0], ax=ax)
