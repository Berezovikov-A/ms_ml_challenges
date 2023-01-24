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

    sns.histplot(var_data, binwidth = abs(mean_val), kde=True, ax=ax[0])
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
    fig = plt.figure(figsize=(10,6))
    
    fig = sns.boxplot(data = var_data, y = var_data.columns[0], x = var_data.columns[1])

def show_correlation(label: pd.Series, feature: pd.Series):
    '''
    This function accepts 2 Pandas series and shows their correlation as a scatter plot and a line demonstrating linear regression
    '''
    fig = plt.figure(figsize=(9,9))
    ax = fig.gca()
    correlation = feature.corr(label)

    # regression slope and intercept
    slope, intercept, r, p, std_err = stats.linregress(feature, label)
    #calculate f(x)
    df_reg = pd.Series(dtype='object')
    df_reg ['fx']= (slope * feature) + intercept
    
    plt.scatter(x=feature, y=label)
    plt.xlabel(feature.name)
    plt.ylabel(label.name)
    plt.title(label.name + ' vs ' + feature.name + ' - correlation:' + str(correlation) + '\nslope: {:.4f}\ny-intercept: {:.4f}'.format(slope,intercept))
    plt.plot(feature, df_reg['fx'], color = 'red')

def show_correlation_heatmap(df: pd.DataFrame):
    '''
    This function accepts a dataset with numeric values and transforms it to a correlation matrix in form of a heatmap
    '''
    corr_matrix = round(df.corr(), 2)

    fig, ax = plt.subplots(figsize=(9,6))

    ax.imshow(corr_matrix)
    ax.set_xticks(range(len(corr_matrix)), corr_matrix.columns)
    ax.set_yticks(range(len(corr_matrix)), corr_matrix.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    for i, ind in enumerate(corr_matrix.index):
        for j, col in enumerate(corr_matrix.columns):
            ax.text(j, i, corr_matrix.loc[ind, col],
                        ha="center", va="center", color="w")
