from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def show_distribution (var: pd.Series):
    '''
    This function accepts a pandas series and shows distribution of values
    '''
    #find min, max, average, median and mode:
    min_val = var.min()
    max_val = var.max()
    ave_val = var.mean()
    med_val = var.median()
    mod_val = var.mode()[0]

    #create a figure for 2 plots 2 rows 1 column size 18 by 5
    fig, ax = plt.subplots(2,1,figsize = (18, 5))

    #plot histogram
    ax[0].hist(var)
    ax[0].set_ylabel('Frequency')

    #lines for min, max, average, median and mode
    ax[0].axvline(x=min_val, color='yellow', linestyle = 'dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color='grey', linestyle = 'dashed', linewidth = 2)
    ax[0].axvline(x=ave_val, color='red', linestyle = 'dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color='cyan', linestyle = 'dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color='blue', linestyle = 'dashed', linewidth = 2)

    #plot the box figure
    ax[1].boxplot(var, vert=False)
    ax[1].set_xlabel('value')

    #set the title
    fig.suptitle('Data Distribution for {} Minimum = {:.2f} Average = {:.2f} Median = {:.2f} Mode = {:.2f} Maximum = {:.2f}'.format(var.name,min_val, ave_val, med_val, mod_val, max_val))

def show_box(df: pd.DataFrame, feature: str, label: str):
    '''
    This function accepts a Pandas dataframe and column names for a categorical feature and label and returns a box plot to show their relationship
    '''
    df.boxplot(column=label, by=feature, figsize=(8,8))
    plt.xticks(rotation=90)

def show_correlation(label: pd.Series, feature: pd.Series):
    '''
    This function accepts 2 Pandas series and shows their correlation as a scatter plot and a line demonstrating linear regression
    '''
    fig = plt.figure(figsize=(9,6))
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
