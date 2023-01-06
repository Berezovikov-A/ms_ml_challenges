from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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
