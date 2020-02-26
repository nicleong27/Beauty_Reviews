
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EDA(object):
    '''EDA class containing multiple functions needed for EDA'''

    def __init__(self):
        '''Initialize with dataframe '''
        pass

    def missing_zero_values_table(self, df):
        ''' Returns a dataframe showing the number of missing or zero values in the input dataframe.

        Parameters
        ----------
        df: pandas dataframe 

        Returns:
        --------
        mz_table: pandas dataframe
        '''
        zero_val = (df == 0.00).astype(int).sum(axis=0)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
        mz_table['Data Type'] = df.dtypes
        mz_table = mz_table[
            mz_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " + str(mz_table.shape[0]) +
                " columns that have missing values.")
    #         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
        return mz_table

    def plot_skin(self, df, group, count_by):
        ''' Plots a bar graph by grouping by a specific column and plotting the count of those categories.

        Parameters
        ----------
        df: pandas dataframe
        group: str
        count_by: str

        Returns:
        --------
        None
        '''
        fig, ax = plt.subplots()
        skin_type_groups = df.groupby(group).count()[count_by]
        skin_type_groups.sort_values(ascending=False).plot(kind='bar', figsize=(10,7), color='midnightblue')
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='both', labelsize=10)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    
    def plot_histograms_by_outcome(self, x, y, y0_name, y1_name):
        ''' Plots histogram distribution by outcome categories.

        Parameters
        ----------
        x = pandas series
        y = pandas series
        y0_name = str
        y1_name = str

        Returns:
        --------
        None
        '''
        fig, ax = plt.subplots(figsize=(10,7))
        plt.hist(list(x[y==0]), alpha=0.5, label='{}=0'.format(y0_name), color='blue')
        plt.hist(list(x[y==1]), alpha=0.5, label='{}=1'.format(y1_name), color='midnightblue')
        plt.title('Histogram of {}'.format(x.name), fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
