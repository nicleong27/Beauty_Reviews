
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



class EDA(object):
    '''EDA class containing multiple functions that help visualize the dataset'''

    def __init__(self):
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
        ''' Groups by a specific column and plots the count of those categories in a bar graph.

        Parameters
        ----------
        df: pandas dataframe
        group: str
            Column name that is to be grouped by
        count_by: str
            Column name whose values are to be counted by

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
        x : pandas series
            Feature who values are to be plotted against y or predicted label
        y : pandas series
            Label to be predicted
        y0_name : str
            Y value that is equal to 0
        y1_name : str
            Y value that is equal to 1

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

    def create_wordcloud(self, df, col_name):
        '''
        Creates wordcloud based on dataframe column name.

        Parameters
        ----------
        df: pandas dataframe
        col_name : str 
            Text column name looking to plot

        Returns:
        --------
        None
        '''
        text = ' '.join(review for review in df[col_name])
        wordcloud = WordCloud(background_color='white').generate(text)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation='bilinear')

    def w_freq(self, text_series):
        '''
        Maps bag of words with count frequencies and sorts these from 
        most to least frequent.

        Parameters
        ----------
        text_series : pandas series

        Returns:
        --------
        words_freq : lst
            List of tuples 
        '''
        countvect = CountVectorizer()
        bag_of_words = countvect.fit_transform(text_series)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in countvect.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq

    def top_10_word_freq(self, word_lst, word_count):
        fig, ax = plt.subplots(figsize=(10,7))
        ax.bar(word_lst[:10], word_count[:10], color='midnightblue')
        ax.set_xlabel('Words', fontsize=16)
        ax.set_ylabel('Count', fontsize=16)
        plt.xticks(rotation=0)
        ax.tick_params(axis='both', labelsize=16)
        plt.title('Top 10 Word Counts', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
