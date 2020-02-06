import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import operator
import unicodedata
import string
from bs4 import BeautifulSoup

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
# import xgboost as xgb
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.metrics import confusion_matrix

from string import digits
import string
from nltk.corpus import stopwords


import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelBinarizer

def missing_zero_values_table(df):
    ''' Shows the number of missing or zero values in a dataframe.
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


def cleaning(df):
    # df_no_na = df.dropna(subset=['skin_concerns', 'skin_tone', 'skin_type'])
    df['skin_type'] = np.where((df['skin_type'] == 'oily'), 1, 0)
    return df



# use to format strings before putting into model after deciding to stem or not
def format_strings(df, col_name, stemming):

    '''
    Below list of functions are used in the format_strings function
    '''
    punctuation_ = set(string.punctuation)
    stopwords_ = set(stopwords.words('english'))

    def remove_accents(input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        only_ascii = nfkd_form.encode('ASCII', 'ignore')
        #     return only_ascii.decode()
        return nfkd_form

    def strip_html_tags(text):
        """remove html tags from text"""
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    def remove_nums(text):
        '''remove numbers from string'''
        remove_digits = str.maketrans('', '', digits)
        no_digits = text.translate(remove_digits)
        return no_digits

    stemmer = stemming
    df[col_name] = [remove_accents(row) for row in df[col_name]]
    df[col_name] = [remove_nums(row) for row in df[col_name]]
    df[col_name] = [strip_html_tags(row) for row in df[col_name]]
    df[col_name] = df[col_name].replace(r'[^A-Za-z0-9 ]+', '', regex=True)
    df[col_name] = [word_tokenize(row.lower()) for row in df[col_name]]
    df[col_name] = [[word for word in row if not word in stopwords_ and not word in punctuation_]
                   for row in df[col_name]]
    # adjust based on stemming chosen
    df[col_name] = df[col_name].apply(lambda row: ' '.join([stemmer.lemmatize(word) for word in row]))
    
    