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

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns


class Prepocessing(object):
    ''' Preprocessing class that will process and clean data, so that
    data can be fitted to the model(s).
    '''

    def __init__(self):
        pass

    def oily_dry_skin(self, df):
        '''
        Replaces oily and dry classes with 1's and 0's.

        Parameters
        ----------
        df: pandas dataframe

        Returns:
        --------
        df: pandas dataframe
        '''
        # df_no_na = df.dropna(subset=['skin_concerns', 'skin_tone', 'skin_type'])
        df['skin_type'] = np.where((df['skin_type'] == 'oily'), 1, 0)
        return df 

    def preprocess_vect(self, df, col_name, vectorizer):
        '''
        Preprocess/vectorize dataframe, so that the dataframe can be passed into the model

        Parameters
        ----------
        df: pandas dataframe
        col_name: column name string
        vectorizer: vectorizer function

        Returns:
        --------
        nlp_df: pandas dataframe
        '''
        vect = vectorizer.fit_transform(df[col_name]).todense()
        bow = vectorizer.get_feature_names()
        nlp_df = pd.DataFrame(vect, columns=bow)
        return nlp_df

    def remove_accents(self, input_str):
        ''' Removes accents from string.

        Parameters
        ----------
        input_str: string

        Returns:
        --------
        nfkd_form: string without accents
        '''
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        only_ascii = nfkd_form.encode('ASCII', 'ignore')
        #     return only_ascii.decode()
        return nfkd_form


    def strip_html_tags(self,text):
        '''Removes html tags from string

        Parameters
        ----------
        text: string

        Returns:
        --------
        stripped_text: string without html tags
        
        '''
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    def remove_nums(self, text):
        '''Removes numbers from string

        Parameters
        ----------
        text: string

        Returns:
        --------
        no_digits: string without numbers
        
        '''
        remove_digits = str.maketrans('', '', digits)
        no_digits = text.translate(remove_digits)
        return no_digits

    # use to format strings before putting into model after deciding to stem or not
    def format_strings(self, df, col_name, stemming):
        '''
        Reformats and tokenizes strings in pandas dataframe column

        Parameters
        ----------
        df: pandas dataframe
        col_name: string
        stemming: stemming function

        Returns:
        --------
        none

        '''
        punctuation_ = set(string.punctuation)
        stopwords_ = set(stopwords.words('english'))
        stopwords_.update(['foundation', 'skin'])

        stemmer = stemming
        df[col_name] = [self.remove_accents(row) for row in df[col_name]]
        df[col_name] = [self.remove_nums(row) for row in df[col_name]]
        df[col_name] = [self.strip_html_tags(row) for row in df[col_name]]
        df[col_name] = df[col_name].replace(r'[^A-Za-z0-9 ]+', '', regex=True)
        df[col_name] = [word_tokenize(row.lower()) for row in df[col_name]]
        df[col_name] = [[word for word in row if not word in stopwords_ and not word in punctuation_]
                    for row in df[col_name]]
        # adjust based on stemming chosen
        df[col_name] = df[col_name].apply(lambda row: ' '.join([stemmer.lemmatize(word) for word in row]))
        return df