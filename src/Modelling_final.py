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


class Create_Models(object):
    ''' Preprocessing class train/test split data, fits, and predict y labels 
    of model.

    This also includes visualizations for the model like ROC curve, 
    precision/recall curve
    '''

    def __init__(self):
        self.y_pred = y_pred
        self.y_test = y_test
        self.precision = precision
        self.accuracy = accuracy
        self.recall = recall
        self.vect = vect 


    def modelling(self, text_series, y, vectorizer, model):
        '''
        Vectorizes, train/test split, and models dataframe

        Parameters
        ----------
        text_series : array of features
        y : array of labels
        vectorizer : vectorizer function
        model_: model function

        Returns:
        --------
        precision, accuracy, recall, thresh_df, roc_auc, model
        '''
        self.vect = vectorizer
        # declare X
        X = vectorizer.fit_transform(text_series)
        bow = vectorizer.get_feature_names()
        
        if model.__class__.__name__ == 'GaussianNB':
            X = X.toarray()

        X_train, X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.25)
        model.fit(X_train, y_train)
        self.y_pred = model.predict(X_test)

        self.precision = precision_score(self.y_test, self.y_pred)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)

        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(self.y_test, probs)
        thresh_df = self.calculate_threshold_values(probs, self.y_test)
        
        return self.precision, self.accuracy, self.recall, thresh_df, roc_auc, model, bow


    def calculate_threshold_values(self, prob, y):
        '''
        Create pandas dataframe that contains True Negatives, False Negatives, True Positives, and
        False Positives

        Parameters
        ----------
        prob: array of predicted probabilities
        y: array of labels

        Returns:
        --------
        df: pandas dataframe
        '''
        df = pd.DataFrame({'prob': prob, 'y': y})
        df.sort_values('prob', inplace=True)
        
        actual_p = df.y.sum()
        actual_n = df.shape[0] - df.y.sum()

        df['tn'] = (df.y == 0).cumsum()
        df['fn'] = df.y.cumsum()
        df['fp'] = actual_n - df.tn
        df['tp'] = actual_p - df.fn

        df['fpr'] = df.fp/(df.fp + df.tn)
        df['tpr'] = df.tp/(df.tp + df.fn)
        df['precision'] = df.tp/(df.tp + df.fp)
        df = df.reset_index(drop=True)
        return df


    def get_conf_matrix(self):

        '''
        Creates confusion matrix 

        Parameters
        ----------
        text_series : pandas series
        y : pandas series
        vectorizer: vectorizer function
        model: model function

        Returns:
        --------
        None
        '''

        cm = confusion_matrix(self.y_test, self.y_pred)
        # flip confusion matrix, so that confusion matrix is properly ordered
        cm_flip = np.flip(cm, 0)
        cm_flip = np.flip(cm_flip, 1)
        df_cm = pd.DataFrame(cm_flip, index=('oily', 'dry'), columns=('oily', 'dry'))

        fig, ax = plt.subplots(figsize=(10,7))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot=True, fmt='g')

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
 

    def plot_roc(self, ax, df, name):
        '''
        Plots single ROC curve

        Parameters
        ----------
        ax: axis in plt.subplots()
        df: pandas dataframe
        name: string

        Returns:
        --------
        None
        '''
        ax.plot([1]+list(df.fpr), [1]+list(df.tpr), label=name)
    #     ax.plot([0,1],[0,1], 'k', label="random")
        ax.set_xlabel('False Positive Rate', fontsize=16)
        ax.set_ylabel('True Positive Rate', fontsize=16)
    #     ax.set_title('ROC Curve - Model Comparison', fontweight='bold', fontsize=24)
        ax.legend(fontsize=14)
        

    # plot multiple rocs for NLP
    def plot_multiple_rocs(self, model_list, text_series, y, vectorizer, ax):
        '''
        Plots multiple ROC curves for model comparison

        Parameters
        ----------
        model_list : lst
                    List of models looking to compare
        text_series : pandas series
                    Pandas series of text
        y : pandas series 
            Y labels
        vectorizer : vectorizer function
        ax : axis in plt.subplots()

        Returns:
        --------
        None
        '''
        ax.plot([0,1],[0,1], 'k', label="random")
        for model in model_list:
            results = self.modelling(text_series, y, vectorizer, model)
            auc_score = results[4]
            self.plot_roc(ax, results[3], '{} AUC {}'.format(model.__class__.__name__, round(auc_score, 3)))
        
        
    def plot_precision_recall(self, ax, df, name):
        '''
        Plots precision recall curve

        Parameters
        ----------
        ax: axis in plt.subplots()
        df: pandas dataframe
        name: str
            Name of precision recall graph

        Returns:
        --------
        None
        '''
        ax.plot(df.tpr,df.precision, label=name)
    #     ax.plot([0,1],[0,1], 'k')
        ax.set_xlabel('Recall', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
    #     ax.set_title('Precision/Recall Curve')
        # ax.plot([0,1],[df.precision[0],df.precision[0]], 'k', label='random')
    #     ax.set_title('ROC Curve - Model Comparison', fontweight='bold', fontsize=24)
        ax.set_xlim(xmin=0,xmax=1)
        ax.set_ylim(ymin=0,ymax=1)
        ax.legend(fontsize=14)

