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

from sklearn.preprocessing import LabelBinarizer

def modelling(text_series, y, vectorizer, model):
    # declare X
    X = vectorizer.fit_transform(text_series)
    
    if model.__class__.__name__ == 'GaussianNB':
        X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probs)
    thresh_df = calculate_threshold_values(probs, y_test)
    
    return precision, accuracy, recall, thresh_df, roc_auc, model



fig, ax = plt.subplots(figsize=(15,8))

def plot_roc(ax, df, name):
    '''Plots single ROC'''
    
    ax.plot([1]+list(df.fpr), [1]+list(df.tpr), label=name)
#     ax.plot([0,1],[0,1], 'k', label="random")
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
#     ax.set_title('ROC Curve - Model Comparison', fontweight='bold', fontsize=24)
    ax.legend(fontsize=14)


def plot_multiple_rocs(model_list, text_series, y, vectorizer):

    def modelling(text_series, y, vectorizer, model):
        # declare X
        X = vectorizer.fit_transform(text_series)
        
        if model.__class__.__name__ == 'GaussianNB':
            X = X.toarray()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
        thresh_df = calculate_threshold_values(probs, y_test)
        
        return precision, accuracy, recall, thresh_df, roc_auc, model
    
    
    def plot_roc(ax, df, name):
        '''Plots single ROC'''
    
        ax.plot([1]+list(df.fpr), [1]+list(df.tpr), label=name)
    #     ax.plot([0,1],[0,1], 'k', label="random")
        ax.set_xlabel('False Positive Rate', fontsize=16)
        ax.set_ylabel('True Positive Rate', fontsize=16)
    #     ax.set_title('ROC Curve - Model Comparison', fontweight='bold', fontsize=24)
        ax.legend(fontsize=14)

    '''Plot multiple ROCs'''
    ax.plot([0,1],[0,1], 'k', label="random")
    for model in model_list:
        results = modelling(text_series, y, vectorizer, model)
        auc_score = results[4]
        plot_roc(ax, results[3], '{} AUC {}'.format(model.__class__.__name__, round(auc_score, 3)))


def calculate_threshold_values(prob, y):
    '''
    Build dataframe of the various confusion-matrix ratios by threshold
    from a list of predicted probabilities and actual y values
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

def print_precision_acc_recall(df, text_series, y, model):
    '''Prints precision, accuracy, and recall for model'''
    print('Precision for {} is {:.2f}'.format(modelling(df[text_series], y, TfidfVectorizer(), model)[5].__class__.__name__,
                                                modelling(df[text_series], y, TfidfVectorizer(), model)[0]))
    print('Accuracy for {} is {:.2f}'.format(modelling(df[text_series], y, TfidfVectorizer(), model)[5].__class__.__name__,
                                                modelling(df[text_series], y, TfidfVectorizer(), model)[1]))
    print('Recall for {} is {:.2f}'.format(modelling(df[text_series], y, TfidfVectorizer(), model)[5].__class__.__name__,
                                                modelling(df[text_series], y, TfidfVectorizer(), model)[2]))

def get_conf_matrix(text_series, y, vectorizer, model):
    # declare X
    X = vectorizer.fit_transform(text_series)
    
    if model.__class__.__name__ == 'GaussianNB':
        X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))