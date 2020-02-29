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


class Create_Models(object):
    ''' Preprocessing class train/test split data, fits, and predict y labels 
    of model.

    This also includes visualizations for the model like ROC curve, 
    precision/recall curve
    '''

    def __init__(self):
        pass

    def modelling(self, text_series, y, vectorizer, model):
        '''
        Vectorizes, train/test split, and models dataframe

        Parameters
        ----------
        text_series : array of features
        y : array of labels
        vectorizer : vectorizer function
        model : model function

        Returns:
        --------
        precision, accuracy, recall, thresh_df, roc_auc, model
        '''
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
        thresh_df = self.calculate_threshold_values(probs, y_test)
        
        return precision, accuracy, recall, thresh_df, roc_auc, model

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


    # def get_conf_matrix(self,text_series, y, vectorizer, model):
    #     '''
    #     Create confusion matrix 

    #     Parameters
    #     ----------
    #     text_series: array of features
    #     y: array of labels
    #     vectorizer: vectorizer function
    #     model: model function

    #     Returns:
    #     --------
    #     df: pandas dataframe
    #     '''

    #     # declare X
    #     X = vectorizer.fit_transform(text_series)
        
    #     if model.__class__.__name__ == 'GaussianNB':
    #         X = X.toarray()

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
        
    #     print(confusion_matrix(y_test, y_pred))
 
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
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

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

    def print_precision_acc_recall(self, df, text_series, y, model, vectorizer):
        '''
        Print precision, accuracy, and recall scores for model

        Parameters
        ----------
        df : pandas dataframe
        test_series : 
        name: string

        Returns:
        --------
        None
        '''
        print('Precision for {} is {:.2f}'.format(self.modelling(df[text_series], y, vectorizer, model)[5].__class__.__name__,
                                                    self.modelling(df[text_series], y, vectorizer, model)[0]))
        print('Accuracy for {} is {:.2f}'.format(self.modelling(df[text_series], y, vectorizer, model)[5].__class__.__name__,
                                                    self.modelling(df[text_series], y, vectorizer, model)[1]))
        print('Recall for {} is {:.2f}'.format(self.modelling(df[text_series], y, vectorizer, model)[5].__class__.__name__,
                                                self.modelling(df[text_series], y, vectorizer, model)[2]))
