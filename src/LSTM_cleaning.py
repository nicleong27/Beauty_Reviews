import tensorflow as tf
keras = tf.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
# from tensorflow.keras.embeddings import Embedding
# from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
import pandas as pd
import seaborn as sns


import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')



def remove_accents(input_str):
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


def remove_nums(text):
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
def format_strings(text):
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
    # punctuation_ = set(string.punctuation)
    stopwords_ = set(stopwords.words('english'))

    text = text.lower()
    text = remove_accents(text)
    text = remove_nums(text)
    text = text.replace(r'[^A-Za-z0-9 ]+', '')
    text = ' '.join(word for word in text.split() if word not in stopwords_)


    return text

