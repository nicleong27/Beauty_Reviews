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
from src.EDA import *
from src.Preprocessing import *

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')


df = pd.read_csv('sephora_review_db.csv.zip')

model_df = df[['review_text', 'skin_type']].copy()

def remove_accents(input_str):
    ''' 
    Removes accents from string

    Parameters
    ----------
    input_str: str

    Returns:
    --------
    nfkd_form: str
        String without accents
    '''
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    #     return only_ascii.decode()
    return nfkd_form


def remove_nums(text):
    '''
    Removes numbers from string

    Parameters
    ----------
    text: str

    Returns:
    --------
    no_digits: str
        String without numbers
    
    '''
    remove_digits = str.maketrans('', '', digits)
    no_digits = text.translate(remove_digits)
    return no_digits


def format_strings(text):
    '''
    Reformats and tokenizes strings in pandas dataframe column

    Parameters
    ----------
    text : str

    Returns:
    --------
    text : str

    '''
    # punctuation_ = set(string.punctuation)
    stopwords_ = set(stopwords.words('english'))
    stopwords_.update(['foundation', 'skin'])


    text = text.lower()
    text = remove_accents(text)
    text = remove_nums(text)
    text = text.replace(r'[^A-Za-z0-9 ]+', '')
    text = ' '.join(word for word in text.split() if word not in stopwords_)

    return text

model_df['review_text'] = model_df['review_text'].apply(format_strings)
# drop NaN values
model_df.dropna(inplace=True, axis=0)
model_df2 = model_df[(model_df['skin_type'] == 'oily') | (model_df['skin_type'] == 'dry') | (model_df['skin_type'] == 'normal')].copy()
# model_df2['skin_type'] = np.where((model_df2['skin_type'] == 'oily'), 1, 0)


def featurize_split_data(text_series, label_series, max_nb_words, max_seq_length):
    ''' 
    Tokenizes and train/test splits data

    Parameters
    ----------
    text_series : pandas series
    label_series : pandas series
    max_nb_words : int
    max_seq_length : int 

    Returns:
    --------
    X_train, X_test, y_train, y_test : array, array, array, array
    '''
    tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(text_series.values)
    word_index = tokenizer.word_index

    X = tokenizer.texts_to_sequences(text_series.values)
    # pad sequences vectorizes and creates a uniform length of sentences
    X = pad_sequences(X, maxlen=max_seq_length)
    y = pd.get_dummies(label_series).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    
    return X_train, X_test, y_train, y_test


def define_model(max_nb_words, embedding_dim, max_seq_length):
    ''' 
    Defines model, includes embedding layer, dropouts, and 
    LSTM layers.

    Parameters
    ----------
    max_nb_words : int
    embedding_dim : int
    max_seq_length : int

    Returns:
    --------
    model : obj
    '''
    # LSTM model3
    model = Sequential()
    # embedding layer in which words are encoded as real-valued vectors and 
    # where similarity between words is translated to closeness in vector space
    model.add(Embedding(max_nb_words, embedding_dim, input_length=max_seq_length))
    # whole feature map may be dropped out, prevents co-adaptation of feature
    #  & its neighbors
    model.add(SpatialDropout1D(0.2))
    # LSTM layer with 100 memory units
    model.add(LSTM(100, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(LSTM(100, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(LSTM(100, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_cm(y_test, y_pred_labels):
    ''' 
    Plots confusion matrix

    Parameters
    ----------
    y_test : arr
    y_pred_labels : arr

    Returns:
    --------
    None
    '''
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred_labels.argmax(axis=1))

    # # flip confusion matrix, so that confusion matrix is properly ordered
    cm_flip = np.fliplr(cm)
    cm_flip = np.flipud(cm_flip)
    df_cm = pd.DataFrame(cm_flip, index=('oily', 'dry', 'normal'), columns=('oily', 'dry', 'normal'))
    
    fig, ax = plt.subplots(figsize=(10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig('imgs/Model19_cm.png')


def plot_line(history, train_metric, val_metric, graph_name):
    ''' 
    Draws line graph of indicated metric

    Parameters
    ----------
    history : obj
        Fitted model
    train_metric : str
        Training set metric
    val_metric : str
        Validation/test set metric
    graph_name : str
        Name of graph

    Returns:
    --------
    None
    '''
    fig, ax = plt.subplots(figsize=(10,7))
    plt.title(graph_name)
    plt.plot(history.history[train_metric], label='train')
    plt.plot(history.history[val_metric], label='test')
    plt.legend()

    plt.tight_layout()

if __name__ == '__main__':
    max_nb_words = 20000
    batch_size = 64  
    nb_epoch = 20     
    embedding_dim = 100
    max_seq_length = 93
    

    X_train, X_test, Y_train, Y_test = featurize_split_data(model_df2['review_text'], model_df2['skin_type'],
                                        max_nb_words, max_seq_length)

    model = define_model(max_nb_words, embedding_dim, max_seq_length)
    
    # during fit process watch train and test error simultaneously

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    # model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
    #         verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    y_pred = model.predict(X_test)
    y_pred_labels = (y_pred > 0.5).astype(np.int)

    plot_cm(Y_test, y_pred_labels)
    plot_line(history, 'loss', 'val_loss', 'Model Loss')
    plt.savefig('imgs/Model19_Loss.png')
    plot_line(history, 'accuracy', 'val_accuracy', 'Model Accuracy')
    plt.savefig('imgs/Model19_Accuracy.png')

    # save model to file
    model.save('models/model19.h5')

   





