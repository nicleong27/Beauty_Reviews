import operator
import unicodedata
import string
# from bs4 import BeautifulSoup

from string import digits
import string
from nltk.corpus import stopwords
import pandas as pd

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


# def strip_html_tags(text):
#     '''Removes html tags from string

#     Parameters
#     ----------
#     text: string

#     Returns:
#     --------
#     stripped_text: string without html tags
    
#     '''
#     soup = BeautifulSoup(text, "html.parser")
#     stripped_text = soup.get_text(separator=" ")
#     return stripped_text

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
# def format_strings(df, col_name, stopwords_):
#     '''
#     Reformats and tokenizes strings in pandas dataframe column

#     Parameters
#     ----------
#     df: pandas dataframe
#     col_name: string
#     stemming: stemming function

#     Returns:
#     --------
#     none

#     '''
    # punctuation_ = set(string.punctuation)
    

    # stemmer = stemming
    # df[col_name] = [remove_accents(row) for row in df[col_name]]
    # df[col_name] = [remove_nums(row) for row in df[col_name]]
    # df[col_name] = [strip_html_tags(row) for row in df[col_name]]
    # df[col_name] = df[col_name].replace(r'[^A-Za-z0-9 ]+', '', regex=True)
    # df[col_name] = [row.lower() for row in df[col_name]]
    # df[col_name] = [' '.join(word for word in df[col_name].split() if word not in stopwords_)]
    # adjust based on stemming chosen
    # df[col_name] = df[col_name].apply(lambda row: ' '.join([stemmer.lemmatize(word) for word in row]))

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
    punctuation_ = set(string.punctuation)
    stopwords_ = set(stopwords.words('english'))

    text = text.lower()
    text = remove_accents(text)
    text = remove_nums(text)
    text = text.replace(r'[^A-Za-z0-9 ]+', '')
    text = ' '.join(word for word in text.split() if word not in stopwords_)


    return text

def missing_zero_values_table(df):
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