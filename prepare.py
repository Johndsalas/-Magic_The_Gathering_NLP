import numpy as np
import pandas as pd
import re
import unicodedata
import pathlib

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

def get_preped_data():
    '''
    Get or create prepared data "mtgprep.csv"
    '''

    # define file name
    file = pathlib.Path("mtgprep.csv")

    # if file exists open file as pandas data frame
    if file.exists ():

        df = pd.read_csv('mtgprep.csv')

    # if file does not exist aquire and prep data from 'cards.csv' and write that data to 'mtgprep.csv'
    else:
        
        df = prepare_mtg(wrangle_mtg())

        df.to_csv('mtgprep.csv', index=False)

    return df

def wrangle_mtg():
    '''
    Aquire data from 'cards.csv' and convert it to a pandas data frame
    '''

    # read cards.csv into a pandas dataframe
    df = pd.read_csv('cards.csv')

    return df

def prepare_mtg(df):
    '''
    Prepare mtg data for analysis
    '''

    # rewite data frame with only relavent columns
    df = df[['name','colorIdentity','text','isPaper']]

    # use only cards that exist as phisycal cards
    df = df[df.isPaper==1]
    df = df.drop(columns='isPaper')

    # use only cards with text
    df = df[df.text.notna()]

    # use only cards with a single color identity 
    colors = ['W','U','B','R','G']
    df = df.loc[df.colorIdentity.isin(colors)]

    # rewrite values in color identity to write the full word
    df['colorIdentity'] = np.where(df['colorIdentity'] == 'G', 'Green', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'U', 'Blue', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'W', 'White', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'B', 'Black', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'R', 'Red', df['colorIdentity'])

    # rename columns 
    df=df.rename(columns={'colorIdentity':'color'})

    # apply cleaning functions to text
    df['text'] = df.text.apply(modify_text).apply(basic_clean).apply(lemmatize).apply(remove_stopwords)

    # drop duplicate rows and name column
    df.drop_duplicates(subset='name',inplace=True)
    df = df.drop(columns='name')

    return df

def modify_text(text):
    '''
    modify text to make it machine readable 
    '''
    
    text = text.replace("{T}","Tap")
    text = text.replace("{C}","ColorlessMana")
    text = text.replace("{W}","WhiteMana")
    text = text.replace("{B}","BlackMana")
    text = text.replace("{U}","BlueMana")
    text = text.replace("{R}","RedMana")
    text = text.replace("{G}","GreenMana")

    text = text.replace("+","Plus")
    text = text.replace("-","Minus")
    text = text.replace("/","and")
    text = re.sub(r"[0-9]",'',text)
    
    return text

def basic_clean(article):
    '''
    calls child functions preforms basic cleaning on a string
    converts string to lowercase, ASCII characters,
    and eliminates special characters
    '''
    # lowercases letters
    article = article.lower()

    # convert to ASCII characters
    article = get_ASCII(article)

    # remove non characters
    article = purge_non_characters(article)
    
    return article

def get_ASCII(article):
    '''
    normalizes a string into ASCII characters
    '''

    article = unicodedata.normalize('NFKD', article)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    
    return article

def purge_non_characters(article):
    '''
    removes special characters from a string
    '''
    
    article = re.sub(r"[^a-z\s]", ' ', article)
    
    return article

def remove_stopwords(article,extra_words=[],exclude_words=[]):
    '''
    removes stopwords from a string
    user may specify a list of words to add or remove from the list of stopwords
    '''

    # create stopword list using english
    stopword_list = stopwords.words('english')
    
    # remove words in extra_words from stopword list 
    [stopword_list.remove(f'{word}') for word in extra_words]
    
    # add words fin exclude_words to stopword list
    [stopword_list.append(f'{word}') for word in exclude_words]
    
    # slpit article into list of words
    words = article.split()

    # remove words in stopwords from  list of words
    filtered_words = [w for w in words if w not in stopword_list]
    
    # rejoin list of words into article
    article_without_stopwords = ' '.join(filtered_words)
    
    return article_without_stopwords

def lemmatize(article):
    '''
    lemmatizes words in a string
    '''

    # create lemmatize object
    wnl = nltk.stem.WordNetLemmatizer()
    
    # split article into list of words and stem each word
    lemmas = [wnl.lemmatize(word) for word in article.split()]

    #  join words in list into a string
    article_lemmatized = ' '.join(lemmas)
    
    return article_lemmatized

def split_data(df):

    train, test = train_test_split(df, train_size = .80, random_state = 123)

    return train, test