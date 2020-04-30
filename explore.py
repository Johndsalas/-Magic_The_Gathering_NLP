import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

def get_distribution(df):
    '''
    display distribution of rows by color in pandas data frame
    '''

    dist = pd.concat([df.color.value_counts(),
                    df.color.value_counts(normalize=True)], axis=1)

    dist.columns = ['n', 'percent']

    return dist

def word_soup(text):
    '''
    Turn text into list of words in text
    '''
    wnl = nltk.stem.WordNetLemmatizer()
 
    words = re.sub(r'[^\w\s]', '', text).split()
    return [word for word in words]

def word_count(df):
    '''
    display count of each word by color in a pandas data frame
    '''

    all_words = word_soup(' '.join(df.text))
    blue_words = word_soup(' '.join(df[df.color == 'Blue'].text))
    green_words = word_soup(' '.join(df[df.color == 'Green'].text))
    red_words = word_soup(' '.join(df[df.color == 'Red'].text))
    white_words = word_soup(' '.join(df[df.color == 'White'].text))
    black_words = word_soup(' '.join(df[df.color == 'Black'].text))

    all_freq = pd.Series(all_words).value_counts()
    blue_freq = pd.Series(blue_words).value_counts()
    green_freq = pd.Series(green_words).value_counts()
    red_freq = pd.Series(red_words).value_counts()
    white_freq = pd.Series(white_words).value_counts()
    black_freq = pd.Series(black_words).value_counts()

    word_counts = (pd.concat([all_freq,blue_freq,green_freq,red_freq,white_freq,black_freq], axis=1, sort=True)
                .set_axis(['all','blue','green','red','white','black'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))

    return word_counts.sort_values(by='all', ascending=False).head(10)