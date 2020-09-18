import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

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
 
    words = re.sub(r'[^\w\s]', '', text).split()

    print(words)
    
    return [word for word in words]

def word_count(df):
    '''
    display count of each word by color in a pandas data frame
    '''

    # create bag (list) of words for all words in text and for all words in text by color
    all_words = word_soup(' '.join(df.text))
    blue_words = word_soup(' '.join(df[df.color == 'Blue'].text))
    green_words = word_soup(' '.join(df[df.color == 'Green'].text))
    red_words = word_soup(' '.join(df[df.color == 'Red'].text))
    white_words = word_soup(' '.join(df[df.color == 'White'].text))
    black_words = word_soup(' '.join(df[df.color == 'Black'].text))

    print(all_words)

    # create a pandas series with each word an
    all_freq = pd.Series(all_words).value_counts()
    blue_freq = pd.Series(blue_words).value_counts()
    green_freq = pd.Series(green_words).value_counts()
    red_freq = pd.Series(red_words).value_counts()
    white_freq = pd.Series(white_words).value_counts()
    black_freq = pd.Series(black_words).value_counts()

    print(all_freq)

    # combine value counts into one pandas data frame
    word_counts = (pd.concat([all_freq,blue_freq,green_freq,red_freq,white_freq,black_freq], axis=1, sort=True)
                .set_axis(['all','blue','green','red','white','black'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))

    # display ratio of top 20 most frequent words by color
    (word_counts.assign(p_blue=word_counts.blue / word_counts['all'],
                        p_green=word_counts.green / word_counts['all'],
                        p_red=word_counts.red / word_counts['all'],
                        p_white=word_counts.white / word_counts['all'],
                        p_black=word_counts.black / word_counts['all'])
                .sort_values(by='all')[['p_blue','p_green','p_red','p_white','p_black']]
                .tail(20)
                .plot.barh(stacked=True, color= ['#cbd7fb','#96aba8','#d26e4a','#f8c8aa','#e9e9e9'], figsize= (12,12)))

    plt.title('Proportion of color for the 20 most common words')
   
    return all_words
   