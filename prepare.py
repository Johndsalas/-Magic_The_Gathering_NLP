import numpy as np
import pandas as pd
import re
import pathlib

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
    df=df.rename(columns={'colorIdentity':'color','types':'type','convertedManaCost':'cost'})

    # apply cleaning functions to text 

    df = df.drop_duplicates()

    return df

def remove_space(value):
    '''
    remove whitespace from around text
    '''
    
    return value.strip()