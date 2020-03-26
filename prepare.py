import numpy as np
import pandas as pd
import re
import pathlib

def get_mtg_data():
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
    df = df[['colorIdentity','types','convertedManaCost','rarity','flavorText','isPaper']]

    # use only cards that exist as phisycal cards
    df = df[df.isPaper==1]
    df = df.drop(columns='isPaper')

    # use only cards with flavor text
    df = df[df.flavorText.notna()]

    # use only cards with a single color identity 
    colors = ['W','U','B','R','G']
    df = df.loc[df.colorIdentity.isin(colors)]

    # merge like card types 

    df['types'] = np.where(df['types'] == 'Artifact,Creature', 'Creature', df['types'])

    df['types'] = np.where(df['types'] == 'Summon', 'Creature', df['types'])

    df['types'] = np.where(df['types'] == 'Land,Creature', 'Land', df['types'])

    df['types'] = np.where(df['types'] == 'Artifact,Land', 'Land', df['types'])

    df['types'] = np.where(df['types'] == 'Tribal,Instant', 'Instant', df['types'])

    df['types'] = np.where(df['types'] == 'Tribal,Sorcery', 'Sorcery', df['types'])

    df['types'] = np.where(df['types'] == 'Tribal,Enchantment', 'Enchantment', df['types'])

    df['types'] = np.where(df['types'] == 'instant', 'Instant', df['types'])

    # remove remaining cards that are not exclusive to one of the seven card types
    types = ['Creature','Instant','Sorcery','Enchantment','Land','Artifact','Planeswalker']
    df = df.loc[df.types.isin(types)]

    # rewrite values in color identity to write the full word
    df['colorIdentity'] = np.where(df['colorIdentity'] == 'G', 'Green', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'U', 'Blue', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'W', 'White', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'B', 'Black', df['colorIdentity'])

    df['colorIdentity'] = np.where(df['colorIdentity'] == 'R', 'Red', df['colorIdentity'])

    # clean up flavorText and groupby flavorText to reduce duplicates
    df['flavorText'] = df.flavorText.apply(remove)

    df['flavorText'] = df.flavorText.apply(erase_end)

    df['flavorText'] = df.flavorText.apply(remove_space)

    df = df.groupby('flavorText').agg('max').reset_index()

    # reorder columns 
    df = df[['colorIdentity','types','convertedManaCost','rarity','flavorText']]

    # remove rows with flavor text no in English
    df = df.drop([12450,12451,12453,12454,12455,12456,12457,12458,12459,12460,12461,12462])

    # remove seen duplicates
    df = df.drop([2,7968,6562])

    # add sentament and intensity columns
    df['sentiment'] = df.flavorText.apply(sent_score)

    df['intensity'] = df.sentiment.abs()

    # rename columns 
    df=df.rename(columns={'colorIdentity':'color','types':'type','convertedManaCost':'cost','flavorText':'flavor'})

    df = df.round(2)

    return df

def remove_space(value):
    '''
    remove whitespace from around text
    '''
    
    return value.strip()