import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sqlalchemy import text, create_engine

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

####################### Imports ############################

                                                        ############### Acquire Functions ###########################

def get_aa_data(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else:
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df

                                                        #################### Prepare Functions ##########################

def transform_intake_condition(df):
    """
    Transforms the intake_condition column of a DataFrame by performing several operations.

    Args:
        df (pandas.DataFrame): The input DataFrame containing an 'intake_condition' column.

    Returns:
        pandas.DataFrame: The transformed DataFrame.

    """

    df = df.apply(lambda x: x.astype(str).str.lower())

    # Change 'Feral', 'Neurologic', 'Behavior', 'Space' to 'mental' category
    df['intake_condition'] = df['intake_condition'].replace(['feral', 'neurologic', 'behavior', 'space'], 'mental')

    # Set values indicating medical attention
    df['intake_condition'] = df['intake_condition'].replace(['nursing', 'neonatal', 'medical', 'pregnant', 'med attn', 
                                                            'med urgent', 'parvo', 'agonal', 'panleuk'], 'medical attention')

    # Drop rows with 'other', 'unknown', and 'nan' values
    df = df[df['intake_condition'].isin(['other', 'unknown', 'nan']) == False]

    return df

    
def transform_color(df):
    """
    Transforms the color column of a DataFrame by performing several operations.

    Args:
        df (pandas.DataFrame): The input DataFrame containing a 'color' column.

    Returns:
        pandas.DataFrame: The transformed DataFrame with additional columns.

    """

    # lowercase everything
    df = df.apply(lambda x: x.astype(str).str.lower())

    # Add spaces between color names separated by slashes
    df['color'] = df['color'].str.replace('/', ' / ')

    # Replace color names with their corresponding standard names
    replacements = {
        'chocolate': 'brown',
        'liver': 'brown',
        'ruddy': 'brown',
        'apricot': 'orange',
        'pink': 'red',
        'cream': 'white',
        'flame point': 'white',
        'blue': 'gray',
        'silver': 'gray',
        'yellow': 'gold',
        'torbie': 'tricolor',
        'tortie': 'tricolor',
        'calico': 'tricolor'
    }
    df['color'] = df['color'].replace(replacements, regex=True)

    # Create new column 'primary_color' with the first color
    colors = ['black', 'brown', 'white', 'tan', 'brindle', 'gray', 'fawn', 'red', 'sable', 'buff', 'orange', 'blue',
              'tricolor', 'gold', 'cream', 'lynx point', 'seal point', 'agouti', 'lilac point']
    for color in colors:
        df.loc[df['color'].str.startswith(color), 'primary_color'] = color

    # Drop rows with 'unknown' color
    df = df[df['color'] != 'unknown']

    # Create column indicating if the animal has a tabby pattern
    df['is_tabby'] = df['color'].str.contains('tabby').astype(int)

    # Create column indicating if the animal has mixed colors
    df["mix_color"] = np.where(df['color'].str.contains(r'\/|tricolor|torbie|tortie'), 1, 0)

    return df
