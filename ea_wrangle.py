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

def left_join_csv(outcomes_file, intakes_file, merged_file):
    # Read the CSV files
    outcomes = pd.read_csv(outcomes_file)
    intakes = pd.read_csv(intakes_file)

    # Perform the left join
    merged_data = pd.merge(outcomes, intakes, on='Animal ID', how='left')

    # Save the merged data to a new CSV file
    merged_data.to_csv(merged_file, index=False)    
    
    return merged_data

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

    df = df.drop(columns=["color"])

    return df

def get_prep_aa(df):
    # made all column names lower case
    df.columns = df.columns.str.lower()
    df = df.apply(lambda x: x.astype(str).str.lower())
    # changed column names to make them more readable
    new_columns = {
        'datetime_x': 'outcome_datetime',
        'datetime_y': 'intake_datetime',
        'monthyear_x': 'outcome_monthyear',
        'monthyear_y': 'intake_monthyear',
        'name_y': 'name',
        'breed_y': 'breed',
        'animal type_y': 'species',
        'outcome type': 'outcome',
        'color_y': 'color',
        'sex upon outcome': 'outcome_sex',
        'sex upon intake': 'intake_sex',
        'intake type': 'intake_type',
        'age upon intake': 'intake_age',
        'age upon outcome': 'outcome_age',
        'date of birth': 'dob',
        'intake condition': 'intake_condition',
        'found location': 'found_location',
        'animal id': 'id'      
    }
    df = df.rename(columns=new_columns)
    
    #dropped unnecessary column names, outcome subtype, due to having over 119k of 193k rows empty, intake_monthyear, outcome_month_year, animal type_x, are predominantly the same, 
    columns_to_drop = ['outcome subtype', 'name_x', 'breed_x', 'animal type_x', 'color_x', 'intake_monthyear', 'outcome_monthyear', 'found_location']
    df = df.drop(columns=columns_to_drop)
    
    df.dropna(subset=['intake_sex'], inplace=True)
    df.dropna(subset=['outcome'], inplace=True)
    
    #converted date to dates
    df['outcome_datetime'] = pd.to_datetime(df['outcome_datetime'])
    df['intake_datetime'] = pd.to_datetime(df['intake_datetime'])
    df['dob'] = pd.to_datetime(df['dob'], format='%m/%d/%Y')
    
    #pulled dates from intake and outcome date**************************
    df['intake_date'] = df['intake_datetime'].dt.date
    df['outcome_date'] = df['outcome_datetime'].dt.date
    
    #converted pulled dates to datetimes
    df['outcome_date'] = pd.to_datetime(df['outcome_date'])
    df['intake_date'] = pd.to_datetime(df['intake_date'])
    
    #calculated age at intake
    df['intake_age'] = df['intake_date'] - df['dob']
    
    #calculated age at outcome
    df['outcome_age'] = df['outcome_date'] - df['dob']
    
    
    df['intake_age'] = df['intake_age'] / 30
    df['outcome_age'] = df['outcome_age'] / 30
    
    #filtered for cats and dogs
    df = df[df['species'].isin(['cat', 'dog'])]
    df = df[df['outcome'].isin(['adoption', 'transfer', 'rto-adopt', 'return to owner', 'euthanasia'])]
    df = df[df['intake_type'].isin(['stray', 'owner surrender', 'public assist', 'abandoned'])]
    
    #created columns for breeds*****************************************************************
    df['mixed'] = np.where(df['breed'].str.contains('mix', case=False, na=False), 1, 0)
    df['two_breeds'] = np.where(df['breed'].str.contains('/', case=False, na=False), 1, 0)
    df['pure_bred'] = np.where(df['breed'].isin(['/', 'mix']), 1, 0)
    
    
                                
    #changed the order of the columns for readability
    desired_order = ['name', 'outcome', 'dob', 'intake_type', 'intake_datetime', 'outcome_datetime', 'intake_condition', 
                 'intake_age', 'outcome_age', 'species', 'intake_sex', 'breed', 'mixed', 'two_breeds', 'pure_breeds', 'color']
    df = df.reindex(columns=desired_order)
    
    df['intake_age'] = df['intake_age'].dt.days
    df['outcome_age'] = df['outcome_age'].dt.days
    
    df = transform_intake_condition(df)
    df = transform_color(df)

    return df
