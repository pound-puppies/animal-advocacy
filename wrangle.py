import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sqlalchemy import text, create_engine
import scipy.stats as stats

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
    df['condition'] = df['condition'].replace(['feral', 'neurologic', 'behavior', 'space'], 'mental')

    # Set values indicating medical attention
    df['condition'] = df['condition'].replace(['nursing', 'neonatal', 'medical', 'pregnant', 'med attn', 
                                                            'med urgent', 'parvo', 'agonal', 'panleuk'], 'medical attention')

    # Drop rows with 'other', 'unknown', and 'nan' values
    df = df[df['condition'].isin(['other', 'unknown', 'nan']) == False]

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

def prep_df(df):
# lower cases df
    df.columns = df.columns.str.lower()
    df = df.apply(lambda x: x.astype(str).str.lower())

    # returns all dupes
    duplicates = df[df['animal id'].duplicated()]

    # duplicate ids -- that need to drop
    dupe_list = list(duplicates['animal id'].unique())

    # removed dupes --  shape after dropping dupes (124940, 23)
    df = df[~df['animal id'].isin(dupe_list)]

    # rename columns
    new_columns = {
        'datetime_x': 'outcome_datetime',
        'datetime_y': 'intake_datetime',
        'name_y': 'name',
        'breed_y': 'breed',
        'animal type_y': 'species',
        'outcome type': 'outcome',
        'color_y': 'color',
        'sex upon outcome': 'sex',
        'intake type': 'intake_type',
        'date of birth': 'dob',
        'intake condition': 'condition',
        'animal id': 'id'      
    }
    df = df.rename(columns=new_columns)



    # Filter 'species' to only return cats or dogs
    df = df[df['species'].isin(['dog', 'cat'])]

    ### drop nulls
    # drop nan from outcome
    df = df[df.outcome != "nan"]

    # drop nan from intake type
    df = df[df.intake_type != "nan"]

    # drop nan from sex and 
    df = df[~df['sex'].isin(['nan', 'unknown'])]

    # Replace 'nan' values in 'name' column with 0
    df['name'] = df['name'].replace('nan', 0)
    # Replace all other names with 1
    df.loc[df['name'] != 0, 'name'] = 1

    # outlier drops
    # drop wildlife variable from intake type
    df = df[df.intake_type != "wildlife"]

    # fix datatypes
    df['dob'] = pd.to_datetime(df['dob'])

    # change dtype to datetime
    df['outcome_date'] = pd.to_datetime(df['outcome_datetime']).dt.strftime('%m/%d/%Y').astype("datetime64")
    df['intake_date'] = pd.to_datetime(df['intake_datetime']).dt.strftime('%m/%d/%Y').astype("datetime64")

    # create release age
    df['outcome_age'] = (df.outcome_date - df.dob).dt.days

    # Convert 'outcome_date' column to datetime
    df['outcome_date'] = pd.to_datetime(df['outcome_date'])
    # create month and year 
    df["rel_month"] = df['outcome_date'].dt.strftime('%b')
    df["rel_year"] = df['outcome_date'].dt.year

    # age column
    # Define the conditions for each age category
    conditions = [
        (df['outcome_age'] <= 730),
        (df['outcome_age'] >= 731) & (df['outcome_age'] <= 2920),
        (df['outcome_age'] >= 2921)
    ]
    # Define the corresponding values for each age category
    values = ['puppy', 'adult', 'senior']

    # lower cases df
    df.columns = df.columns.str.lower()
    df = df.apply(lambda x: x.astype(str).str.lower())

    # returns all dupes
    duplicates = df[df['id'].duplicated()]

    # duplicate ids -- that need to drop
    dupe_list = list(duplicates['id'].unique())

    # removed dupes --  shape after dropping dupes (124940, 23)
    df = df[~df['id'].isin(dupe_list)]

    # rename columns
    new_columns = {
        'datetime_x': 'outcome_datetime',
        'datetime_y': 'intake_datetime',
        'name_y': 'name',
        'breed_y': 'breed',
        'animal type_y': 'species',
        'outcome type': 'outcome',
        'color_y': 'color',
        'sex upon outcome': 'sex',
        'intake type': 'intake_type',
        'date of birth': 'dob',
        'intake condition': 'condition',
        'animal id': 'id'      
    }
    df = df.rename(columns=new_columns)

    # Filter 'species' to only return cats or dogs
    df = df[df['species'].isin(['dog', 'cat'])]

    ### drop nulls
    # drop nan from outcome
    df = df[df.outcome != "nan"]

    # drop nan from intake type
    df = df[df.intake_type != "nan"]

    # drop nan from sex and 
    df = df[~df['sex'].isin(['nan', 'unknown'])]

    # Replace 'nan' values in 'name' column with 0
    df['name'] = df['name'].replace('nan', 0)
    # Replace all other names with 1
    df.loc[df['name'] != 0, 'name'] = 1

    # outlier drops
    # drop wildlife variable from intake type
    df = df[df.intake_type != "wildlife"]

    # change dtype to datetime
    df['outcome_date'] = pd.to_datetime(df['outcome_datetime']).dt.strftime('%m/%d/%Y').astype("datetime64")
    df['intake_date'] = pd.to_datetime(df['intake_datetime']).dt.strftime('%m/%d/%Y').astype("datetime64")

    # Convert 'outcome_date' column to datetime
    df['outcome_date'] = pd.to_datetime(df['outcome_date'])
    # create month and year 
    df["rel_month"] = df['outcome_date'].dt.strftime('%b')
    df["rel_year"] = df['outcome_date'].dt.year

    # Create a mapping dictionary for renaming
    mapping = {
        'return to owner': 'adoption',
        'rto-adopt': 'adoption'
    }
    # Rename values in 'outcome' column based on the mapping dictionary
    df['outcome'] = df['outcome'].replace(mapping)

    # Rename remaining values to 'other'
    df.loc[~df['outcome'].isin(['adoption', 'transfer']), 'outcome'] = 'other'

    # create intake columns and colors
    df = transform_intake_condition(df)
    df = transform_color(df)
    
    # update dtypes
    df.name = df.name.astype('int')
    df.outcome_age = df.outcome_age.astype('int')
    df['dob'] = pd.to_datetime(df['dob'])

    # drop these columns
    df = df.drop(columns=["id","name_x", "monthyear_x", "animal type_x",
                     "sex upon intake", "age upon outcome", "breed_x",
                     "color_x", "monthyear_y", "found location", "age upon intake",
                          "outcome subtype", "intake_datetime", "outcome_datetime", "outcome_date", "intake_date"])
    # Rename values in 'breed' column
    df.loc[df['breed'].str.contains('mix|domestic shorthair|domestic medium hair|domestic longhair', case=False), 'breed'] = 'mix'
    df.loc[df['breed'].str.contains('/', na=False), 'breed'] = 'two breeds'
    df.loc[~df['breed'].isin(['two breeds', 'mix']), 'breed'] = 'single breed'

    dummy_df = pd.get_dummies(df[['outcome', 'sex','intake_type', 'condition',
                             'species', 'breed', 'rel_month', 'rel_year', 'primary_color']],
                          drop_first=True)
    
    bool_df = df[['name', 'outcome_age', 'is_tabby', 'mix_color']]

    model_df = pd.concat([bool_df, dummy_df], axis=1)

    return df, model_df

#This confirms and Validates my split.
def split_data(df, target_variable):
    '''
    Takes in two arguments the dataframe name and the ("target_variable" - must be in string format) to stratify  and
    return train, validate, test subset dataframes will output train, validate, and test in that order.
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2,
                                   random_state=123,
                                   stratify= df[target_variable])
    train, validate = train_test_split(train, #second split
                                       test_size=.25,
                                       random_state=123,
                                       stratify=train[target_variable])
    print(f'train -> {train.shape}, {round(train.shape[0]*100 / df.shape[0],2)}%')
    print(f'validate -> {validate.shape},{round(validate.shape[0]*100 / df.shape[0],2)}%')
    print(f'test -> {test.shape}, {round(test.shape[0]*100 / df.shape[0],2)}%')
    
    return train, validate, test