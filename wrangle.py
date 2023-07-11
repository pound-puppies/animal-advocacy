import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
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

def get_prep_aa(df):
    # made all column names lower case
    df.columns = df.columns.str.lower()
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
        'found location': 'found_location'
    }
    df = df.rename(columns=new_columns)
    
    #dropped unnecessary column names, outcome subtype, due to having over 119k of 193k rows empty, intake_monthyear, outcome_month_year, animal type_x, are predominantly the same, 
    columns_to_drop = ['outcome subtype', 'name_x', 'breed_x', 'animal type_x', 'color_x', 'intake_monthyear', 'outcome_monthyear']
    df = df.drop(columns=columns_to_drop)
    
    #converted dates to proper format
    df['outcome_datetime'] = pd.to_datetime(df['outcome_datetime'])
    df['intake_datetime'] = pd.to_datetime(df['intake_datetime'])
    df['dob'] = pd.to_datetime(df['dob'], format='%m/%d/%Y')
    df['intake_date'] = df['intake_datetime'].dt.date
    df['outcome_date'] = df['outcome_datetime'].dt.date
    df['outcome_date'] = pd.to_datetime(df['outcome_date'])
    df['intake_date'] = pd.to_datetime(df['intake_date'])
    df['intake_age'] = df['intake_date'] - df['dob']
    df['outcome_age'] = df['outcome_date'] - df['dob']
    df['intake_age'] = df['intake_age'] / 30
    df['outcome_age'] = df['outcome_age'] / 30
    #filtered for cats and dogs
    df = df[df['species'].isin(['Cat', 'Dog'])]
    
    #changed the order of the columns for readability
    desired_order = ['animal id', 'name', 'outcome', 'dob', 'intake_type', 'intake_datetime', 'outcome_datetime', 'intake_condition', 
                 'intake_age', 'outcome_age', 'species', 'found_location', 'intake_sex', 'breed', 'color']
    df = df.reindex(columns=desired_order)

    return df

