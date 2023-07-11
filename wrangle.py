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
    df.columns = df.columns.str.lower()

    new_columns = {
        'datetime_x': 'outcome_datetime',
        'datetime_y': 'intake_datetime',
        'monthyear_x': 'outcome_monthyear',
        'monthyear_y': 'intake_monthyear',
        'name_x': 'name',
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

    columns_to_drop = ['outcome subtype', 'name_y', 'breed_x', 'animal type_x', 'color_x']
    df = df.drop(columns=columns_to_drop)

    df['outcome_datetime'] = pd.to_datetime(df['outcome_datetime'])
    df['intake_datetime'] = pd.to_datetime(df['intake_datetime'])

    df = df[df['species'].isin(['Cat', 'Dog'])]

    return df
