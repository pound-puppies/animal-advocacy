import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import env
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

    
    