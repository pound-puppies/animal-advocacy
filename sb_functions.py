######IMPORTS#####

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import wrangle as w

import pandas as pd
import numpy as np

#splits
from sklearn.model_selection import train_test_split

#visualization
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

#scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score, classification_report

#model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression




                                                            ##################### Functions for Exploration #########

    
    

                                                            ################# Functions for Visualization ############

def sex_viz():
    '''
    This function pulls in a chart comparing sex and outcome using plotly express
    '''
    grouped_data = train.groupby(['sex', 'outcome']).size().reset_index(name='count')
    fig = px.bar(grouped_data, x='sex', y='count', color='outcome', barmode='group')
    fig.update_layout(title='Sex vs Outcome')  # Update layout to set the title
    fig.show()        
        


                                                            ####################### Stats Functions ###################

def sex_stats():
    '''
    This function runs a chi2 stats test on sex and outcome.
    It returns the contingency table and results in a pandas DataFrame.
    '''
    # Create a contingency table
    contingency_table = pd.crosstab(train['sex'], train['outcome'])

    # Perform the chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Create a DataFrame for the contingency table
    contingency_sex = pd.DataFrame(contingency_table)

    # Create a DataFrame for the results
    results = pd.DataFrame({
        'Chi-square statistic': [chi2],
        'p-value': [p_value],
        'Degrees of freedom': [dof]
    })

    # Return the contingency table and results DataFrame
    return results


                                                            ###################### Modeling Functions ##################