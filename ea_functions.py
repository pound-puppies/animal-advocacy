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
import plotly.express as px



#scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score, classification_report

#model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression




         ################# Functions for Exploration #########

def summary(df):
    print(df.head(),
          df.describe(),
          df.shape)
    
def plot_categorical_variables_1(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            if len(df[column].unique()) > 10:
                   plt.figure(figsize=(12, 6))
            else:
                plt.figure()
                
            sns.countplot(df[column])
            plt.title(column)
            plt.xticks(rotation=45)
            plt.show()
            
def plot_categorical_variables_2(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            fig = px.histogram(df, x=column)
            fig.update_traces(hovertemplate='Count: %{y}')
            fig.update_layout(title=column)
            fig.show()
            
def plot_categorical_variables_3(df):
    color_palette = px.colors.qualitative.Plotly

    for column in df.columns:
        if df[column].dtype == 'object':
            fig = px.histogram(df, x=column, color_discrete_sequence=color_palette)
            fig.update_traces(hovertemplate='Count: %{y}')
            fig.update_layout(title=column)
            fig.show()
            

def month_adopt(df):
    sns.barplot(x=df.value_counts('rel_month'),
                y=df.value_counts('rel_month'), data=df)
    # Set the plot title and labels
    plt.title('Count of Records by Month')
    plt.xlabel('Mont')
    plt.ylabel('Adopted')
    # Show the plot
    plt.show()

# Example usage:
# Assuming 'data' is your DataFrame containing categorical variables
    

                                                            ################# Functions for Visualization ############

        
        


                                                            ####################### Stats Functions ###################


def eval_dist(r, p, α=0.05):
    if p > α:
        return print(f"""The data is normally distributed""")
    else:
        return print(f"""The data is NOT normally distributed""")   
    
def eval_Spearman(r, p, α=0.05):
    if p < α:
        return print(f"""We reject H₀, there is a monotonic relationship.
Spearman's r: {r:2f}
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a monotonic relationship.
Spearman's r: {r:2f}
P-value: {p}""")

    
def eval_Pearson(r, p, α=0.05):
    if p < α:
        return print(f"""We reject H₀, there is a linear relationship with a Correlation Coefficient of {r:2f}.
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a linear relationship.
Pearson's r: {r:2f}
P-value: {p}""")


                                                            ###################### Modeling Functions ##################