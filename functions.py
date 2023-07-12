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

import scipy.stats as stats



                                                            ##################### Functions for Exploration #########

    
    

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