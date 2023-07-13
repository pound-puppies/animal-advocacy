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
def age_cat_plots(model_df):
    sns.barplot(x = model_df.age_category_puppy, y = model_df.outcome_adoption, data=df)
    # Add labels and title
    plt.ylabel('age_category_puppy')
    plt.title('adoption')
    # Show the plot
    plt.show()
    
    

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
        
        
        
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def get_baseline(y_train):
    '''
    this function returns a baseline for accuracy
    '''
    baseline_prediction = y_train.mode()
    # Predict the majority class in the training set
    baseline_pred = [baseline_prediction] * len(y_train)
    accuracy = accuracy_score(y_train, baseline_pred)
    baseline_results = {'Baseline': [baseline_prediction],'Metric': ['Accuracy'], 'Score': [accuracy]}
    baseline_df = pd.DataFrame(data=baseline_results)
    return baseline_df 

#creating X,y
def get_xy():
    '''
    This function generates X and y for train, validate, and test to use : X_train, y_train, X_validate, y_validate, X_test, y_test = get_xy()

    '''
    # Acquiring data
    df = w.left_join_csv('austin_animal_outcomes.csv', 'austin_animal_intakes.csv', 'merged_data.csv')
    # Running preperation 
    df, model_df = w.prep_df(df)
    # Split
    train, validate, test = w.split_data(model_df,'outcome')
    # create X & y version of train, where y is a series with just the target variable and X are all the features.    
    X_train = train.drop(['outcome'], axis=1)
    y_train = train.outcome
    X_validate = validate.drop(['outcome'], axis=1)
    y_validate = validate.outcome
    X_test = test.drop(['outcome'], axis=1)
    y_test = test.outcome
    return X_train,y_train,X_validate,y_validate,X_test,y_test

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd


def create_models(seed=123):
    '''
    Create a list of machine learning models.
            Parameters:
                    seed (integer): random seed of the models
            Returns:
                    models (list): list containing the models
    This includes best fit hyperparamaenters                
    '''
    models = []
    models.append(('k_nearest_neighbors', KNeighborsClassifier(n_neighbors=100)))
    models.append(('logistic_regression', LogisticRegression(random_state=seed)))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=3,min_samples_split=4,random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(max_depth=3,random_state=seed)))
    models.append(('support_vector_machine', SVC(random_state=seed)))
    models.append(('naive_bayes', GaussianNB()))
    models.append(('gradient_boosting', GradientBoostingClassifier(random_state=seed)))
    return models


def get_models():
    # create models list
    models = create_models(seed=123)
    X_train, y_train, X_validate, y_validate, X_test, y_test = get_xy()
    # initialize results dataframe
    results = pd.DataFrame(columns=['model', 'set', 'accuracy', 'recall'])
    
    # loop through models and fit/predict on train and validate sets
    for name, model in models:
        # fit the model with the training data
        model.fit(X_train, y_train)
        
        # make predictions with the training data
        train_predictions = model.predict(X_train)
        
        # calculate training accuracy, recall, and precision
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_recall = recall_score(y_train, train_predictions, average='weighted')
        train_precision = precision_score(y_train, train_predictions, average='weighted')
        
        # make predictions with the validation data
        val_predictions = model.predict(X_validate)
        
        # calculate validation accuracy, recall, and precision
        val_accuracy = accuracy_score(y_validate, val_predictions)
        val_recall = recall_score(y_validate, val_predictions, average='weighted')
        val_precision = precision_score(y_validate, val_predictions, average='weighted')

        
        # append results to dataframe
        results = results.append({'model': name, 'set': 'train', 'accuracy': train_accuracy, 'recall': train_recall, 'precision' : train_precision},ignore_index=True)
        results = results.append({'model': name, 'set': 'validate', 'accuracy': val_accuracy, 'recall': val_recall, 'precision' : val_precision}, ignore_index=True)
  

        '''
        this section left in case I want to return to printed format rather than data frame
        # print classifier accuracy and recall
        print('Classifier: {}, Train Accuracy: {}, Train Recall: {}, Validation Accuracy: {}, Validation Recall: {}'.format(name, train_accuracy, train_recall, val_accuracy, val_recall))
        '''
    return results