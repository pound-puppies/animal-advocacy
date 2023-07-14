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


#model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, accuracy_score, recall_score, classification_report
from scipy import stats
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



                                                            ##################### Functions for Exploration #########
#df = pd.read_csv('merged_data.csv')   
#w.prep_df(df)
#train, validate, test = w.split_data(df, 'outcome')   


def age_cat_plots(model_df):
    sns.barplot(x = model_df.age_category_puppy, y = model_df.outcome_adoption, data=df)
    # Add labels and title
    plt.ylabel('age_category_puppy')
    plt.title('adoption')
    # Show the plot
    plt.show()
    
def summary(df):
    print(df.head()),
    print(df.describe()),
    print(df.shape)
    
def plot_categorical_variables_1(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            if len(df[column].unique()) > 10:
                plt.figure(figsize=(12, 6))
            else:
                plt.figure()
                
                sns.countplot(df[column])
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
    plt.xlabel('Month')
    plt.ylabel('Adopted')
    # Show the plot
    plt.show()
    
def month_outcome(df):
    # Group the data by month and AdoptionStatus, and count the occurrences
    grouped = df.groupby(['rel_month', 'outcome']).size().unstack()
    # Create a stacked bar plot
    grouped.plot(kind='bar', stacked=True)
    plt.title('Adoptions by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    # Show the legend
    plt.legend()
    # Display the plot
    plt.show()
    

def sex_viz(train):
    '''
    This function pulls in a chart comparing sex and outcome using plotly express
    '''
    grouped_data = train.groupby(['sex', 'outcome']).size().reset_index(name='count')
    fig = px.bar(grouped_data, x='sex', y='count', color='outcome', barmode='group')
    fig.update_layout(title='Sex vs Outcome')  # Update layout to set the title
    fig.show()        

    
def species_viz(train):
    '''
    This function pulls in a chart comparing sex and outcome using plotly express
    '''
    grouped_data = train.groupby(['species', 'outcome']).size().reset_index(name='count')
    fig = px.bar(grouped_data, x='species', y='count', color='outcome', barmode='group')
    fig.update_layout(title='Species vs Outcome')  # Update layout to set the title
    fig.show()   
    
def month_viz(train):
    '''
    This function pulls in a chart comparing sex and outcome using plotly express
    '''
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    train['rel_month'] = pd.Categorical(train['rel_month'], categories=month_order, ordered=True)
    grouped_data = train.groupby(['rel_month', 'outcome']).size().reset_index(name='count')
    grouped_data.sort_values('rel_month', inplace=True)
    fig = px.bar(grouped_data, x='rel_month', y='count', color='outcome', barmode='group')
    fig.update_layout(title='Sex vs Outcome')  # Update layout to set the title
    fig.show() 

def breed_viz(train):
    '''
    This function pulls in a chart comparing sex and outcome using plotly express
    '''
    grouped_data = train.groupby(['breed', 'outcome']).size().reset_index(name='count')
    fig = px.bar(grouped_data, x='breed', y='count', color='outcome', barmode='group')
    fig.update_layout(title='Breed vs Outcome')  # Update layout to set the title
    fig.show()  


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
    

def sex_stats(train):
    '''
    This function runs a chi2 stats test on sex and outcome.
    It returns the contingency table and results in a pandas DataFrame.
    '''
    # Create a contingency table
    contingency_table = pd.crosstab(train['sex'], train['outcome'])

    # Perform the chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

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

    
def breed_stats(train):
    '''
    This function runs a chi2 stats test on sex and outcome.
    It returns the contingency table and results in a pandas DataFrame.
    '''
    # Create a contingency table
    contingency_table = pd.crosstab(train['breed'], train['outcome'])

    # Perform the chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

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

def species_stats(train):
    '''
    This function runs a chi2 stats test on sex and outcome.
    It returns the contingency table and results in a pandas DataFrame.
    '''
    # Create a contingency table
    contingency_table = pd.crosstab(train['species'], train['outcome'])

    # Perform the chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

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

def month_stats(train):
    '''
    This function runs a chi2 stats test on sex and outcome.
    It returns the contingency table and results in a pandas DataFrame.
    '''
    # Create a contingency table
    contingency_table = pd.crosstab(train['rel_month'], train['outcome'])

    # Perform the chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

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