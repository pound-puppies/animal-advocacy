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
import plotly.graph_objects as go

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
    '''
    Plot a bar chart to visualize the relationship between the age_category_puppy and the outcome_adoption in the given DataFrame.

    Parameters:
        model_df (DataFrame): The DataFrame containing the data to be visualized.
    '''
    sns.barplot(x = model_df.age_category_puppy, y = model_df.outcome_adoption, data=model_df)
    # Add labels and title
    plt.ylabel('age_category_puppy')
    plt.title('adoption')
    # Show the plot
    plt.show()
    
def summary(df):
    '''
    Display a summary of the DataFrame, including the first few rows, basic statistics, and the shape of the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame to be summarized.
    '''
    print(df.head()),
    print(df.describe()),
    print(df.shape)
    
def plot_categorical_variables_1(df):
    '''
    Plot count plots for categorical variables in the DataFrame, one plot for each categorical column. If a column has more than 10 unique categories, the plot will be displayed in a larger size.

    Parameters:
        df (DataFrame): The DataFrame containing categorical variables to be visualized.
    '''
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
    '''
    Plot histograms for categorical variables in the DataFrame using Plotly. Each histogram shows the count of occurrences for each unique category.

    Parameters:
        df (DataFrame): The DataFrame containing categorical variables to be visualized.
    '''
    for column in df.columns:
        if df[column].dtype == 'object':
            fig = px.histogram(df, x=column)
            fig.update_traces(hovertemplate='Count: %{y}')
            fig.update_layout(title=column)
            fig.show()
            
def plot_categorical_variables_3(df):
    '''
    Plot histograms for categorical variables in the DataFrame using Plotly with colored bars for each unique category.

    Parameters:
        df (DataFrame): The DataFrame containing categorical variables to be visualized.
    '''
    color_palette = px.colors.qualitative.Plotly
    for column in df.columns:
        if df[column].dtype == 'object':
            fig = px.histogram(df, x=column, color_discrete_sequence=color_palette)
            fig.update_traces(hovertemplate='Count: %{y}')
            fig.update_layout(title=column)
            fig.show()

def month_adopt(df):
    '''
    Plot a bar chart to display the count of adoption records by month.

    Parameters:
        df (DataFrame): The DataFrame containing adoption data with a column representing the month of adoption.
    '''
    sns.barplot(x=df.value_counts('rel_month'),
                y=df.value_counts('rel_month'), data=df)
    # Set the plot title and labels
    plt.title('Count of Records by Month')
    plt.xlabel('Month')
    plt.ylabel('Adopted')
    # Show the plot
    plt.show()
    
def month_outcome(df):
    '''
    Plot a stacked bar chart to visualize the number of adoptions by month and the outcome type (e.g., adoption, return, etc.).

    Parameters:
        df (DataFrame): The DataFrame containing adoption data with a column representing the month of adoption and another column for outcome type.
    '''
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
    plt.show(renderer='png') 
    
def month_viz(train, target, title_name):
    '''
    This function pulls in a chart comparing sex and outcome using plotly express
    '''
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    train['rel_month'] = pd.Categorical(train['rel_month'], categories=month_order, ordered=True)
    grouped_data = train.groupby(['rel_month', 'outcome']).size().reset_index(name='count')
    grouped_data.sort_values('rel_month', inplace=True)
    fig = px.bar(grouped_data, x='rel_month', y='count', color='outcome', barmode='group')
        # Calculate the total count for each 'rel_month' category
    total_counts = grouped_data.groupby('rel_month')['count'].transform('sum')

    # Calculate the percentage of each 'outcome' category within each 'rel_month' category
    grouped_data['percentage'] = grouped_data['count'] / total_counts * 100

    # Create the stacked bar chart with percentages
    fig = px.bar(grouped_data, x='rel_month', y='percentage', color='outcome', barmode='group',
                 labels={'percentage': 'Total Percentage (%)'})

    # Set x-axis title
    fig.update_xaxes(title_text='Month')
    # Calculate the overall percentage of adoption & transfer line
    overall_adoption_percentage = train[train[target] == 'adoption'].shape[0] / train.shape[0] * 100
    overall_transfer_percentage = train[train[target] == 'transfer'].shape[0] / train.shape[0] * 100
    overall_other_percentage = train[train[target] == 'other'].shape[0] / train.shape[0] * 100
    
    #Add the average line for overall adoption percentage
    fig.add_hline(y=overall_adoption_percentage, line_dash='dash', line_color='blue')
    # Add the average line for overall transferred percentage
    fig.add_hline(y=overall_transfer_percentage, line_dash='dash', line_color='green')
    # Add the average line for overall other percentage
    fig.add_hline(y=overall_other_percentage, line_dash='dash', line_color='red')

    # Set x-axis title
    fig.update_xaxes(title_text="Month")
  # Update layout to set the title
    # Update legend labels
    for trace in fig.data:
        if 'name' in trace:
            trace['name'] = str(trace['name']).capitalize()
        # Add invisible dummy traces to the plot for the legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='blue', dash='dash'), name='Avg. Adoption'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green', dash='dash'), name='Avg. Transferred'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', dash='dash'), name='Avg. Other'))
    # Capitalize words in the legend box titles
    fig.update_layout(legend_title_text='Outcome')
    fig.update_layout(title=title_name)  # Update layout to set the title
    
    fig.update_layout(plot_bgcolor = "rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))  # Update layout to set the title
    fig.show(renderer='png')
    

def px_viz(train, feature, target, ax_name, title_name):
    '''
    This function pulls in a chart comparing sex and outcome using plotly express
    '''
    # Calculate the count of each outcome within each species category
    grouped_data = train.groupby([feature, target]).size().reset_index(name='count')

    # Calculate the total count for each species category to get percentages
    species_total_counts = grouped_data.groupby(feature)['count'].transform('sum')
    grouped_data['percentage'] = grouped_data['count'] / species_total_counts * 100

    # Calculate the overall percentage of adoption & transfer line
    overall_adoption_percentage = train[train[target] == 'adoption'].shape[0] / train.shape[0] * 100
    overall_transfer_percentage = train[train[target] == 'transfer'].shape[0] / train.shape[0] * 100
    overall_other_percentage = train[train[target] == 'other'].shape[0] / train.shape[0] * 100
    
    # Convert  column to uppercase
    grouped_data[feature] = grouped_data[feature].str.capitalize()

    # Create the stacked bar chart
    fig = px.bar(grouped_data, x=feature, y='percentage', color=target, barmode='group',
                 labels={'percentage': 'Total Percentage (%)'})

    #Add the average line for overall adoption percentage
    fig.add_hline(y=overall_adoption_percentage, line_dash='dash', line_color='blue')

    # Add the average line for overall transferred percentage
    fig.add_hline(y=overall_transfer_percentage, line_dash='dash', line_color='green')
    # Add the average line for overall other percentage
    fig.add_hline(y=overall_other_percentage, line_dash='dash', line_color='red')
    # Set x-axis title
    fig.update_xaxes(title_text=ax_name)
    fig.update_layout(title=title_name)  # Update layout to set the title
    # Update legend labels
    for trace in fig.data:
        if 'name' in trace:
            trace['name'] = str(trace['name']).capitalize()
        # Add invisible dummy traces to the plot for the legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='blue', dash='dash'), name='Avg. Adoption'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green', dash='dash'), name='Avg. Transferred'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', dash='dash'), name='Avg. Other'))
    # Capitalize words in the legend box titles
    fig.update_layout(legend_title_text='Outcome')


    fig.update_layout(plot_bgcolor = "rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))  # Update layout to set the title
    fig.show(renderer='png')

def px_num_viz(train, feature, target, title_name, ax_name=None, tick_labels=None):
    '''
    This function pulls in a chart comparing sex and outcome using plotly express
    '''
    # Calculate the count of each outcome within each species category
    grouped_data = train.groupby([feature, target]).size().reset_index(name='count')

    # Calculate the total count for each species category to get percentages
    species_total_counts = grouped_data.groupby(feature)['count'].transform('sum')
    grouped_data['percentage'] = grouped_data['count'] / species_total_counts * 100

    # Calculate the overall percentage of adoption & transfer line
    overall_adoption_percentage = train[train[target] == 'adoption'].shape[0] / train.shape[0] * 100
    overall_transfer_percentage = train[train[target] == 'transfer'].shape[0] / train.shape[0] * 100
    overall_other_percentage = train[train[target] == 'other'].shape[0] / train.shape[0] * 100

    # Create the stacked bar chart
    fig = px.bar(grouped_data, x=feature, y='percentage', color=target, barmode='group',
                 labels={'percentage': 'Total Percentage (%)'})

    #Add the average line for overall adoption percentage
    fig.add_hline(y=overall_adoption_percentage, line_dash='dash', line_color='blue')

    # Add the average line for overall transferred percentage
    fig.add_hline(y=overall_transfer_percentage, line_dash='dash', line_color='green')
    
    # Add the average line for overall other percentage
    fig.add_hline(y=overall_other_percentage, line_dash='dash', line_color='red')
    # Set x-axis title
    fig.update_xaxes(title_text=ax_name)
    fig.update_layout(title=title_name)  # Update layout to set the title
    
    if tick_labels is not None:
        # Use custom tick labels if provided
        fig.update_layout(xaxis=dict(tickvals=list(range(len(tick_labels))), ticktext=tick_labels))
    else:
        # Hide x-axis tick labels
        fig.update_xaxes(showticklabels=False)

    # Update legend labels
    for trace in fig.data:
        if 'name' in trace:
            trace['name'] = str(trace['name']).capitalize()
    
    # Add invisible dummy traces to the plot for the legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='blue', dash='dash'), name='Avg. Adoption'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green', dash='dash'), name='Avg. Transferred'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', dash='dash'), name='Avg. Other'))
    # Capitalize words in the legend box titles
    fig.update_layout(legend_title_text='Outcome')

    fig.update_layout(plot_bgcolor = "rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))  # Update layout to set the title
    fig.show(renderer='png')
                               ####################### Stats Functions ###################


def eval_dist(r, p, α=0.05):
    """
    Evaluate if the data is normally distributed based on the p-value.

    Parameters:
        r (float): The correlation coefficient.
        p (float): The p-value from the statistical test of normality.
        α (float, optional): The significance level (default is 0.05).

    Returns:
        None: Prints a message indicating whether the data is normally distributed or not.
    """
    if p > α:
        return print(f"""The data is normally distributed""")
    else:
        return print(f"""The data is NOT normally distributed""")   
    
def eval_Spearman(r, p, α=0.05):
    """
    Evaluate the Spearman's rank correlation test results.

    Parameters:
        r (float): The Spearman's rank correlation coefficient.
        p (float): The p-value from the Spearman's rank correlation test.
        α (float, optional): The significance level (default is 0.05).

    Returns:
        None: Prints the test result and the correlation coefficient with associated p-value.
    """
    if p < α:
        return print(f"""We reject H₀, there is a monotonic relationship.
Spearman's r: {r:2f}
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a monotonic relationship.
Spearman's r: {r:2f}
P-value: {p}""")

    
def eval_Pearson(r, p, α=0.05):
    """
    Evaluate the Pearson correlation test results.

    Parameters:
        r (float): The Pearson correlation coefficient.
        p (float): The p-value from the Pearson correlation test.
        α (float, optional): The significance level (default is 0.05).

    Returns:
        None: Prints the test result and the correlation coefficient with associated p-value.
    """
    if p < α:
        return print(f"""We reject H₀, there is a linear relationship with a Correlation Coefficient of {r:2f}.
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a linear relationship.
Pearson's r: {r:2f}
P-value: {p}""")
    



def chi_stats(train, feature, target):
    '''
    This function runs a chi2 stats test on feature and target variable.
    It returns the contingency table and results in a pandas DataFrame.
    '''
    # Create a contingency table
    contingency_table = pd.crosstab(train[feature], train[target])

    # Perform the chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Create a DataFrame for the contingency table
    contingency_sex = pd.DataFrame(contingency_table)

    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value == alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"

    # Create a DataFrame for the results
    results = pd.DataFrame({
        'Chi-square statistic': [chi2],
        'p-value': [p_value], 
        'Decision': [decision]
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
def get_xy(df, target):
    '''
    This function generates X and y for train, validate, and test to use : X_train, y_train, X_validate, y_validate, X_test, y_test = get_xy()

    '''
    train, validate, test = w.split_data(df,target)

    X_train = train.drop([target], axis=1)
    y_train = train[target]
    X_validate = validate.drop([target], axis=1)
    y_validate = validate[target]
    X_test = test.drop([target], axis=1)
    y_test = test[target]
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


def get_models(X_train, y_train, X_validate, y_validate):
    """
    Fits multiple machine learning models to the training data and evaluates their performance on the training and validation sets.

    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target data.
    X_validate (array-like): Validation feature data.
    y_validate (array-like): Validation target data.
    X_test (array-like): Test feature data (not used in this function).
    y_test (array-like): Test target data (not used in this function).

    Returns:
    pandas.DataFrame: A dataframe containing the model names, set (train or validate), accuracy, recall, and precision scores.
    """

    # create models list
    models = create_models(seed=123)

    # initialize results dataframe
    results = pd.DataFrame(columns=['model', 'set', 'accuracy'])

    # loop through models and fit/predict on train and validate sets
    for name, model in models:
        # fit the model with the training data
        model.fit(X_train, y_train)
        
        # make predictions with the training data
        train_predictions = model.predict(X_train)
        
        # calculate training accuracy, recall, and precision
        train_accuracy = accuracy_score(y_train, train_predictions)

        
        # make predictions with the validation data
        val_predictions = model.predict(X_validate)
        
        # calculate validation accuracy, recall, and precision
        val_accuracy = accuracy_score(y_validate, val_predictions)

        
        # append results to dataframe
        results = results.append({'model': name, 'set': 'train', 'accuracy': train_accuracy}, ignore_index=True)
        results = results.append({'model': name, 'set': 'validate', 'accuracy': val_accuracy}, ignore_index=True)

    return results


# test model function 
def run_gradient_boost(X_train, y_train, X_test, y_test):
    """
    Trains a Gradient Boosting Classifier on the given training data (X_train, y_train),
    makes predictions on the test data (X_test), and calculates accuracy, recall, and precision
    on the test set.

    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data target labels.
        X_test (array-like): Test data features.
        y_test (array-like): Test data target labels.

    Returns:
        pandas.DataFrame: A DataFrame containing the results of the model evaluation on the test set.
                          The DataFrame has the following columns:
                          - 'model': Name of the model used ('gradient_boosting').
                          - 'set': Indicates the data set evaluated ('test').
                          - 'accuracy': Accuracy score on the test set.
                          - 'recall': Weighted recall score on the test set.
                          - 'precision': Weighted precision score on the test set.
    """
    # Create and fit the Gradient Boosting model
    model = GradientBoostingClassifier(random_state=123)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    test_predictions = model.predict(X_test)

    # Calculate accuracy, recall, and precision on the test set
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions, average='weighted')
    test_precision = precision_score(y_test, test_predictions, average='weighted')

    # Create a results DataFrame
    results = pd.DataFrame({
        'model': ['gradient_boosting'],
        'set': ['test'],
        'accuracy': [test_accuracy]
    })

    return results

def run_support_vector(X_train, y_train, X_test, y_test):
    """
    Trains a Gradient Boosting Classifier on the given training data (X_train, y_train),
    makes predictions on the test data (X_test), and calculates accuracy, recall, and precision
    on the test set.

    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data target labels.
        X_test (array-like): Test data features.
        y_test (array-like): Test data target labels.

    Returns:
        pandas.DataFrame: A DataFrame containing the results of the model evaluation on the test set.
                          The DataFrame has the following columns:
                          - 'model': Name of the model used ('gradient_boosting').
                          - 'set': Indicates the data set evaluated ('test').
                          - 'accuracy': Accuracy score on the test set.
                          - 'recall': Weighted recall score on the test set.
                          - 'precision': Weighted precision score on the test set.
    """
    # Create and fit the Gradient Boosting model
    model = SVC(random_state=123)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    test_predictions = model.predict(X_test)

    # Calculate accuracy, recall, and precision on the test set
    test_accuracy = accuracy_score(y_test, test_predictions)
#     test_recall = recall_score(y_test, test_predictions, average='weighted')
#     test_precision = precision_score(y_test, test_predictions, average='weighted')

    # Create a results DataFrame
    results = pd.DataFrame({
        'model': ['support_vector'],
        'set': ['test'],
        'accuracy': [test_accuracy]
    })

    return results