'''
A module to train a model for customer churn prediction.

Author: Italo
Date: September 27, 2022
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Creates Churn column and plot histogram
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    ax = df['Churn'].hist()
    ax.figure.savefig('./images/eda/churn.png')

    # Customer_Age histogram
    plt.figure(figsize=(20, 10))
    ax = df['Customer_Age'].hist()
    ax.figure.savefig('./images/eda/customer-age.png')

    # Marital_Status histogram
    plt.figure(figsize=(20, 10))
    ax = df.Marital_Status.value_counts('normalize').plot(kind='bar')
    ax.figure.savefig('./images/eda/marital-status.png')

    # Total_Trans_Ct Histogram
    plt.figure(figsize=(20, 10))
    ax = sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    ax.figure.savefig('./images/eda/total-trans-ct.png')

    # Correlation Matrix
    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    ax.figure.savefig('./images/eda/heatmap-correlation.png')


def encoder_helper(df, category_lst, response="_Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
                [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # Create a new column with Churn mean for each col in category_lst
    for col in category_lst:
        df[col + response] = df.groupby(col)["Churn"].transform(np.mean)

    return df


def perform_feature_engineering(df, response="_Churn"):
    '''
    input:
              df: pandas dataframe
              response: string of response name
                [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # Columns to keep as features
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender' + response,
        'Education_Level' + response,
        'Marital_Status' + response,
        'Income_Category' + response,
        'Card_Category' + response]

    # Defining X
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]

    # Target
    y = df['Churn']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Random Forest Classification Report
    plt.figure(figsize=(7, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/clf-rep-rf.png')

    # Logistic Regression Classification Report
    plt.figure(figsize=(7, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/clf-rep-lr.png')


def feature_importance_plot(model, X_test, y_test, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_test: X testing data
            y_test: y testing data
            output_pth: path to store the figure

    output:
            None
    '''
    # ROC-AUC
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig(output_pth + "roc.png")

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_test.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_test.shape[1]), importances[indices])

    # Add feature names as x-axis labels and save fig
    plt.xticks(range(X_test.shape[1]), names, rotation=90)
    plt.savefig(output_pth + "feat-imp.png")


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # Classificator to be used
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Param grid for Random Forest
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Grid Search and fit Random Forest
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Fit Linear Regression
    lrc.fit(X_train, y_train)

    # Predictions for both
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Dump both models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Save Classification Report to images folder
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Save ROC and Feature Importance to images folder
    feature_importance_plot(cv_rfc.best_estimator_,
                            X_test,
                            y_test,
                            "./images/results/")


if __name__ == "__main__":
    # Loads bank_data and perform EDA
    df = import_data("./data/bank_data.csv")
    perform_eda(df)

    # Encoding categories
    category_lst = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"
    ]
    df = encoder_helper(df, category_lst)

    # Train/test split
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)

    # Model training and saving as joblib
    train_models(X_train, X_test, y_train, y_test)
