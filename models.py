# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 20:00:31 2018

@author: Alvaro
"""
def classificationModels():

    models = {}
    
    ######################################
    # Logistic Regression
    ######################################
    from sklearn.linear_model import LogisticRegression
    models['Logistic Regression'] = {}
    models['Logistic Regression'] = LogisticRegression()
    
    ######################################
    # Random Forest
    ######################################
    from sklearn.ensemble import RandomForestClassifier
    models['Random Forests'] = RandomForestClassifier()
    
    ######################################
    # K Nearest Neighbors
    ######################################
    from sklearn.neighbors import KNeighborsClassifier
    models['K Nearest Neighbors'] = KNeighborsClassifier()
    
    ######################################
    # AdaBoost
    ######################################
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    models['Ada Boost'] = AdaBoostClassifier()
    
    ######################################
    # X Gradient Boosting
    ######################################
    from xgboost import XGBClassifier
    models['X Gradient Boosting'] = XGBClassifier()
    
    ######################################
    # Neural Networks: MultiLayer Perceptron
    ######################################
    from sklearn.neural_network import MLPClassifier
    models['MultiLayer Perceptron'] = MLPClassifier()
    
    return models


def regressionModels():

    models = {}
    
    ######################################
    # Linear Regression
    ######################################
    from sklearn.linear_model import LinearRegression
    models['Linear Regression'] = {}
    models['Linear Regression'] = LinearRegression()
    
    ######################################
    # Random Forest
    ######################################
    from sklearn.ensemble import RandomForestRegressor
    models['Random Forests'] = RandomForestRegressor()
    
    ######################################
    # K Nearest Neighbors
    ######################################
    from sklearn.neighbors import KNeighborsRegressor
    models['K Nearest Neighbors'] = KNeighborsRegressor()
    
    ######################################
    # AdaBoost
    ######################################
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    models['AdaBoost'] = AdaBoostRegressor()
    
    ######################################
    # X Gradient Boosting
    ######################################
    from xgboost import XGBRegressor
    models['X Gradient Boosting'] = XGBRegressor()
    
    ######################################
    # Neural Networks: MultiLayer Perceptron
    ######################################
    from sklearn.neural_network import MLPRegressor
    models['MultiLayer Perceptron'] = MLPRegressor()
    
    return models