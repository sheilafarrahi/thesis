import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from time import sleep
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

import distribution_modules as dm
import density_estimation_modules as dem
import importlib

importlib.reload(dm)
importlib.reload(dem)



def prepare_data(df, test_size):
    X = df.iloc[:, :-1]
    y = df.iloc[:,-1]
    
    # Scaling data
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = test_size, stratify = y, random_state=10)
    return X, y, X_train, X_test, y_train, y_test


def svm_model(df, cv_config, plot=0):
    X, y, X_train, X_test, y_train, y_test = prepare_data(df, cv_config[0])
    param_grid = [
        {'C':np.logspace(-2,2,15),
         'gamma':np.logspace(-3,1,15), 
         'kernel':['rbf']},
    ]

    optimal_params = GridSearchCV(SVC(), param_grid,cv=cv_config[1], verbose=0)
    optimal_params.fit(X_train, y_train)
    
    cost = optimal_params.best_params_['C']
    gamma = optimal_params.best_params_['gamma']

    clf_svm = SVC(random_state=100, C=cost, gamma=gamma, class_weight = 'balanced')
    clf_svm.fit(X_train, y_train)
    y_pred = clf_svm.predict(X_test)
    
    if plot==1:
        c_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(c_matrix, display_labels=clf_svm.classes_)
        fig, ax = plt.subplots(figsize=(10,10))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, xticks_rotation='vertical')
        plt.show()

    f1 = cross_val_score(clf_svm, X_train, y_train, cv=cv_config[1], scoring='f1_macro')
    return f1, cost, gamma


def lr_model(df, cv_config, plot=0):
    X, y, X_train, X_test, y_train, y_test = prepare_data(df, cv_config[0])
    alphas = np.logspace(0,20,50)
    
    clf_lr = RidgeClassifierCV(alphas, class_weight = 'balanced')
    clf_lr.fit(X_train, y_train)
    y_pred = clf_lr.predict(X_test)
    
    if plot==1:
        c_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(c_matrix, display_labels=clf_lr.classes_)
        fig, ax = plt.subplots(figsize=(10,10))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, xticks_rotation='vertical')
        plt.show()
    
    f1 =  cross_val_score(clf_lr, X_train, y_train, cv=cv_config[1], scoring='f1_macro')
    alpha = clf_lr.alpha_
    
    return f1, alpha


def cv_numsteps_edf(num_steps_list, data, cv_config, classifier):
    # num_steps_list: list of different number of steps to test
    # data: the graphwave data
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    
    acc, std, cost, gamma, alpha = list(), list(), list(), list(), list()
    for i in tqdm(num_steps_list):
        x = np.linspace(0,1,i)
        df = dem.get_edf(data, x)
        if classifier == 1:
            score, c, g = svm_model(df,cv_config)
            cost.append(c)
            gamma.append(g)
            
        elif classifier == 2:
            score, a = rr_model(df,cv_config)
            alpha.append(a)
            
        acc.append(score.mean())
        std.append(score.std())
        sleep(0.1)
        
    result = dict(zip(['acc','std','cost','gamma','alpha','num_steps_list'],[acc, std, cost, gamma, alpha, num_steps_list]))

    return result



def plot_cv_edf(svm_result, lr_result, errbar=0):
    # clf_result is an output dict from classification that includes:
        # num_steps_list: list of different number of steps to test
        # acc: list of accuracy
        # std: list of standard deviations
        # cost
        # gamma
        # alpha
    # errbar: 1 if error bar should be included, 0 otherwise
    fig, ax = plt.subplots()
    if errbar == 0 :
        plt.plot(svm_result['num_steps_list'], svm_result['acc'], label = 'svm')
        plt.plot(lr_result['num_steps_list'], lr_result['acc'], label = 'Logistic Regression')
    elif errbar == 1:
        ax.errorbar(svm_result['num_steps_list'], svm_result['acc'], yerr= svm_result['std'], capsize=4, label = 'svm')
        ax.errorbar(lr_result['num_steps_list'], lr_result['acc'], yerr= lr_result['std'], capsize=4, label = 'Logistic Regression')
            
    plt.title('Accuracy for Diffrent Number of Steps')
    plt.xlabel('Number of Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

###################################################
#             cross validation for KDE            #
###################################################

def cv_numsteps(num_steps_list, data, cv_config, classifier):
    # num_steps_list: list of different number of steps to test
    # data: the graphwave data
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    
    acc, std, cost, gamma, alpha = list(), list(), list(), list(), list()
    for i in tqdm(num_steps_list):
        x = np.linspace(0,1,i)
        df = dem.get_kde(data, x)
        if classifier == 1:
            score, c, g = svm_model(df,cv_config)
            cost.append(c)
            gamma.append(g)
            
        elif classifier == 2:
            score, a = rr_model(df,cv_config)
            alpha.append(a)
            
        acc.append(score.mean())
        std.append(score.std())
        sleep(0.1)
        
    result = dict(zip(['acc','std','cost','gamma','alpha','num_steps_list'],[acc, std, cost, gamma, alpha, num_steps_list]))

    return result


def plot_cv_kde(svm_result, lr_result, errbar=0):
    # clf_result is an output dict from classification that includes:
        # num_steps_list: list of different number of steps to test
        # acc: list of accuracy
        # std: list of standard deviations
        # cost
        # gamma
        # alpha
    # errbar: 1 if error bar should be included, 0 otherwise
    fig, ax = plt.subplots()
    if errbar == 0 :
        plt.plot(svm_result['num_steps_list'], svm_result['acc'], label = 'svm')
        plt.plot(lr_result['num_steps_list'], lr_result['acc'], label = 'Logistic Regression')
    elif errbar == 1:
        ax.errorbar(svm_result['num_steps_list'], svm_result['acc'], yerr= svm_result['std'], capsize=4, label = 'svm')
        ax.errorbar(lr_result['num_steps_list'], lr_result['acc'], yerr= lr_result['std'], capsize=4, label = 'Logistic Regression')
            
    plt.title('Accuracy for Diffrent Number of Steps')
    plt.xlabel('Number of Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
def plot_h_params_kde(clf_result):
    if bool(clf_result['cost']):
        fig, ax = plt.subplots()
        ax.plot(clf_result['num_steps_list'], clf_result['gamma'], label='Gamma', alpha=0.5)
        ax.plot(clf_result['num_steps_list'], clf_result['cost'], label='Cost', alpha=0.5)
        ax.set_title('Hyperparameters of SVM')
        ax.set_xlabel('Number of Steps')
        ax.legend()
        
    else:
        fig, ax = plt.subplots()
        ax.plot(clf_result['num_steps_list'], clf_result['alpha'], label = 'alpha')
        ax.set_title('Hyperparameter of Logistic Regression')
        ax.set_xlabel('Number of Steps')
        ax.legend()
        
    plt.show()
    
###################################################
#      cross validation for number of moments     #
###################################################
def cv_moments(nr_moments_list, data, cv_config, classifier):
    # nr_moments_list: list of different number of moments to test
    # data: the graphwave data
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    
    acc, std, cost, gamma, alpha = list(), list(), list(), list(), list()
    for i in tqdm(nr_moments_list):
        df = dem.get_moments_df(data, i)
        if classifier == 1:
            score, c, g = svm_model(df,cv_config)
            cost.append(c)
            gamma.append(g)
            
        elif classifier == 2:
            score, a = lr_model(df,cv_config)
            alpha.append(a)
            
        acc.append(score.mean())
        std.append(score.std())
        sleep(0.1)
        
    result = dict(zip(['acc','std','cost','gamma','alpha','nr_moments'],[acc, std, cost, gamma, alpha, nr_moments_list]))

    return result


def plot_cv_moments(svm_result, lr_result, errbar=0):
    # clf_result is an output dict from classification that includes:
        # nr_moments: list of different number of moments to test
        # acc: list of accuracy
        # std: list of standard deviations
        # cost
        # gamma
        # alpha
    # errbar: 1 if error bar should be included, 0 otherwise
    fig, ax = plt.subplots()
    if errbar == 0 :
        plt.plot(svm_result['nr_moments'], svm_result['acc'], label = 'svm')
        plt.plot(lr_result['nr_moments'], lr_result['acc'], label = 'Logistic Regression')
    elif errbar == 1:
        ax.errorbar(svm_result['nr_moments'], svm_result['acc'], yerr= svm_result['std'], capsize=4, label = 'svm')
        ax.errorbar(lr_result['nr_moments'], lr_result['acc'], yerr= lr_result['std'], capsize=4, label = 'Logistic Regression')
            
    plt.title('Accuracy for Diffrent Number of Moments')
    plt.xlabel('Number of Moments')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
def plot_cv_h_params(clf_result):
    if bool(clf_result['cost']):
        fig, ax = plt.subplots()
        ax.plot(clf_result['nr_moments'], clf_result['gamma'], label='Gamma', alpha=0.5)
        ax.plot(clf_result['nr_moments'], clf_result['cost'], label='Cost', alpha=0.5)
        ax.set_title('Hyperparameters of SVM')
        ax.set_xlabel('Number of Moments as Features')
        ax.legend()
        
    else:
        fig, ax = plt.subplots()
        ax.plot(clf_result['nr_moments'], clf_result['alpha'], label = 'alpha')
        ax.set_title('Hyperparameter of Logistic Regression')
        ax.set_xlabel('Number of Moments as Features')
        ax.legend()
  
        
    plt.show()