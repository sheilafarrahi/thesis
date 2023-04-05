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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import distribution_modules as dm
import density_estimation_modules as dem
import importlib

importlib.reload(dm)
importlib.reload(dem)



def prepare_data(df, test_size):
    #X = df.drop([0,'dist'], axis=1)
    X = df.iloc[:, :-1]
    y = df.iloc[:,-1]
    
    # Scaling data
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = test_size, stratify = y, random_state=10)
    return X, y, X_train, X_test, y_train, y_test


def svm_model(df, test_size, cv, plot=0):
    X, y, X_train, X_test, y_train, y_test = prepare_data(df, test_size)
    param_grid = [
        {'C':np.logspace(0,1,10),
         'gamma':np.logspace(0,1,10), 
         'kernel':['rbf']},
    ]

    optimal_params = GridSearchCV(SVC(), param_grid,cv=cv, verbose=0)
    optimal_params.fit(X_train, y_train)

    clf_svm = SVC(random_state=100, C=optimal_params.best_params_['C'], gamma=optimal_params.best_params_['gamma'])
    clf_svm.fit(X_train, y_train)

    if plot==1:
        y_pred = clf_svm.predict(X_test)
        c_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(c_matrix, display_labels=clf_svm.classes_)
        disp.plot(cmap=plt.cm.Blues, colorbar=False, xticks_rotation='vertical')
        plt.show()
    
    scores = cross_val_score(clf_svm, X_train, y_train, cv=cv)
    return scores


def rr_model(df, test_size, cv, plot=0):
    X, y, X_train, X_test, y_train, y_test = prepare_data(df, test_size)
    alphas = np.logspace(0,20,50)
    
    clf_rr = RidgeClassifierCV(alphas)
    clf_rr.fit(X_train, y_train)
    
    if plot==1:
        y_pred = clf_rr.predict(X_test)
        c_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(c_matrix, display_labels=clf_rr.classes_)
        disp.plot(cmap=plt.cm.Blues, colorbar=False, xticks_rotation='vertical')
        plt.show()
    
    scores = cross_val_score(clf_rr, X_train, y_train, cv=cv)
    return scores



def cv_num_steps_step_size(step_size_list, num_steps_list, dists, sample_config, cv_config, classifier):
    # sample_config: 
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    sample_dict = dm.get_samples(dists, sample_config[1], sample_config[0])
    acc = list()
    
    for i in tqdm(range(len(num_steps_list))):
        num_t_steps = num_steps_list[i]
        acc_ = []

        for j in range(len(step_size_list)):
            t = np.arange(1, num_t_steps+1) * step_size_list[j]
            ecf_df = dem.get_ecf(sample_dict, t)
            if classifier == 1:
                score = svm_model(ecf_df,cv_config[0], cv_config[1])
            elif classifier == 2:
                score = rr_model(ecf_df,cv_config[0], cv_config[1])
            acc_.append(score.mean()) 
            sleep(0.1)
        acc.append(acc_)
    
    # plot
    fig, ax = plt.subplots(figsize=(10,8))
    for i in range(len(acc)):
        plt.plot(step_size_list, acc[i], label=str(num_steps_list[i]), alpha = 0.5)
        plt.title('accuracy for different step size and number of steps, sample size =%d' % sample_config[0])
        plt.xlabel('step size')
        plt.ylabel('accuracy')
        
        pos = ax.get_position()
        ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5), title='number of steps')
    plt.show()
    return acc


def cv_moments(nr_moments_list, data, cv_config, classifier):
    # nr_moments_list: list of different number of moments to test
    # data: 
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    
    acc = list()
    for i in tqdm(nr_moments_list):
        df = dem.get_moments_df(data, i)
        if classifier == 1:
            score = svm_model(df,cv_config[0], cv_config[1])
        elif classifier == 2:
            score = rr_model(df,cv_config[0], cv_config[1])
        acc.append(score.mean())
        sleep(0.1)
        
    # plot    
    plt.plot(nr_moments_list, acc, alpha = 0.5)
    plt.title('accuracy for diffrent number of moments')
    plt.xlabel('number of moments')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(min(nr_moments_list), max(nr_moments_list)+1, 1.0))
    plt.show()  

    return acc