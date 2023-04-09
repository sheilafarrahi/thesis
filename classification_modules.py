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
    X = df.iloc[:, :-1]
    y = df['dist']
    
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
        {'C':np.logspace(-2,1,15),
         'gamma':np.logspace(-2,1,15), 
         'kernel':['rbf']},
    ]
    
    optimal_params = GridSearchCV(SVC(), param_grid,cv=cv, verbose=0)
    optimal_params.fit(X_train, y_train)
    
    cost = optimal_params.best_params_['C']
    gamma = optimal_params.best_params_['gamma']

    clf_svm = SVC(random_state=100, C=cost, gamma=gamma)
    clf_svm.fit(X_train, y_train)

    if plot==1:
        y_pred = clf_svm.predict(X_test)
        c_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(c_matrix, display_labels=clf_svm.classes_)
        disp.plot(cmap=plt.cm.Blues, colorbar=False, xticks_rotation='vertical')
        plt.show()
    
    scores = cross_val_score(clf_svm, X_train, y_train, cv=cv)
    return scores, cost, gamma


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
    alpha= clf_rr.alpha_
    
    return scores, alpha

def cv_numsteps_samplesize(sample_size_list, num_steps_list, dists, nr_sample, cv_config, classifier, transform=False):
    acc, std, cost, gamma, alpha = list(), list(), list(), list(), list()
    
    for i in tqdm(sample_size_list, desc ='% completed'):
        samples = dm.get_samples(dists, nr_sample, i, transform = transform)
        acc_, std_, cost_, gamma_, alpha_ = list(), list(), list(), list(), list()

        for j in num_steps_list:
            x = np.linspace(0,1,j)
            df = dem.get_kde(samples, x)
            if classifier == 1:
                score, c, g = svm_model(df,cv_config[0], cv_config[1])
                cost_.append(c)
                gamma_.append(g)
            elif classifier == 2:
                score = rr_model(df,cv_config[0], cv_config[1])
            acc_.append(score.mean())
            std_.append(score.std())

        acc.append(acc_)
        std.append(std_)
        cost.append(cost_)
        gamma.append(gamma_)
        sleep(0.1)
    return acc, std, cost, gamma

def plot_cv_numsteps_samplesize(sample_size_list, num_steps_list, acc, std, errbar=0):
    # sample_size_list: list of different sample sizes to test
    # nr_moments_list: list of different number of moments to test
    # acc: list of accuracy
    # std: list of standard deviations
    # errbar: 1 if error bar should be included, 0 otherwise
    fig, ax = plt.subplots(figsize=(10,8))
    for i in range(len(sample_size_list)):
        if errbar == 0 :
            plt.plot(num_steps_list, acc[i], label=str(num_steps_list[i]), alpha = 0.5)
        elif errbar == 1:
            ax.errorbar(num_steps_list,acc[i],yerr= std[i], fmt='-', capsize=4, label=str(sample_size_list[i]), alpha=0.5)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title='Sample Size')
        plt.title('Accuracy for Diffrent Sample Size and Number of Steps')
        plt.xlabel('Number of Steps')
        plt.ylabel('Accuracy')
        plt.ylim(0,1.1)
    plt.show()


def cv_samplesize_moments(sample_size_list, nr_moments_list, dists, nr_sample, cv_config, classifier, transform = False):
    # sample_size_list: list of different sample sizes to test
    # nr_moments_list: list of different number of moments to test
    # dists: bounded_dists or heavytail_dists
    # nr_sample: 
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    acc, std, cost, gamma, alpha = list(), list(), list(), list(), list()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_samples(dists, nr_sample, i, transform = transform)
        acc_, std_, cost_, gamma_, alpha_ = list(), list(), list(), list(), list()

        for j in nr_moments_list:
            df = dem.get_moments_df(samples, j)
            if classifier == 1:
                score, c, g = svm_model(df,cv_config[0], cv_config[1])
                cost_.append(c)
                gamma_.append(g)
            elif classifier == 2:
                score, a = rr_model(df,cv_config[0], cv_config[1])
                alpha_.append(a)
            acc_.append(score.mean())
            std_.append(score.std())

        acc.append(acc_)
        std.append(std_)
        cost.append(cost_)
        gamma.append(gamma_)
        alpha.append(alpha_)
        
        result = dict(zip(['acc','std','cost','gamma','alpha','sample_size','nr_moments'],
                          [acc, std, cost, gamma, alpha, sample_size_list, nr_moments_list]))
        
        sleep(0.1)
    return result

def plot_cv_moments(clf_result, errbar=0):
    # clf_result is an output dict from classification that includes:
        # sample_size: list of different sample sizes to test
        # nr_moments: list of different number of moments to test
        # acc: list of accuracy
        # std: list of standard deviations
        # cost
        # gamma
        # alpha
    # errbar: 1 if error bar should be included, 0 otherwise
    fig, ax = plt.subplots(figsize=(10,8))
    for i in range(len(clf_result['sample_size'])):
        if errbar == 0 :
            plt.plot(clf_result['nr_moments'], 
                     clf_result['acc'][i], 
                     label=str(clf_result['sample_size'][i]), 
                     alpha = 0.5)
        elif errbar == 1:
            ax.errorbar(clf_result['nr_moments'],
                        clf_result['acc'][i],
                        yerr= clf_result['std'][i], 
                        fmt='-', 
                        capsize=4, 
                        label=str(clf_result['sample_size'][i]), 
                        alpha=0.5)
            
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title='Sample Size')
        plt.title('Accuracy for Diffrent Sample Size and Moments')
        plt.xlabel('Moments')
        plt.ylabel('Accuracy')
        plt.ylim(0,1.1)
    plt.show()
    

    
def plot_cv_h_params(clf_result):
    if bool(clf_result['cost'][0]):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
        for i in range(len(clf_result['gamma'])):
            ax1.plot(clf_result['nr_moments'], clf_result['gamma'][i], label = str(clf_result['sample_size'][i]))
            ax1.set_title('Optimal Gamma')

            ax2.plot(clf_result['nr_moments'], clf_result['cost'][i], label = str(clf_result['sample_size'][i]))
            ax2.set_title('Optimal Cost')
            ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title='Sample Size')
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(8,5))
        for i in range(len(clf_result['alpha'])):
            ax.plot(clf_result['nr_moments'], clf_result['alpha'][i], label = str(clf_result['sample_size'][i]))
            ax.set_title('Optimal Alpha')
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title='Sample Size')