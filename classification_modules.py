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

def get_default_plt_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def split_data(df, test_size): 
    # split synthetic data into test and train
    X = df.iloc[:, :-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = y, random_state=10)
    train = pd.concat([X_train,y_train], axis = 1)
    test = pd.concat([X_test,y_test], axis = 1)
    return train, test


def prepare_data(test_data, train_data):
    X_train = train_data.iloc[:, :-1]
    X_test = test_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    y_test = test_data.iloc[:, -1]

    # Scaling data
    scaler_train = StandardScaler()
    scaler_train.fit(X_train)
    X_train_scaled = scaler_train.transform(X_train)
    
    scaler_test = StandardScaler()
    scaler_test.fit(X_test)
    X_test_scaled = scaler_test.transform(X_test)

    X = pd.concat([X_train,X_test], ignore_index=True)
    y = pd.concat([y_train,y_test], ignore_index=True)
    
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test



def svm_model(test_data, train_data, cv_config, plot=0):
    X, y, X_train, X_test, y_train, y_test = prepare_data(test_data, train_data)
    param_grid = [
        {'C':np.logspace(-2,1,5),
         'gamma':np.logspace(-2,1,5), 
         'kernel':['rbf']},
    ]
    
    optimal_params = GridSearchCV(SVC(), param_grid,cv=cv_config[1], verbose=0)
    optimal_params.fit(X_train, y_train)
    cost = optimal_params.best_params_['C']
    gamma = optimal_params.best_params_['gamma']

    clf_svm = SVC(random_state=100, C=cost, gamma=gamma)
    clf_svm.fit(X_train, y_train)
    
    # plotting part out will be moved out of this function
    
    if plot==1:
        y_pred = clf_svm.predict(X_test)
        c_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(c_matrix, display_labels=clf_svm.classes_)
        disp.plot(cmap=plt.cm.Blues, colorbar=False, xticks_rotation='vertical')
        plt.show()
    
    scores = cross_val_score(clf_svm, X_test, y_test, cv=cv_config[1])
    return scores, cost, gamma


def rr_model(test_data, train_data, cv_config, plot=0):
    X, y, X_train, X_test, y_train, y_test = prepare_data(test_data, train_data)
    alphas = np.logspace(-2, 2, 10)
    clf_rr = RidgeClassifierCV(alphas)
    clf_rr.fit(X_train, y_train)
    
    # plotting part out will be moved out of this function
    y_pred = clf_rr.predict(X_test)
    if plot==1:
        c_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(c_matrix, display_labels=clf_rr.classes_)
        disp.plot(cmap=plt.cm.Blues, colorbar=False, xticks_rotation='vertical')
        plt.show()
    
    scores = cross_val_score(clf_rr, X_test, y_test, cv=cv_config[1])
    alpha= clf_rr.alpha_
    
    return scores, alpha

#####################################################
#                   KDE and EDF                     #
#####################################################
def cv_numsteps_samplesize(sample_size_list, num_steps_list, dists, nr_sample, cv_config, method, classifier, transform=False):
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # method = kde or edf
    # classifier: integer value, 1: svm, 2: Ridge Regression
    # transform: set true for heavytail distribution
    acc, std, cost, gamma, alpha = list(), list(), list(), list(), list()
    for i in tqdm(sample_size_list, desc ='% completed'):
        samples = dm.get_samples(dists, nr_sample, i, transform = transform)
        train_data, test_data = split_data(samples, cv_config[0])
        acc_, std_, cost_, gamma_, alpha_ = list(), list(), list(), list(), list()
        
        for j in num_steps_list:
            if transform == False:
                x = np.linspace(0,1,j)
            elif transform == True:
                perc_95 = np.percentile(train_data.iloc[:,:-1],95)
                x = np.linspace(0,perc_95,j)
            if method == 'kde':
                train_df = dem.get_kde(train_data, x)
                test_df = dem.get_kde(test_data, x)
            elif method == 'edf':
                train_df = dem.get_edf(train_data, x)
                test_df = dem.get_edf(test_data, x)
            if classifier == 1:
                score, c, g = svm_model(test_df, train_df,cv_config)
                cost_.append(c)
                gamma_.append(g)
            elif classifier == 2:
                score, a = rr_model(test_df, train_df,cv_config)
                alpha_.append(a)
            acc_.append(score.mean())
            std_.append(score.std())

        acc.append(acc_)
        std.append(std_)
        cost.append(cost_)
        gamma.append(gamma_)
        alpha.append(alpha_)
        
    result = dict(zip(['acc','std','cost','gamma','alpha','sample_size','num_steps'],
                          [acc, std, cost, gamma, alpha, sample_size_list, num_steps_list]))
        
    return result

def plot_cv_numsteps_samplesize(clf_result):
    # sample_size_list: list of different sample sizes to test
    # nr_moments_list: list of different number of moments to test
    # acc: list of accuracy
    # std: list of standard deviations
    colors = get_default_plt_colors()
    ax = plt.gca()
    for i, color in zip(range(len(clf_result['sample_size'])), colors):
        plt.plot(clf_result['num_steps'], clf_result['acc'][i], label=str(clf_result['sample_size'][i]), c = color, alpha = 0.7)
        plt.gca().fill_between(clf_result['num_steps'], [i-j for i,j in zip(clf_result['acc'][i], clf_result['std'][i])], 
                               [i+j for i,j in zip(clf_result['acc'][i], clf_result['std'][i])],
                               facecolor=color, alpha=0.1) 
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
        plt.title('Optimizing Number of Steps & Sample size to Maximize Accuracy')
        plt.xlabel('Number of Steps')
        plt.ylabel('Accuracy')
        plt.ylim(0,1.1)
    plt.show()

###########################################################
#                          Moments                        #
###########################################################
def cv_samplesize_moments(sample_size_list, nr_moments_list, dists, nr_sample, cv_config, classifier, transform = False):
    # sample_size_list: list of different sample sizes to test
    # nr_moments_list: list of different number of moments to test
    # dists: bounded_dists or heavytail_dists
    # nr_sample: 
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    # transform: set true for heavytail distribution
    acc, std, cost, gamma, alpha = list(), list(), list(), list(), list()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_samples(dists, nr_sample, i, transform = transform)
        train_data, test_data = split_data(samples, cv_config[0])
        acc_, std_, cost_, gamma_, alpha_ = list(), list(), list(), list(), list()

        for j in nr_moments_list:
            moments_test = dem.get_moments(test_data, j)
            moments_train = dem.get_moments(train_data, j)
            if classifier == 1:
                score, c, g = svm_model(moments_test, moments_train, cv_config)
                cost_.append(c)
                gamma_.append(g)
            elif classifier == 2:
                score, a = rr_model(moments_test, moments_train, cv_config)
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
        
    return result

def plot_cv_moments(clf_result):
    # clf_result is an output dict from classification that includes:
        # sample_size: list of different sample sizes to test
        # nr_moments: list of different number of moments to test
        # acc: list of accuracy
        # std: list of standard deviations
        # cost: list of optimal cost, used in SVM
        # gamma: list of optimal value of gamma, used in SVM
        # alpha: list of optimal value of alpha, used in logistic regression

    colors = get_default_plt_colors()
    ax = plt.gca()
    for i,color in zip(range(len(clf_result['sample_size'])), colors):
        plt.plot(clf_result['nr_moments'], clf_result['acc'][i], label=str(clf_result['sample_size'][i]), c = color, alpha = 0.7)
        plt.gca().fill_between(clf_result['nr_moments'], [i-j for i,j in zip(clf_result['acc'][i], clf_result['std'][i])], 
                               [i+j for i,j in zip(clf_result['acc'][i], clf_result['std'][i])],
                               facecolor=color, alpha=0.1)       
            
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
        plt.title('Optimizing Number of Moments & Sample Size to Maximize Accuracy')
        plt.xlabel('Moments')
        plt.ylabel('Accuracy')
        plt.ylim(0,1.1)
    plt.show()
    
    
def plot_cv_h_params(clf_result):
    if 'nr_moments' in clf_result.keys():
        x = clf_result['nr_moments']
    elif 'num_steps' in clf_result.keys():
        x = clf_result['num_steps']
    if bool(clf_result['cost'][0]):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,4.5))
        for i in range(len(clf_result['gamma'])):
            ax1.plot(x, clf_result['gamma'][i], label = str(clf_result['sample_size'][i]), alpha=0.7)
            ax1.set_title('Optimal Gamma')

            ax2.plot(x, clf_result['cost'][i], label = str(clf_result['sample_size'][i]), alpha=0.7)
            ax2.set_title('Optimal Cost')
            ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
        plt.show()
    else:
        fig, ax = plt.subplots()
        for i in range(len(clf_result['alpha'])):
            ax.plot(x, clf_result['alpha'][i], label = str(clf_result['sample_size'][i]), alpha=0.7)
            ax.set_title('Optimal Alpha')
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
            
            
#####################################################
#                        ECF                        #
#####################################################            
def cv_ecf(sample_size_list, max_t_list, num_steps_list, dists, sample_config, cv_config, classifier, transform = False):
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    # transform: set true for heavytail distribution
    size_result = len(sample_size_list)*len(max_t_list)*len(num_steps_list)
    result = pd.DataFrame(columns=['sample_size','num_steps','max_t','acc','std','cost','gamma','alpha'],index=range(0, size_result))
    c, g, a, row = 0, 0, 0, 0
    for i in tqdm(sample_size_list):
        samples = dm.get_samples(dists, sample_config[1], i, transform = transform)
        train_data, test_data = split_data(samples, cv_config[0])
        for j in num_steps_list:
            for k in max_t_list:
                t = np.linspace(k/j, k, j)
                train_df = dem.get_ecf(train_data, t)
                test_df = dem.get_ecf(test_data, t)
                
                if classifier == 1:
                    score, c, g = svm_model(test_df, train_df,cv_config)
                    acc = score.mean()
                    std = score.std()
                elif classifier == 2:
                    score, a = rr_model(test_df, train_df, cv_config)
                    acc = score.mean()
                    std = score.std()
                    
                result.iloc[row] = ([i, j, k, acc, std, c, g, a])
                row = row + 1
    return result

def plot_cv_ecf(clf_result):
    sample_size = clf_result['sample_size'].unique()
    max_t = clf_result['max_t'].unique()
    colors = get_default_plt_colors()

    for i in sample_size:
        fig, ax = plt.subplots()
        handles = []
        for j, color in zip(max_t, colors):
            df = clf_result.loc[(clf_result['sample_size'] == i) & (clf_result['max_t'] == j)]
            x = df['num_steps'].tolist()
            y = df['acc'].tolist()
            plt.plot(x, y, label = j , c = color, alpha = 0.7)
            plt.gca().fill_between(x,[i-j for i,j in zip(df['acc'], df['std'])], 
                                   [i+j for i,j in zip(df['acc'], df['std'])],
                                   facecolor=color, alpha=0.1) 
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Max t')
            plt.title('Optimizing Number of Steps & Max t to Maximize Accuracy, Sample Size =%i' %i)
            plt.xlabel('Number of Steps')
            plt.ylabel('Accuracy')
            plt.ylim(0,1.1)