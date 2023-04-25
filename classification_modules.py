import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from time import sleep
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

import distribution_modules as dm
import density_estimation_modules as dem
import importlib

importlib.reload(dm)
importlib.reload(dem)



def prepare_data(test_data, train_data):
    X_train = train_data.iloc[:, :-1]
    X_test = train_data.iloc[:, :-1]
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


def gbm_model(test_data, train_data, gbm_config, cv_config, plot=0):
    X, y, X_train, X_test, y_train, y_test = prepare_data(test_data, train_data)    
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    
    clf_gbm = GradientBoostingClassifier(n_estimators=gbm_config[0], learning_rate=0.1, max_depth=gbm_config[1], random_state=0)
    clf_gbm.fit(X_train, encoded_y_train)
    y_pred_encoded = clf_gbm.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred_encoded)
        
    f1 = cross_val_score(clf_gbm, X_train, encoded_y_train, cv=cv_config[1], scoring='f1_macro')
    return f1

#######################################################
#                         SVM                         #
#######################################################

def svm_model(test_data, train_data, cv_config, plot=0):
    X, y, X_train, X_test, y_train, y_test = prepare_data(test_data, train_data)
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
    
    if plot==1:
        y_pred = clf_svm.predict(X_test)
        c_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(c_matrix, display_labels=clf_svm.classes_)
        fig, ax = plt.subplots(figsize=(10,10))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, xticks_rotation='vertical')
        plt.show()

    f1 = cross_val_score(clf_svm, X_test, y_test, cv=cv_config[1], scoring='f1_macro')
    return f1, cost, gamma


#######################################################
#                  Logistic Regression                #
#######################################################
def lr_model(test_data, train_data, cv_config, plot=0):
    X, y, X_train, X_test, y_train, y_test = prepare_data(test_data, train_data)
    alphas = np.logspace(0,20,50)
    clf_lr = RidgeClassifierCV(alphas, class_weight = 'balanced')
    clf_lr.fit(X_train, y_train)
    
    if plot==1:
        y_pred = clf_lr.predict(X_test)
        c_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(c_matrix, display_labels=clf_lr.classes_)
        fig, ax = plt.subplots(figsize=(10,10))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, xticks_rotation='vertical')
        plt.show()
    
    f1 =  cross_val_score(clf_lr, X_test, y_test, cv=cv_config[1], scoring='f1_macro')
    alpha = clf_lr.alpha_
    return f1, alpha


###################################################
#      cross validation for number of moments     #
###################################################
def cv_moments(nr_moments_list, test_data, train_data, cv_config, classifier):
    # nr_moments_list: list of different number of moments to test
    # data: the graphwave data
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression 
    
    f1, std, cost, gamma, alpha = list(), list(), list(), list(), list()
    for i in tqdm(nr_moments_list):
        moments_test = dem.get_moments(test_data, i)
        moments_train = dem.get_moments(train_data, i)
        if classifier == 1:
            f1_, c, g = svm_model(moments_test, moments_train, cv_config)
            cost.append(c)
            gamma.append(g)
            
        elif classifier == 2:
            f1_, a = lr_model(moments_test, moments_train, cv_config)
            alpha.append(a)
            
        f1.append(f1_.mean())
        std.append(f1_.std())
        sleep(0.1)
        
    result = dict(zip(['f1','std','cost','gamma','alpha','nr_moments'],[f1, std, cost, gamma, alpha, nr_moments_list]))

    return result


def plot_cv_moments(svm_result, lr_result):
    # clf_result is an output dict from classification that includes:
        # nr_moments: list of different number of moments to test
        # acc: list of accuracy
        # std: list of standard deviations
        # cost
        # gamma
        # alpha
    fig, ax = plt.subplots()
    plt.plot(svm_result['nr_moments'], svm_result['f1'], label = 'svm')
    plt.gca().fill_between(svm_result['nr_moments'], 
                           [i-j for i,j in zip(svm_result['f1'], svm_result['std'])], 
                           [i+j for i,j in zip(svm_result['f1'], svm_result['std'])],
                           alpha=0.1) 
        
    plt.plot(lr_result['nr_moments'], lr_result['f1'], label = 'Logistic Regression')
    plt.gca().fill_between(lr_result['nr_moments'], 
                           [i-j for i,j in zip(lr_result['f1'], lr_result['std'])], 
                           [i+j for i,j in zip(lr_result['f1'], lr_result['std'])],
                           alpha=0.1) 
  
            
    plt.title('Optimizing Number of Moments to Maximize f1 Score')
    plt.xlabel('Number of Moments')
    plt.ylabel('f1 score')
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.show()
    
    
def plot_cv_h_params(clf_result):
    if bool(clf_result['cost']):
        fig, ax = plt.subplots()
        ax.plot(clf_result['nr_moments'], clf_result['gamma'], label='Gamma', alpha=0.5)
        ax.plot(clf_result['nr_moments'], clf_result['cost'], label='Cost', alpha=0.5)
        ax.set_title('Hyperparameters of SVM')
        ax.set_xlabel('Number of Moments as Features')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        
    else:
        fig, ax = plt.subplots()
        ax.plot(clf_result['nr_moments'], clf_result['alpha'], label = 'alpha')
        ax.set_title('Hyperparameter of Logistic Regression')
        ax.set_xlabel('Number of Moments as Features')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
  
    plt.show()

def cv_moments_gbm(nr_moments_list, test_data, train_data, gbm_config, cv_config):
    # nr_moments_list: list of different number of moments to test
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # gbm_config = [n_estimators_list, max_depth_list]
    
    f1, std = list(), list()
    for n in tqdm(gbm_config[0]):
        f1_, std_ = list(), list()
        for m in gbm_config[1]:
            f1__, std__ = list(), list()
            for i in nr_moments_list:
                moments_test = dem.get_moments(test_data, i)
                moments_train = dem.get_moments(train_data, i)
                gbm_config_ = [m,n]
                f1_score = gbm_model(moments_test, moments_train, gbm_config_, cv_config)
                f1__.append(f1_score.mean())
                std__.append(f1_score.std())

            f1_.append(f1__)
            std_.append(std__)

        f1.append(f1_)
        std.append(std_)
        sleep(0.1)

    result = dict(zip(['f1','std','nr_moments','n_estimators', 'max_depth'],
                      [f1, std, nr_moments_list, gbm_config[0], gbm_config[1]]))

    return result
###################################################
#             cross validation for KDE            #
###################################################
def cv_numsteps(num_steps_list, test_data, train_data, cv_config, classifier):
    # num_steps_list: list of different number of steps to test
    # data: the graphwave data
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    
    f1, std, cost, gamma, alpha = list(), list(), list(), list(), list()
    for i in tqdm(num_steps_list):
        x = np.linspace(0,1,i)
        test_kde = dem.get_kde(test_data, x)
        train_kde = dem.get_kde(train_data, x)
        if classifier == 1:
            f1_, c, g = svm_model(test_kde, train_kde, cv_config)
            cost.append(c)
            gamma.append(g)
            
        elif classifier == 2:
            f1_, a = lr_model(test_kde, train_kde, cv_config)
            alpha.append(a)
            
        f1.append(f1_.mean())
        std.append(f1_.std())
        sleep(0.1)
        
    result = dict(zip(['f1','std','cost','gamma','alpha','num_steps_list'],[f1, std, cost, gamma, alpha, num_steps_list]))
    return result


def plot_cv_kde(svm_result, lr_result):
    # clf_result is an output dict from classification that includes:
        # num_steps_list: list of different number of steps to test
        # acc: list of accuracy
        # std: list of standard deviations
        # cost
        # gamma
        # alpha
    fig, ax = plt.subplots()
    plt.plot(svm_result['num_steps_list'], svm_result['f1'], label = 'svm')
    plt.gca().fill_between(svm_result['num_steps_list'], 
                           [i-j for i,j in zip(svm_result['f1'], svm_result['std'])], 
                           [i+j for i,j in zip(svm_result['f1'], svm_result['std'])],
                           alpha=0.1) 
        
    plt.plot(lr_result['num_steps_list'], lr_result['f1'], label = 'Logistic Regression')
    plt.gca().fill_between(lr_result['num_steps_list'], 
                           [i-j for i,j in zip(lr_result['f1'], lr_result['std'])], 
                           [i+j for i,j in zip(lr_result['f1'], lr_result['std'])],
                           alpha=0.1) 
            
    plt.title('Optimizing Number of Steps to Maximize f1 Score')
    plt.xlabel('Number of Steps')
    plt.ylabel('f1 Score')
    plt.legend()
    plt.show()
    
def cv_kde_gbm(num_steps_list, test_data, train_data, gbm_config, cv_config):
    # num_steps_list: : list of different number of steps to test
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # gbm_config = [n_estimators_list, max_depth_list]
    
    f1, std = list(), list()
    for n in tqdm(gbm_config[0]):
        f1_, std_ = list(), list()
        for m in gbm_config[1]:
            f1__, std__ = list(), list()
            for i in num_steps_list:
                x = np.linspace(0,1,i)
                kde_test = dem.get_kde(test_data, x)
                kde_train = dem.get_kde(train_data, x)
                gbm_config_ = [m,n]
                f1_score = gbm_model(kde_test, kde_train, gbm_config_, cv_config)
                f1__.append(f1_score.mean())
                std__.append(f1_score.std())

            f1_.append(f1__)
            std_.append(std__)

        f1.append(f1_)
        std.append(std_)
        sleep(0.1)

    result = dict(zip(['f1','std','num_steps','n_estimators', 'max_depth'],
                      [f1, std, num_steps_list, gbm_config[0], gbm_config[1]]))

    return result
###################################################
#             cross validation for EDF            #
###################################################
def cv_numsteps_edf(num_steps_list, test_data, train_data, cv_config, classifier):
    # num_steps_list: list of different number of steps to test
    # test_data, train_data: the graphwave test and train data
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    
    f1, std, cost, gamma, alpha = list(), list(), list(), list(), list()
    for i in tqdm(num_steps_list):
        x = np.linspace(0,1,i)
        test_edf = dem.get_edf(test_data, x)
        train_edf = dem.get_edf(train_data, x)
        if classifier == 1:
            f1_, c, g = svm_model(test_edf, train_edf, cv_config)
            cost.append(c)
            gamma.append(g)
            
        elif classifier == 2:
            f1_, a = lr_model(test_edf, train_edf, cv_config)
            alpha.append(a)
            
        f1.append(f1_.mean())
        std.append(f1_.std())
        sleep(0.1)
        
    result = dict(zip(['f1','std','cost','gamma','alpha','num_steps_list'],[f1, std, cost, gamma, alpha, num_steps_list]))
    return result



def plot_cv_edf(svm_result, lr_result):
    # clf_result is an output dict from classification that includes:
        # num_steps_list: list of different number of steps to test
        # acc: list of accuracy
        # std: list of standard deviations
        # cost
        # gamma
        # alpha
    # errbar: 1 if error bar should be included, 0 otherwise
    fig, ax = plt.subplots()
    plt.plot(svm_result['num_steps_list'], svm_result['f1'], label = 'svm')
    plt.gca().fill_between(svm_result['num_steps_list'], 
                           [i-j for i,j in zip(svm_result['f1'], svm_result['std'])], 
                           [i+j for i,j in zip(svm_result['f1'], svm_result['std'])],
                           alpha=0.1) 
        
    plt.plot(lr_result['num_steps_list'], lr_result['f1'], label = 'Logistic Regression')
    plt.gca().fill_between(lr_result['num_steps_list'], 
                           [i-j for i,j in zip(lr_result['f1'], lr_result['std'])], 
                           [i+j for i,j in zip(lr_result['f1'], lr_result['std'])],
                           alpha=0.1) 
    
    plt.title('Optimizing Number of Steps to Maximize f1 Score')
    plt.xlabel('Number of Steps')
    plt.ylabel('f1 Score')
    plt.legend()
    plt.show()
    
    
def cv_edf_gbm(num_steps_list, test_data, train_data, gbm_config, cv_config):
    # num_steps_list: : list of different number of steps to test
    # test_data, train_data: the graphwave test and train data
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # gbm_config = [n_estimators_list, max_depth_list]
    
    f1, std = list(), list()
    for n in tqdm(gbm_config[0]):
        f1_, std_ = list(), list()
        for m in gbm_config[1]:
            f1__, std__ = list(), list()
            for i in num_steps_list:
                x = np.linspace(0,1,i)
                edf_test = dem.get_edf(test_data, x)
                edf_train = dem.get_edf(train_data, x)
                gbm_config_ = [m,n]
                f1_score = gbm_model(edf_test, edf_train, gbm_config_, cv_config)
                f1__.append(f1_score.mean())
                std__.append(f1_score.std())

            f1_.append(f1__)
            std_.append(std__)

        f1.append(f1_)
        std.append(std_)
        sleep(0.1)

    result = dict(zip(['f1','std','num_steps','n_estimators', 'max_depth'],
                      [f1, std, num_steps_list, gbm_config[0], gbm_config[1]]))

    return result
###################################################
#             cross validation for ECF            #
###################################################
def cv_ecf(num_steps_list, step_size_list, test_data, train_data, cv_config, classifier):
    # num_steps_list: list of different number of steps to test
    # test_data, train_data: the graphwave test and train data
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    size_result = len(step_size_list)*len(num_steps_list)
    result = pd.DataFrame(columns=['num_steps','step_size','f1','std','cost','gamma','alpha'],index=range(0, size_result))
    c, g, a, row = 0, 0, 0, 0
    
    for i in tqdm(num_steps_list):
        for j in step_size_list:
            t = np.arange(1, i+1) * j
            test_df = dem.get_ecf(test_data, t)
            train_df = dem.get_ecf(train_data, t)
            
            if classifier == 1:
                f1_score, c, g = svm_model(test_df, train_df, cv_config)
                f1 = f1_score.mean()
                std = f1_score.std()

            elif classifier == 2:
                f1_score, a = lr_model(test_df, train_df, cv_config)
                f1 = f1_score.mean()
                std = f1_score.std()
            
            result.iloc[row] = ([i, j, f1, std, c, g, a])
            row = row + 1
        
    return result


def plot_cv_ecf(svm_result, lr_result):
    # clf_result is an output dict from classification that includes:
        # num_steps_list: list of different number of steps to test
        # acc: list of accuracy
        # std: list of standard deviations
        # cost
        # gamma
        # alpha
    # errbar: 1 if error bar should be included, 0 otherwise
    fig, ax = plt.subplots()
    for i in range(len(step_size_list)):
        if errbar == 0 :
            plt.plot(num_steps_list, f1[i], label=str(num_steps_list[i]), alpha = 0.5)
        elif errbar == 1:
            ax.errorbar(num_steps_list,f1[i],yerr= std[i], fmt='-', capsize=4, label=str(sample_size_list[i]), alpha=0.5)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title='Sample Size')
        plt.title('Optimizing Number of Steps and step size to Maximize f1 Score')
        plt.xlabel('Number of Steps')
        plt.ylabel('f1 Score')
        
    plt.show()



def plot_h_params_edf(clf_result):
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