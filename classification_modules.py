import pandas as pd
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from functools import partial

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


def prepare_data(train_data, test_data):
    X_train = train_data.iloc[:, :-1]
    X_test = test_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    y_test = test_data.iloc[:, -1]

    # Standardize features by removing the mean and scaling to unit variance
    scaler_train = StandardScaler()
    scaler_train.fit(X_train)
    X_train_scaled = scaler_train.transform(X_train)
    X_test_scaled = scaler_train.transform(X_test)
    X = pd.concat([X_train,X_test], ignore_index=True)
    y = pd.concat([y_train,y_test], ignore_index=True)
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test


def svm_model(data, n_folds):
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    result = list() # empty list to store the result
    skf = StratifiedKFold(n_splits = n_folds)
    
    # split data into test & train
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        
        # standardize the data
        scaler_train = StandardScaler()
        scaler_train.fit(X_train)
        X_train_scaled = scaler_train.transform(X_train)
        X_test_scaled = scaler_train.transform(X_test)

        # find the best hyperparams for the model
        param_grid = [{'C':[0.01, 0.25, 1, 5, 10],'gamma':[0.01, 0.25, 1, 5, 10], 'kernel':['rbf']},]
        optimal_params = GridSearchCV(SVC(), param_grid, cv=n_folds, verbose=0)
        
        # fit the model
        optimal_params.fit(X_train_scaled, y_train)
        cost = optimal_params.best_params_['C']
        gamma = optimal_params.best_params_['gamma']

        clf_svm = SVC(kernel='rbf', C=cost, gamma=gamma)
        clf_svm.fit(X_train_scaled, y_train)
        y_pred = clf_svm.predict(X_test_scaled)
        score = accuracy_score(y_test, y_pred)
        result.append( dict(zip(['score','cost','gamma'],[score, cost, gamma])))
    result_df = pd.DataFrame(result)
    return result_df

def svm_model_not_in_use(data, test_size, n_folds):
    result = list() # empty list to store the result
    train, test = split_data(data, test_size)
    X, y, X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(train, test)

    # find the best hyperparams for the model
    param_grid = [{'C':[0.01, 0.25, 1, 5, 10],'gamma':[0.01, 0.25, 1, 5, 10], 'kernel':['rbf']},]
    optimal_params = GridSearchCV(SVC(), param_grid, cv=n_folds, verbose=0)
        
    # fit the model
    optimal_params.fit(X_train_scaled, y_train)
    cost = optimal_params.best_params_['C']
    gamma = optimal_params.best_params_['gamma']
    clf_svm = SVC(kernel='rbf', decision_function_shape = 'ovr', class_weight='balanced', C=cost, gamma=gamma)
    clf_svm.fit(X_train_scaled, y_train)
    y_pred = clf_svm.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    result.append( dict(zip(['score','cost','gamma'],[score, cost, gamma])))
    result_df = pd.DataFrame(result)
    return result_df

def svm_train_err(data):
    nr_moments_list = data['nr_moments'].unique()
    sample_size_list = data['sample_size'].unique()
    
    for i in sample_size_list:
        for j in nr_moments_list:
            row = data.loc[(data['sample_size']==5) & (data['nr_moments']==2)]
            clf_svm = SVC(kernel='rbf', decision_function_shape = 'ovr', class_weight='balanced', C=row['cost'], gamma=row['gamma'])
       
            
            
def lr_model_not_in_use(data, test_size, n_folds):
    C = [0.01, 0.25, 1, 5, 10]
    result = list() # empty list to store the result
    train, test = split_data(data, test_size)
    X, y, X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(train, test)
    # find the best hyperparam for the model
    clf_lr = LogisticRegressionCV(Cs=C, cv=n_folds, penalty='l2', class_weight='balanced', multi_class ='ovr')
    clf_lr.fit(X_train_scaled, y_train)
    y_pred = clf_lr.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    lambda_ = 1/clf_lr.C_
    result.append( dict(zip(['score','lambda'],[score, lambda_])))
    result_df = pd.DataFrame(result)
    return result_df
    
    
def rr_model(data, n_folds):
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    result = list() # empty list to store the result
    skf = StratifiedKFold(n_splits = n_folds)

    # split data into test & train
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # standardize the data
        scaler_train = StandardScaler()
        scaler_train.fit(X_train)
        X_train_scaled = scaler_train.transform(X_train)
        X_test_scaled = scaler_train.transform(X_test)

        alphas = np.logspace(-2, 2, 10)
        clf_rr = RidgeClassifierCV(alphas)
        clf_rr.fit(X_train_scaled, y_train)
        alpha= clf_rr.alpha_
        y_pred = clf_rr.predict(X_test_scaled)
        score = accuracy_score(y_test, y_pred)
        result.append( dict(zip(['score','alpha'],[score, alpha])))
    result_df = pd.DataFrame(result)
    
    return result_df

###########################################################
#                    Moments Approach                     #
###########################################################
def cv_samplesize_moments(sample_size_list, nr_moments_list, dists, nr_sample_sets, n_folds, test_size, classifier, transform = False):
    # sample_size_list: list of different sample sizes to test
    # nr_moments_list: list of different number of moments to test
    # dists: bounded_dists or heavytail_dists
    # nr_sample_sets: 
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    # transform: set true for heavytail distribution
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
        for j in nr_moments_list:
            moments_df = dem.get_moments(samples, j)
            if classifier == 1:
                #result_ = svm_model(moments_df, test_size, n_folds)
                result_ = svm_model(moments_df, n_folds)
            elif classifier == 2:
                #result_ = lr_model(moments_df, test_size, n_folds)
                result_ = rr_model(moments_df, n_folds)
            result_['nr_moments'] = j
            result_['sample_size'] = i
            result = result.append(result_, ignore_index = True)
    return result

def cv_samplesize_moments_mm(sample_size_list, nr_moments_list, nr_sample_sets, nr_mm_dist, nr_modes, n_folds, test_size, classifier):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_sample_sets, nr_modes, i)
        for j in nr_moments_list:
            moments_df = dem.get_moments(samples, j)
            if classifier == 1:
                #result_ = svm_model(moments_df, test_size, n_folds)
                result_ = svm_model(moments_df, n_folds)
            elif classifier == 2:
                #result_ = lr_model(moments_df, test_size, n_folds)
                result_ = rr_model(moments_df, n_folds)
            result_['nr_moments'] = j
            result_['sample_size'] = i
            result = result.append(result_, ignore_index = True)
    return result

def cv_samplesize_moments_flex(sample_size_list, nr_moments_list, dists, nr_sample_sets, n_folds, test_size, classifier, transform = False):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_samples_flex(dists, nr_sample_sets, i)
        for j in nr_moments_list:
            partial_moments = partial(dem.get_moments_partial, nr_moments=j)
            moments_res = samples['sample_set'].apply(partial_moments)
            moments_df = pd.DataFrame(moments_res.tolist())
            moments_df['label'] = samples['label']
            if classifier == 1:
                #result_ = svm_model(moments_df, test_size, n_folds)
                result_ = svm_model(moments_df, n_folds)
            elif classifier == 2:
                #result_ = lr_model(moments_df, test_size, n_folds)
                result_ = rr_model(moments_df, n_folds)
            result_['nr_moments'] = j
            result_['sample_size'] = i
            result = result.append(result_, ignore_index = True)
        
    return result

def plot_cv_moments(clf_result):
    # clf_result is an output dataframe from classification that includes:
        # sample_size: list of different sample sizes to test
        # nr_moments: list of different number of moments to test
        # acc: list of accuracy
    ax = sns.lineplot(data = clf_result, x='nr_moments',y='score', hue='sample_size', ci = 'sd', legend='full', palette='muted')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
    plt.title('Moments Approach')
    plt.xlabel('Number of moments')
    plt.ylabel('Accuracy')
    plt.grid(color='#DDDDDD')
    plt.ylim(0,1.1)
    plt.show()
    

#####################################################
#                   KDE and EDF                     #
#####################################################
def cv_numsteps_samplesize(sample_size_list, num_steps_list, dists, nr_sample_sets, n_folds, test_size, method, classifier, transform=False):
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # method = kde or edf
    # classifier: integer value, 1: svm, 2: Ridge Regression
    # transform: set true for heavytail distribution
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc ='Completed'):
        samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
        for j in num_steps_list:
            if transform == False:
                x = np.linspace(0,1,j)
            elif transform == True:
                perc_95 = np.percentile(samples.iloc[:,:-1],95)
                x = np.linspace(0,perc_95,j)
            if method == 'kde':
                df = dem.get_kde(samples, x)
            elif method == 'edf':
                #df = dem.get_edf(samples, x)
                y = np.linspace(0.01,1,j)
                df = dem.get_edf_v2(samples, y)
            if classifier == 1:
                result_ = svm_model(df, test_size, n_folds)
            elif classifier == 2:
                result_ = lr_model(df, test_size, n_folds)
            result_['num_steps'] = j
            result_['sample_size'] = i
            result = result.append(result_, ignore_index = True)
    return result


def cv_numsteps_samplesize_flex(sample_size_list, num_steps_list, dists, nr_sample_sets, n_folds, test_size, method, classifier, transform=False):
    result = pd.DataFrame()
    if method=='kde':
        min_nr_sample = 2
    else:
        min_nr_sample=1
    for i in tqdm(sample_size_list, desc ='Completed'):
        samples = dm.get_samples_flex(dists, nr_sample_sets, i, min_nr_sample, transform=transform)
        for j in num_steps_list:
            if transform == False:
                x = np.linspace(0,1,j)
                y = np.linspace(0.01,1,j)
            elif transform == True:
                perc_95 = np.percentile(samples.iloc[:,:-1],95)
                x = np.linspace(0,perc_95,j)
                y = np.linspace(0.01,1,j)
            if method == 'kde':
                partial_kde = partial(dem.get_kde_partial, x=x)
                kde_res = samples['sample_set'].apply(partial_kde)
                df = pd.DataFrame(kde_res.tolist())
                df['label'] = samples['label']
            elif method == 'edf':
                partial_edf = partial(dem.get_edf_partial, y=y)
                edf_res = samples['sample_set'].apply(partial_edf)
                df = pd.DataFrame(edf_res.tolist())
                df['label'] = samples['label']
            if classifier == 1:
                result_ = svm_model(df, test_size, n_folds)
            elif classifier == 2:
                result_ = lr_model(df, test_size, n_folds)
            result_['num_steps'] = j
            result_['sample_size'] = i
            result = result.append(result_, ignore_index = True)
    return result


def plot_cv_numsteps_samplesize(clf_result, method):
    ax = sns.lineplot(data = clf_result, x='num_steps',y='score', hue='sample_size', ci = 'sd', legend='full', palette='muted')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.title(method)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Steps')
    plt.grid(color='#DDDDDD')
    plt.ylim(0,1.1)
    plt.show()
    
    
#####################################################
#                        ECF                        #
#####################################################            
def cv_ecf(sample_size_list, max_t_list, num_steps_list, dists, nr_sample_sets, n_folds, classifier, transform = False):
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    # transform: set true for heavytail distribution
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc ='Completed'):
        samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
        for j in num_steps_list:
            for k in max_t_list:
                t = np.linspace(k/j, k, j)
                ecf_df = dem.get_ecf(samples, t)
                if classifier == 1:
                    result_ = svm_model(ecf_df, n_folds)
      
                elif classifier == 2:
                    result_ = rr_model(ecf_df, n_folds)
   
                result_['sample_size'] = i
                result_['num_steps'] = j
                result_['max_t'] = k
                result = result.append(result_, ignore_index = True)
                
    return result

def plot_cv_ecf(clf_result):
    for i in (clf_result['sample_size'].unique()):
        fig, ax = plt.subplots()
        ax = sns.lineplot(data = clf_result.loc[clf_result['sample_size']==i], 
                          x='num_steps',y='score', hue='max_t', ci = 'sd', legend='full', palette='muted')
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Max t')
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.title('ECF, Sample Size =%i' %i)
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Steps')
        plt.grid(color='#DDDDDD')
        plt.ylim(0,1.1)
        plt.show()
        

########################################################### Manual grid search ###########################################################

def split_n_folds(data, n_folds):
    skf = StratifiedKFold(n_splits = n_folds)
    train_index_list = list()
    test_index_list = list()
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    for train_index, test_index in skf.split(X, y):
        train_index_list.append(train_index)   
        test_index_list.append(test_index)
    return train_index_list, test_index_list

def grid_search_svm(data, n_folds, cost, gamma):
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    train_index_list, test_index_list = split_n_folds(data, n_folds)
    result = list()
    for c in cost:
        for g in gamma:
            cv_error = list()
            for i in range(n_folds): 
                X_train = X.iloc[train_index_list[i]]
                y_train = y.iloc[train_index_list[i]]
                X_test = X.iloc[test_index_list[i]]
                y_test = y.iloc[test_index_list[i]]
                y_test = y_test.reset_index(drop=True)

                # standardize the data
                scaler_train = StandardScaler()
                scaler_train.fit(X_train)
                X_train_scaled = scaler_train.transform(X_train)
                X_test_scaled = scaler_train.transform(X_test)

                clf_svm = SVC(kernel='rbf', probability=True, decision_function_shape = 'ovr', class_weight='balanced', C=c, gamma=g)
                clf_svm.fit(X_train_scaled, y_train)
                prob = clf_svm.predict_proba(X_test_scaled)
                pred  = clf_svm.predict(X_test_scaled)

                # cross entropy loss
                loss = 0
                for j in range(len(y_test)):
                    prob_row=prob[j]
                    for k in range(len(clf_svm.classes_)):
                        if y_test[j]==clf_svm.classes_[k]:
                            loss = loss - np.log(prob_row[k])
                loss = loss/len(y_test)
                cv_error.append(loss)
                result.append( dict(zip(['cv_error','cost','gamma'],[loss, c, g])))   
    result_df = pd.DataFrame(result)
    
    # find the best model, using one standard error rule
    res_agg = result_df.groupby(['cost','gamma'], as_index=False).agg({'cv_error':['mean','std']})
    res_agg.columns = ['cost','gamma','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    min_err_index = np.argmin(res_agg['mean'])
    threshold = res_agg['mean'][min_err_index] + res_agg['se'][min_err_index]
    models = res_agg.loc[res_agg['mean']<=threshold]
    models_cost =list(models['cost'])
    models_gamma =list(models['gamma'])
    
    return models_cost [0], models_gamma [0], result_df

def svm_model_m(data, n_folds, cost_vector, gamma_vector):
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    train_index_list, test_index_list = split_n_folds(data, n_folds)
    result = list()
    for i in range(n_folds): 
        train = data.iloc[train_index_list[i]]
        test = data.iloc[test_index_list[i]]
        
        X_train = X.iloc[train_index_list[i]]
        y_train = y.iloc[train_index_list[i]]
        
        X_test = X.iloc[test_index_list[i]]
        y_test = y.iloc[test_index_list[i]]
        y_test = y_test.reset_index(drop=True)
    
        cost, gamma, result_df = grid_search_svm(train, n_folds, cost_vector, gamma_vector)
    
        scaler_train = StandardScaler()
        scaler_train.fit(X_train)
        X_train_scaled = scaler_train.transform(X_train)
        X_test_scaled = scaler_train.transform(X_test)
        
        clf_svm = SVC(kernel='rbf', C=cost, gamma=gamma)
        clf_svm.fit(X_train_scaled, y_train)
        y_pred = clf_svm.predict(X_test_scaled)
        score = accuracy_score(y_test, y_pred)
        result.append(score)
    
    return result


def grid_search_lr(data, n_folds, C):
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    train_index_list, test_index_list = split_n_folds(data, n_folds)
    result = list()
    for c in C:
        for i in range(n_folds):
            X_train = X.iloc[train_index_list[i]]
            y_train = y.iloc[train_index_list[i]]
            y_train_reset_index = y_train.reset_index(drop=True)
            # standardize the data
            scaler_train = StandardScaler()
            scaler_train.fit(X_train)
            X_train_scaled = scaler_train.transform(X_train)
            clf_lr = LogisticRegression(penalty ='l2', C=c, class_weight='balanced', multi_class='ovr')
            clf_lr.fit(X_train_scaled, y_train)
            prob = clf_lr.predict_proba(X_train_scaled)
            loss = 0
            for j in range(len(prob)):
                k = prob.argmax(axis=1)[j]
                l = prob.max(axis=1)[j]
                if clf_lr.classes_[k] != y_train_reset_index[j]:
                    loss = -np.log(l) + loss

            result.append( dict(zip(['fold','lambda','loss'],[i, 1/c, loss])))
            result_df = pd.DataFrame(result)
                
    res_agg = result_df.groupby(['lambda'], as_index=False).agg({'loss':['mean','std']})
    res_agg.columns = ['lambda','mean','std']
    min_err_index = np.argmin(res_agg['mean'])
    best_model_lambda = res_agg['lambda'][min_err_index]
    return best_model_lambda

def lr_model_m(data, test_size, lambda_):
    train, test = split_data(data, test_size)
    X, y, X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(train, test)
    result = list()
    clf_lr = LogisticRegression(penalty ='l2', C=1/lambda_, class_weight='balanced', multi_class='ovr')    
    clf_lr.fit(X_train_scaled, y_train)
    y_pred = clf_lr.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    result.append( dict(zip(['score','lambda'],[score, lambda_])))
    result_df = pd.DataFrame(result)
    return result_df

###########################################################
#                    Moments Approach                     #
###########################################################
def cv_samplesize_moments_svm(sample_size_list, nr_moments_list, dists, nr_sample_sets, n_folds, test_size, cost_vector, gamma_vector, transform = False):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
        for j in nr_moments_list:
            moments_df = dem.get_moments(samples, j)
            score = svm_model_m(moments_df, n_folds, cost_vector, gamma_vector)
            result_ = dict(zip(['score','nr_moments','sample_size'],[score, j, i]))
            result_df = pd.DataFrame(result_)
            result= result.append(result_df)
    return result


def cv_samplesize_moments_lr(sample_size_list, nr_moments_list, dists, nr_sample_sets, n_folds, test_size, C_vector, transform = False):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
        for j in nr_moments_list:
            moments_df = dem.get_moments(samples, j)
            lambda_ = grid_search_lr(moments_df, n_folds, C_vector)
            result_ = lr_model_m(moments_df, n_folds, lambda_)
            result_['nr_moments'] = j
            result_['sample_size'] = i
            result = result.append(result_, ignore_index = True)
    return result


def plot_cv_moments_v2(clf_result):
    # clf_result is an output dataframe from classification that includes:
        # sample_size: list of different sample sizes to test
        # nr_moments: list of different number of moments to test
        # acc: list of accuracy
    ax = sns.lineplot(data = clf_result, x='nr_moments',y='score', hue='sample_size', ci = 'sd', legend='full', palette='muted')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
    plt.xlabel('Number of moments')
    plt.ylabel('Accuracy')
    plt.grid(color='#DDDDDD')
    plt.ylim(0,1.1)
    plt.show()