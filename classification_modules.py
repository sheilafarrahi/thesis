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

    
def lr_model(data, n_folds):
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

                clf_svm = SVC(kernel='rbf', probability=True, decision_function_shape ='ovr', class_weight='balanced', C=c, gamma=g)
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
    
    # takes smallest gamma....check 
    
    return models_cost[0], models_gamma[0], result_df

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
            clf_lr = LogisticRegression(penalty ='l2', C=c, class_weight='balanced', multi_class='ovr')
            clf_lr.fit(X_train_scaled, y_train)
            prob = clf_lr.predict_proba(X_test_scaled)
            pred = clf_lr.predict(X_test_scaled)
            
            one_hot_matrix = list()
            for l in range(len(y_test)):
                one_hot_matrix.append(list((y_test[l] == clf_lr.classes_ )*1))
            loss = 0
            acc =0
            for j in range(len(y_test)):
                prob_row = prob[j]
                loss = loss - np.log(sum(one_hot_matrix[j]*prob_row))
                if y_test[j] == pred[j]:
                    acc = acc +1

            loss = loss/len(y_test)
            acc = acc / len(y_test)
            cv_error.append(loss)
            result.append( dict(zip(['cv_error','lambda','acc'],[loss, 1/c,acc]))) 
    result_df = pd.DataFrame(result)
    # find the best model, using one standard error rule
    res_agg = result_df.groupby(['lambda'], as_index=False).agg({'cv_error':['mean','std']})
    res_agg.columns = ['lambda','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    min_err_index = np.argmin(res_agg['mean'])
    threshold = res_agg['mean'][min_err_index] + res_agg['se'][min_err_index]
    models = res_agg.loc[res_agg['mean']<=threshold]
    models_lambda =list(models['lambda'])
    return models_lambda[0], result_df

    
def lr_model_m(data, n_folds, c_vector):
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
        lambda_, result_df = grid_search_lr(train, n_folds, c_vector)
    
        scaler_train = StandardScaler()
        scaler_train.fit(X_train)
        X_train_scaled = scaler_train.transform(X_train)
        X_test_scaled = scaler_train.transform(X_test)
        
        clf_lr = LogisticRegression(penalty ='l2', C=1/lambda_, class_weight='balanced', multi_class='ovr')    
        clf_lr.fit(X_train_scaled, y_train)
        y_pred = clf_lr.predict(X_test_scaled)
        score = accuracy_score(y_test, y_pred)
        result.append(score)
    return result

###########################################################
#                    Moments Approach                     #
###########################################################
def cv_samplesize_moments_svm(sample_size_list, nr_moments_list, dists, nr_sample_sets, n_folds, test_size, cost_vector, gamma_vector, transform = False, flex=False, standardize=False):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        if flex == True:
            samples = dm.get_samples_flex(dists, nr_sample_sets, i)
            for j in nr_moments_list:
                partial_moments = partial(dem.get_moments_partial, nr_moments=j)
                moments_res = samples['sample_set'].apply(partial_moments)
                moments_df = pd.DataFrame(moments_res.tolist())
                moments_df['label'] = samples['label']
                score = svm_model_m(moments_df, n_folds, cost_vector, gamma_vector)
                result_ = dict(zip(['score','nr_moments','sample_size'],[score, j, i]))
                result_df = pd.DataFrame(result_)
                result= result.append(result_df)
        else:
            if standardize == True:
                samples = dm.get_st_samples(dists, nr_sample_sets, i)
            else:
                samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
            for j in nr_moments_list:
                moments_df = dem.get_moments(samples, j)
                score = svm_model_m(moments_df, n_folds, cost_vector, gamma_vector)
                result_ = dict(zip(['score','nr_moments','sample_size'],[score, j, i]))
                result_df = pd.DataFrame(result_)
                result= result.append(result_df)
                
    res_agg = result.groupby(['nr_moments','sample_size'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_moments','sample_size','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg


def cv_samplesize_moments_svm_mm(sample_size_list, nr_moments_list, nr_sample_sets, nr_mm_dist, nr_modes, n_folds, test_size, cost_vector, gamma_vector):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_sample_sets, nr_modes, i)
        for j in nr_moments_list:
            moments_df = dem.get_moments(samples, j)
            score = svm_model_m(moments_df, n_folds, cost_vector, gamma_vector)
            result_ = dict(zip(['score','nr_moments','sample_size'],[score, j, i]))
            result_df = pd.DataFrame(result_)
            result = result.append(result_df)
            
    res_agg = result.groupby(['nr_moments','sample_size'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_moments','sample_size','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg

def cv_samplesize_moments_lr(sample_size_list, nr_moments_list, dists, nr_sample_sets, n_folds, test_size, C_vector, transform = False, flex=False, standardize=False):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        if flex == True:
            samples = dm.get_samples_flex(dists, nr_sample_sets, i)
            for j in nr_moments_list:
                partial_moments = partial(dem.get_moments_partial, nr_moments=j)
                moments_res = samples['sample_set'].apply(partial_moments)
                moments_df = pd.DataFrame(moments_res.tolist())
                moments_df['label'] = samples['label']
                score = lr_model_m(moments_df, n_folds, C_vector)
                result_ = dict(zip(['score','nr_moments','sample_size'],[score, j, i]))
                result_df = pd.DataFrame(result_)
                result= result.append(result_df)
        
        else:
            if standardize == True:
                samples = dm.get_st_samples(dists, nr_sample_sets, i)
            else:
                samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
            for j in nr_moments_list:
                moments_df = dem.get_moments(samples, j)
                score = lr_model_m(moments_df, n_folds, C_vector)
                result_ = dict(zip(['score','nr_moments','sample_size'],[score, j, i]))
                result_df = pd.DataFrame(result_)
                result= result.append(result_df)
    res_agg = result.groupby(['nr_moments','sample_size'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_moments','sample_size','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg

def cv_samplesize_moments_lr_mm(sample_size_list, nr_moments_list, nr_sample_sets, nr_mm_dist, nr_modes, n_folds, test_size, C_vector):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_sample_sets, nr_modes, i)
        for j in nr_moments_list:
            moments_df = dem.get_moments(samples, j)
            score = lr_model_m(moments_df, n_folds, C_vector)
            result_ = dict(zip(['score','nr_moments','sample_size'],[score, j, i]))
            result_df = pd.DataFrame(result_)
            result = result.append(result_df)
            
    res_agg = result.groupby(['nr_moments','sample_size'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_moments','sample_size','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg

def plot_cv_moments_v2(clf_result):
    clf_result['nr_features'] = clf_result['nr_moments'] + 1
    sample_size_list = clf_result['sample_size'].unique()
    fig, ax = plt.subplots()
    for i in range(len(sample_size_list)):
        data = clf_result.loc[clf_result['sample_size']==sample_size_list[i]]
        plt.plot(data['nr_features'], data['mean'], label=sample_size_list[i])
        plt.fill_between(data['nr_features'], data['mean']-data['se'], data['mean']+data['se'], alpha=0.2)

    plt.legend(title='Input Size',loc='lower left', ncol=3)
    plt.grid(color='#DDDDDD')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.set_yticks(list(np.linspace(0,1,11)))
    plt.ylim(0,1.1)
    plt.xlabel('Number of Constructed Features')
    plt.ylabel('Accuracy')
    plt.show()


###########################################################
#                        KDE & EDF                        #
###########################################################
def cv_numsteps_samplesize_svm(sample_size_list, num_steps_list, dists, nr_sample_sets, n_folds, test_size, method, cost_vector, gamma_vector, transform = False, flex = False):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc ='Completed'):
        if flex == True:
            samples = dm.get_samples_flex(dists, nr_sample_sets, i, 2)
            for j in num_steps_list:
                x = np.linspace(0,1,j)
                if method == 'kde':
                    partial_kde = partial(dem.get_kde_partial, x=x)
                    kde_res = samples['sample_set'].apply(partial_kde)
                    df = pd.DataFrame(kde_res.tolist())
                    df['label'] = samples['label']
                elif method == 'edf':
                    y = np.linspace(0.01,1,j)
                    partial_edf = partial(dem.get_edf_partial, y=y)
                    edf_res = samples['sample_set'].apply(partial_edf)
                    df = pd.DataFrame(edf_res.tolist())
                    df['label'] = samples['label']
                score = svm_model_m(df, n_folds, cost_vector, gamma_vector)
                result_ = dict(zip(['score','nr_steps','sample_size'],[score, j, i]))
                result_df = pd.DataFrame(result_)
                result= result.append(result_df)
        else:
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
                    y = np.linspace(0.01,0.99,j)
                    df = dem.get_edf_v2(samples, y)
                score = svm_model_m(df, n_folds, cost_vector, gamma_vector)
                result_ = dict(zip(['score','nr_steps','sample_size'],[score, j, i]))
                result_df = pd.DataFrame(result_)
                result = result.append(result_df)
    res_agg = result.groupby(['nr_steps','sample_size'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_steps','sample_size','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg

def cv_numsteps_samplesize_svm_mm(sample_size_list, num_steps_list, nr_sample_sets, nr_mm_dist, nr_modes, n_folds, test_size, method, cost_vector, gamma_vector):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_sample_sets, nr_modes, i)
        for j in num_steps_list:
            min_ = np.percentile(samples.iloc[:,:-1],2.5)
            max_ = np.percentile(samples.iloc[:,:-1],97.5)
            x = np.linspace(min_,max_,j)
            if method == 'kde':
                df = dem.get_kde(samples, x)
            elif method == 'edf':
                y = np.linspace(0.01,1,j)
                df = dem.get_edf_v2(samples, y)
            score = svm_model_m(df, n_folds, cost_vector, gamma_vector)
            result_ = dict(zip(['score','nr_steps','sample_size'],[score, j, i]))
            result_df = pd.DataFrame(result_)
            result = result.append(result_df)
    res_agg = result.groupby(['nr_steps','sample_size'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_steps','sample_size','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg
                
def cv_numsteps_samplesize_lr(sample_size_list, num_steps_list, dists, nr_sample_sets, n_folds, test_size, method, C_vector, transform = False, flex = False):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc ='Completed'):
        if flex == True:
            samples = dm.get_samples_flex(dists, nr_sample_sets, i, 2)
            for j in num_steps_list:
                x = np.linspace(0,1,j)
                if method == 'kde':
                    partial_kde = partial(dem.get_kde_partial, x=x)
                    kde_res = samples['sample_set'].apply(partial_kde)
                    df = pd.DataFrame(kde_res.tolist())
                    df['label'] = samples['label']
                elif method == 'edf':
                    y = np.linspace(0.01,1,j)
                    partial_edf = partial(dem.get_edf_partial, y=y)
                    edf_res = samples['sample_set'].apply(partial_edf)
                    df = pd.DataFrame(edf_res.tolist())
                    df['label'] = samples['label']
                score = lr_model_m(df, n_folds, C_vector)
                result_ = dict(zip(['score','nr_steps','sample_size'],[score, j, i]))
                result_df = pd.DataFrame(result_)
                result= result.append(result_df)
        else:
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
                    y = np.linspace(0.01,1,j)
                    df = dem.get_edf_v2(samples, y)
                score = lr_model_m(df, n_folds, C_vector)
                result_ = dict(zip(['score','nr_steps','sample_size'],[score, j, i]))
                result_df = pd.DataFrame(result_)
                result = result.append(result_df)
    res_agg = result.groupby(['nr_steps','sample_size'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_steps','sample_size','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg

def cv_numsteps_samplesize_lr_mm(sample_size_list, num_steps_list, nr_sample_sets, nr_mm_dist, nr_modes, n_folds, test_size, method, C_vector):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_sample_sets, nr_modes, i)
        for j in num_steps_list:
            min_ = np.percentile(samples.iloc[:,:-1],2.5)
            max_ = np.percentile(samples.iloc[:,:-1],97.5)
            x = np.linspace(min_,max_,j)
            if method == 'kde':
                df = dem.get_kde(samples, x)
            elif method == 'edf':
                y = np.linspace(0.01,1,j)
                df = dem.get_edf_v2(samples, y)
            score = lr_model_m(df, n_folds, C_vector)
            result_ = dict(zip(['score','nr_steps','sample_size'],[score, j, i]))
            result_df = pd.DataFrame(result_)
            result = result.append(result_df)
    res_agg = result.groupby(['nr_steps','sample_size'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_steps','sample_size','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg

def plot_cv_numsteps_samplesize_v2(clf_result):
    sample_size_list = clf_result['sample_size'].unique()
    clf_result['nr_features'] = clf_result['nr_steps'] + 1
    fig, ax = plt.subplots()
    for i in range(len(sample_size_list)):
        data = clf_result.loc[clf_result['sample_size']==sample_size_list[i]]
        plt.plot(data['nr_features'], data['mean'], label=sample_size_list[i])
        plt.fill_between(data['nr_features'], data['mean']-data['se'], data['mean']+data['se'], alpha=0.2)

    plt.legend(title='Input Size',loc='lower left', ncol=3)
    plt.grid(color='#DDDDDD')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.set_yticks(list(np.linspace(0,1,11)))
    plt.ylim(0,1.1)
    plt.xlabel('Number of Constructed Features')
    plt.ylabel('Accuracy')
    plt.show()
    
###########################################################
#                           ECF                           #
###########################################################
def cv_ecf_svm(sample_size_list, max_t_list, num_steps_list, dists, nr_sample_sets, n_folds, test_size, cost_vector, gamma_vector, transform = False, flex=False):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc ='Completed'):
        if flex == True:
            samples = dm.get_samples_flex(dists, nr_sample_sets, i, 2)
            for j in num_steps_list:
                for k in max_t_list:
                    partial_ecf = partial(dem.get_ecf_partial, max_t=k, steps=j)
                    ecf_res = samples['sample_set'].apply(partial_ecf)
                    df = pd.DataFrame(ecf_res.tolist())
                    df['label'] = samples['label']
                    score = svm_model_m(df, n_folds, cost_vector, gamma_vector)
                    result_ = dict(zip(['score','nr_steps','sample_size','max_t'],[score, j, i, k]))
                    result_df = pd.DataFrame(result_)
                    result = result.append(result_df)
        else:
            samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
            for j in num_steps_list:
                for k in max_t_list:
                    t = np.linspace(0.001, k, j)
                    df = dem.get_ecf(samples, t)
                    score = svm_model_m(df, n_folds, cost_vector, gamma_vector)
                    result_ = dict(zip(['score','nr_steps','sample_size','max_t'],[score, j, i, k]))
                    result_df = pd.DataFrame(result_)
                    result = result.append(result_df)
    res_agg = result.groupby(['nr_steps','sample_size','max_t'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_steps','sample_size','max_t','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg  


def cv_ecf_svm_mm(sample_size_list,max_t_list, num_steps_list, nr_sample_sets, nr_mm_dist, nr_modes, n_folds, test_size, cost_vector, gamma_vector):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_sample_sets, nr_modes, i)
        for j in num_steps_list:
            for k in max_t_list:
                t = np.linspace(0.001, k, j)
                df = dem.get_ecf(samples, t)
                score = svm_model_m(df, n_folds, cost_vector, gamma_vector)
                result_ = dict(zip(['score','nr_steps','sample_size','max_t'],[score, j, i, k]))
                result_df = pd.DataFrame(result_)
                result = result.append(result_df)
    res_agg = result.groupby(['nr_steps','sample_size','max_t'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_steps','sample_size','max_t','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg


    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc ='Completed'):
        if flex == True:
            samples = dm.get_samples_flex(dists, nr_sample_sets, i, 2)
            for j in num_steps_list:
                for k in max_t_list:
                    t = np.linspace(0.001, k, j)
                    partial_ecf = partial(dem.get_ecf_partial, x=x)
                    ecf_res = samples['sample_set'].apply(partial_ecf)
                    df = pd.DataFrame(ecf_res.tolist())
                    df['label'] = samples['label']                
                    score = lr_model_m(df, n_folds, C_vector)
                    result_ = dict(zip(['score','nr_steps','sample_size','max_t'],[score, j, i, k]))
                    result_df = pd.DataFrame(result_)
                    result= result.append(result_df)
        else:
            samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
            for j in num_steps_list:
                for k in max_t_list:
                    t = np.linspace(0.001, k, j)
                    df = dem.get_ecf(samples, x)
                    score = lr_model_m(df, n_folds, C_vector)
                    result_ = dict(zip(['score','nr_steps','sample_size','max_t'],[score, j, i, k]))
                    result_df = pd.DataFrame(result_)
                    result = result.append(result_df)
    res_agg = result.groupby(['nr_steps','sample_size','max_t'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_steps','sample_size','max_t','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg


def cv_ecf_lr(sample_size_list, max_t_list, num_steps_list, dists, nr_sample_sets, n_folds, test_size, C_vector, transform = False, flex=False):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc ='Completed'):
        if flex == True:
            samples = dm.get_samples_flex(dists, nr_sample_sets, i, 2)
            for j in num_steps_list:
                for k in max_t_list:
                    partial_ecf = partial(dem.get_ecf_partial, max_t=k, steps=j)
                    ecf_res = samples['sample_set'].apply(partial_ecf)
                    df = pd.DataFrame(ecf_res.tolist())
                    df['label'] = samples['label']
                    score = lr_model_m(df, n_folds, C_vector)
                    result_ = dict(zip(['score','nr_steps','sample_size','max_t'],[score, j, i, k]))
                    result_df = pd.DataFrame(result_)
                    result = result.append(result_df)
        else:
            samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
            for j in num_steps_list:
                for k in max_t_list:
                    t = np.linspace(0.001, k, j)
                    df = dem.get_ecf(samples, t)
                    score = lr_model_m(df, n_folds, C_vector)
                    result_ = dict(zip(['score','nr_steps','sample_size','max_t'],[score, j, i, k]))
                    result_df = pd.DataFrame(result_)
                    result = result.append(result_df)
    res_agg = result.groupby(['nr_steps','sample_size','max_t'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_steps','sample_size','max_t','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg 

def cv_ecf_lr_mm(sample_size_list, max_t_list, num_steps_list, nr_sample_sets, nr_mm_dist, nr_modes, n_folds, test_size, C_vector):
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_sample_sets, nr_modes, i)
        for j in num_steps_list:
            for k in max_t_list:
                t = np.linspace(0.001, k, j)
                df = dem.get_ecf(samples, t)
                score = lr_model_m(df, n_folds, C_vector)
                result_ = dict(zip(['score','nr_steps','sample_size','max_t'],[score, j, i, k]))
                result_df = pd.DataFrame(result_)
                result = result.append(result_df)
    res_agg = result.groupby(['nr_steps','sample_size','max_t'], as_index=False).agg({'score':['mean','std']})
    res_agg.columns = ['nr_steps','sample_size','max_t','mean','std']
    res_agg['se']=res_agg['std']/np.sqrt(n_folds)
    return res_agg

def plot_cv_ecf(clf_result):
    sample_size_list = clf_result['sample_size'].unique()
    clf_result['nr_features'] = clf_result['nr_steps'] + 1
    sns.set_style('whitegrid',{'grid.color':'#DDDDDD'})
    for i in (clf_result['max_t'].unique()):
        fig, ax = plt.subplots()
        sns.set_style('whitegrid',{'grid.color':'#DDDDDD'})
        for j in range(len(sample_size_list)):
            data = clf_result.loc[(clf_result['max_t']==i) & (clf_result['sample_size']==sample_size_list[j])]
            plt.plot(data['nr_features'], data['mean'], label=sample_size_list[j])
            plt.fill_between(data['nr_features'], data['mean']-data['se'], data['mean']+data['se'], alpha=0.2)
        ax.legend(loc='lower left', ncol=3, title='Input size')
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.title('for t in [0.001,%i]' %i)
        plt.ylabel('Accuracy')
        plt.xlabel('Number of constructed features')
        #plt.grid(color='#DDDDDD')
        plt.ylim(0,1.1)
        plt.show()