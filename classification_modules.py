import pandas as pd
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


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
    #scaler_test.fit(X_test)
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
        #param_grid = [{'C':np.logspace(-2,1,5),'gamma':np.logspace(-2,1,5), 'kernel':['rbf']},]
        param_grid = [{'C':[0.01, 0.25, 1, 5, 10],'gamma':[0.01, 0.25, 1, 5, 10], 'kernel':['rbf']},]
        optimal_params = GridSearchCV(SVC(), param_grid, cv=n_folds, verbose=0)
        
        # fit the model
        optimal_params.fit(X_train, y_train)
        cost = optimal_params.best_params_['C']
        gamma = optimal_params.best_params_['gamma']

        clf_svm = SVC(kernel='rbf', C=cost, gamma=gamma)
        clf_svm.fit(X_train, y_train)
        y_pred = clf_svm.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        result.append( dict(zip(['score','cost','gamma'],[score, cost, gamma])))
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
        clf_rr.fit(X_train, y_train)
        alpha= clf_rr.alpha_
        y_pred = clf_rr.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        result.append( dict(zip(['score','alpha'],[score, alpha])))
    result_df = pd.DataFrame(result)
    
    return result_df

#####################################################
#                   KDE and EDF                     #
#####################################################
def cv_numsteps_samplesize(sample_size_list, num_steps_list, dists, nr_sample, n_folds, method, classifier, transform=False):
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # method = kde or edf
    # classifier: integer value, 1: svm, 2: Ridge Regression
    # transform: set true for heavytail distribution
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc ='% completed'):
        samples = dm.get_samples(dists, nr_sample, i, transform = transform)
        for j in num_steps_list:
            if transform == False:
                x = np.linspace(0,1,j)
            elif transform == True:
                perc_95 = np.percentile(samples.iloc[:,:-1],95)
                x = np.linspace(0,perc_95,j)
                
            if method == 'kde':
                df = dem.get_kde(samples, x)
            elif method == 'edf':
                df = dem.get_edf(samples, x)
            if classifier == 1:
                result_ = svm_model(df, n_folds)
             
            elif classifier == 2:
                result_ = rr_model(df, n_folds)
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
    

###########################################################
#                    Moments Approach                     #
###########################################################
def cv_samplesize_moments(sample_size_list, nr_moments_list, dists, nr_sample, n_folds, classifier, transform = False):
    # sample_size_list: list of different sample sizes to test
    # nr_moments_list: list of different number of moments to test
    # dists: bounded_dists or heavytail_dists
    # nr_sample: 
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    # transform: set true for heavytail distribution
    result = pd.DataFrame()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_samples(dists, nr_sample, i, transform = transform)
        for j in nr_moments_list:
            moments_df = dem.get_moments(samples, j)
            if classifier == 1:
                result_ = svm_model(moments_df, n_folds)

            elif classifier == 2:
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
#                        ECF                        #
#####################################################            
def cv_ecf(sample_size_list, max_t_list, num_steps_list, dists, nr_sample, n_folds, classifier, transform = False):
    # cv_config: array of configuration for cross validation [test size, #splits for cross validation]
    # classifier: integer value, 1: svm, 2: Ridge Regression
    # transform: set true for heavytail distribution
    result = pd.DataFrame()
    for i in tqdm(sample_size_list):
        samples = dm.get_samples(dists, nr_sample, i, transform = transform)
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