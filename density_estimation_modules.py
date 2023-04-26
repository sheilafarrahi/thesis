import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF
def get_default_plt_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


##########################################
#            methods of moments          #
##########################################

def get_moments(df, nr_moments):
    # df: a dictionary containing samples of different distribution including preselected parameters
    # nr_moments: desired number of moments to be calculated
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    m1 = np.mean(X, axis=1)
    m1_df = pd.DataFrame(m1, columns=['m1'])
    m1_df = m1_df.reset_index(drop=True)
    moments = np.zeros((len(X), nr_moments - 1)) # array to store moments after mean

    for i in range(2,nr_moments+1): #calculate from 2nd moment
        moments[:,i-2] = stats.moment(X, i, axis=1)

    moments_df = pd.DataFrame(moments, columns=['m'+str(j) for j in range(2,i+1)])
    df = pd.concat([m1_df,moments_df], axis=1)
    df['label'] = y.values.tolist()
    
    return df


def get_histogram_of_moments(df):
    distrubtions = df['label'].unique()
    for moment_name in df.columns[:-1]:
        fig, ax = plt.subplots()
        for distr_name in distrubtions:
            moments = df.loc[df['label'] == distr_name, moment_name]
            ax.hist(moments, density=True, histtype='stepfilled', bins='auto', alpha=0.75, label=distr_name)
        plt.title('moment: ' + moment_name)
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
        #plt.xlim(0,0.6)
        #plt.ylim(0,190)
        

##########################################
#        Kernel density estimation       #
##########################################

def get_kde(df, x):
    # samples_dict: a dictinary containing samples from different distribution including preselected parameters
    # x: array to do kde
    y_estimates = list()
    
    for i in range(len(df)):
        X = df.iloc[i,:-1]
        kde = stats.gaussian_kde(list(X))
        values = x
        y_estimates.append(kde(values))

    kde_df = pd.DataFrame(y_estimates)  
    kde_df['label'] = df.iloc[:,-1].tolist()

    return kde_df 
    

def get_kde_plot(df, x):
    names = df.iloc[:,-1].unique()
    fig, ax = plt.subplots()
    colors = get_default_plt_colors()
    handles = []
    for name, color in zip(names, colors):  # iterate over each distribution
        temp = df.loc[df.iloc[:,-1] == name].iloc[:, :-1].to_numpy()
        hh = ax.plot(x, temp.T, c=color, alpha=0.4, label=name)
        handles.append(hh[0] if isinstance(hh, list) else hh)
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.4, 1))
            
            

##########################################
#      Empirical density estimation      #
##########################################

def get_edf(df, x):
    # df: a dataframe containing samples from different distribution
    # x: array to calculate empirical cdf for it's values
    cum_p = list()  # empty list to store cumulative probability
    for i in range(len(df)):
        ecdf = ECDF(df.iloc[i,:-1])
        cum_p.append(ecdf(x)) # append empirical CDF of values in x

    edf_df = pd.DataFrame(cum_p)
    edf_df['label'] = df.iloc[:,-1].tolist()
    return edf_df 


def get_edf_plot(df, x):
    names = df.iloc[:,-1].unique()
    fig, ax = plt.subplots()
    colors = get_default_plt_colors()
    handles = []
    
    for name, color in zip(names, colors):  # iterate over each distribution
        temp = df.loc[df.iloc[:,-1] == name].iloc[:, :-1].to_numpy()
        hh = ax.plot(x, temp.T, c=color, alpha=0.4, label=name)
        handles.append(hh[0] if isinstance(hh, list) else hh)
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.4, 1))
    

##########################################
#   Empirical characteristics Function   #
##########################################
def get_ecf(df, t):
    # samples_dict: a dictinary containing samples from different distribution including preselected parameters
    # t : array of frequencies to calculate empirical characteristic function for it's values
    ecf_list = list()
    for i in range(len(df)):
        X = df.iloc[i,:-1]
        ecf = np.mean(np.exp(1j * np.outer(X, t).astype(float)), axis=0)
        ecf_r = np.real(ecf)
        ecf_i = np.imag(ecf)
        ecf_list.append(np.concatenate([ecf_r,ecf_i]))

    ecf_df = pd.DataFrame(ecf_list)
    ecf_df['label'] = df.iloc[:,-1].tolist()

    return ecf_df


def get_ecf_plot(df, t):
    names = df.iloc[:,-1].unique()
    fig, ax = plt.subplots()
    colors = get_default_plt_colors()
    handles = []
    for name, color in zip(names, colors):  # iterate over each distribution
        r_part = df.loc[df.iloc[:,-1] == name].iloc[:,0:len(t)]
        i_part = df.loc[df.iloc[:,-1] == name].iloc[:,len(t):-1]
        hh = ax.plot(r_part.T, i_part.T, c=color, alpha=0.4, label=name)
        handles.append(hh[0] if isinstance(hh, list) else hh)
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.xlabel('real part')
    plt.ylabel('imaginary part')