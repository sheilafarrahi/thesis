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
def get_moments(data, nr_moments):
    # samples_dict: a dictionary containing samples of different distribution including preselected parameters
    # nr_moments: desired number of moments to be calculated
    df = pd.DataFrame()
    x = np.zeros((len(data), nr_moments - 1))
    
    m1 = np.mean(data.iloc[:,:-1], axis=1)
    m1_df = pd.DataFrame(m1, columns=['m1']) 
    
    for i in range(2,nr_moments+1): #calculate from 2nd moment
        x[:,i-2] = stats.moment(data.iloc[:,:-1], i, axis=1)

    x_df = pd.DataFrame(x, columns=['m'+str(j) for j in range(2,i+1)])
    x_df['label'] = data.iloc[:,-1]
    final_df = pd.concat([m1_df,x_df], axis=1)
    
    return final_df


def get_histogram_of_moments(df):
    labels = df.iloc[:,-1].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf', '#393b79', '#637939', '#8c6d31', '#d6616b', '#7b4173', '#ce6dbd', 
              '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#8c8c8c', '#9c9ede', '#cedb9c', '#e7cb94', 
              '#e7969c', '#b5cf6b', '#a55194', '#de9ed6', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', 
              '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#74c476', '#9e9ac8', '#e41a1c', '#377eb8']

    for moment_name in df.columns[:-1]:
        i = 0
        fig, ax = plt.subplots()
        for label in labels:
            moments = df.loc[df.iloc[:,-1] == label, moment_name]
            ax.hist(moments, density=True, histtype='stepfilled', color = colors[i], bins='auto', alpha=0.75, label=label)
            i=i+1
        plt.title('moment: ' + moment_name)
        ax.legend(loc='center right', bbox_to_anchor=(1.75, 0.5), ncol=3)
        
##########################################
#        Kernel density estimation       #
##########################################
def get_kde(data, x):
    # data: graphwave data
    # x: array to do kde
    df = pd.DataFrame()
    y_estimates = list()

    for i in range(len(data)):
        X = data.iloc[i,:-1]
        kde = stats.gaussian_kde(list(X))
        values = x
        y_estimates.append(kde(values))

    df = pd.DataFrame(y_estimates)  
    df['label'] = data.iloc[:,-1]

    return df 

   
def get_kde_plot(df, x):
    names = df.iloc[:,-1].unique()
    fig, ax = plt.subplots()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf', '#393b79', '#637939', '#8c6d31', '#d6616b', '#7b4173', '#ce6dbd', 
              '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#8c8c8c', '#9c9ede', '#cedb9c', '#e7cb94', 
              '#e7969c', '#b5cf6b', '#a55194', '#de9ed6', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', 
              '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#74c476', '#9e9ac8', '#e41a1c', '#377eb8']
    handles = []
    for name, color in zip(names, colors):
        temp = df.loc[df['label'] == name].iloc[:, :-1].to_numpy()
        hh = ax.plot(x, temp.T, c=color, alpha=0.4, label=name)
        handles.append(hh[0] if isinstance(hh, list) else hh)
    ax.legend(handles=handles, loc='center right', bbox_to_anchor=(1.75, 0.5), ncol=3)

##########################################
#      Empirical density estimation      #
##########################################
def get_edf(data, x):
    # data: graphwave data
    # x: array to calculate empirical cdf for it's values
    cum_p = list()  # empty list to store cumulative probability
    for i in range(len(data)):
        ecdf = ECDF(data.iloc[i,:-1])
        cum_p.append(ecdf(x)) # append empirical CDF of values in x

    df = pd.DataFrame(cum_p)
    df['label'] = data.iloc[:,-1]
    return df 


def get_edf_plot(df, x):
    names = df.iloc[:,-1].unique()
    fig, ax = plt.subplots()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf', '#393b79', '#637939', '#8c6d31', '#d6616b', '#7b4173', '#ce6dbd', 
              '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#8c8c8c', '#9c9ede', '#cedb9c', '#e7cb94', 
              '#e7969c', '#b5cf6b', '#a55194', '#de9ed6', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', 
              '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#74c476', '#9e9ac8', '#e41a1c', '#377eb8']
    handles = []
    for name, color in zip(names, colors):
        temp = df.loc[df['label'] == name].iloc[:, :-1].to_numpy()
        hh = ax.plot(x, temp.T, c=color, alpha=0.4, label=name)
        handles.append(hh[0] if isinstance(hh, list) else hh)
    ax.legend(handles=handles, loc='center right', bbox_to_anchor=(1.75, 0.5), ncol=3)
       
##########################################
#  Empirical characteristics estimation  #
##########################################
def get_ecf(data, t):
    # data: graphwave data
    # t : array of frequencies to calculate empirical characteristic function for it's values
    ecf_list = list()
    for i in range(len(data)):
        X = data.iloc[i,:-1]
        ecf = np.mean(np.exp(1j * np.outer(X, t).astype(float)), axis=0)
        ecf_r = np.real(ecf)
        ecf_i = np.imag(ecf)
        ecf_list.append(np.concatenate([ecf_r,ecf_i]))

    df = pd.DataFrame(ecf_list)
    df['label'] = data.iloc[:,-1]

    return df


def get_ecf_plot(df, t):
    names = df.iloc[:,-1].unique()
    fig, ax = plt.subplots()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf', '#393b79', '#637939', '#8c6d31', '#d6616b', '#7b4173', '#ce6dbd', 
              '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#8c8c8c', '#9c9ede', '#cedb9c', '#e7cb94', 
              '#e7969c', '#b5cf6b', '#a55194', '#de9ed6', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', 
              '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#74c476', '#9e9ac8', '#e41a1c', '#377eb8']
    handles = []
    for name, color in zip(names, colors):  # iterate over each distribution
        r_part = df.loc[df['label'] == name].iloc[:,0:len(t)]
        i_part = df.loc[df['label'] == name].iloc[:,len(t):-1]
        hh = ax.plot(r_part.T, i_part.T, c=color, alpha=0.4, label=name)
        handles.append(hh[0] if isinstance(hh, list) else hh)
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.7, 1), ncol=3)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    
    
    
    