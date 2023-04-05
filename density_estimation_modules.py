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

def get_moments_df(data, nr_moments):
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
    for moment_name in df.columns[:-1]:
        fig, ax = plt.subplots()
        for label in labels:
            moments = df.loc[df.iloc[:,-1] == label, moment_name]
            ax.hist(moments, density=True, histtype='stepfilled', bins='auto', alpha=0.75, label=label)
        plt.title('moment: ' + moment_name)
        ax.legend(loc='center right', bbox_to_anchor=(1.75, 0.5), ncol=3)
        

##########################################
#        Kernel density estimation       #
##########################################

def get_kde(samples_dict, x):
    # samples_dict: a dictinary containing samples from different distribution including preselected parameters
    # x: array to do kde
    df = pd.DataFrame()
    for i, (name, samples) in enumerate(samples_dict.items()):
        nr_sample = samples.shape[0]
        y_estimates = list()

        for j in range(nr_sample):
            X = samples[j,:]
            kde = stats.gaussian_kde(X)
            values = x
            y_estimates.append(kde(values))

        df_per_dist = pd.DataFrame(y_estimates)  
        df_per_dist['dist'] = name
        df = pd.concat([df,df_per_dist], ignore_index=True)

    return df 

   
def get_kde_plot(df, x):
    names = df.iloc[:,-1].unique()
    fig, ax = plt.subplots()
    colors = get_default_plt_colors()
    handles = []
    for name, color in zip(names, colors):  # iterate over each distribution
        temp = df.loc[df['label'] == name].iloc[:, :-1].to_numpy()
        hh = ax.plot(x, temp.T, c=color, alpha=0.4, label=name)
        handles.append(hh[0] if isinstance(hh, list) else hh)
    ax.legend(handles=handles)
            
            

##########################################
#      Empirical density estimation      #
##########################################

def get_edf(samples_dict, x):
    # samples_dict: a dictinary containing samples from different distribution including preselected parameters
    # x: array to calculate empirical cdf for it's values
    df = pd.DataFrame() # empty dataframe to store empirical CDF 
    
    for i, (name, samples) in enumerate(samples_dict.items()):  # iterate over each distribution
        nr_sample = samples.shape[0]
        cum_p = list() # empty list to store cumulative probability
        
        for j in range(nr_sample): # iterate over each sample
            ecdf = ECDF(samples[j])
            cum_p.append(ecdf(x)) # append empirical CDF of values in x
        
        df_per_dist = pd.DataFrame(cum_p)
        df_per_dist['dist'] = name

        df = pd.concat([df, df_per_dist], ignore_index = True)
    return df 


def get_edf_plot(edf_df, x):
    names = edf_df['dist'].unique()
    fig, ax = plt.subplots()
    colors = get_default_plt_colors()
    handles = []
    
    for name, color in zip(names, colors):  # iterate over each distribution
        temp = edf_df.loc[edf_df['dist'] == name].iloc[:, :-1].to_numpy()
        hh = ax.plot(x, temp.T, c=color, alpha=0.4, label=name)
        handles.append(hh[0] if isinstance(hh, list) else hh)
    ax.legend(handles=handles)
    
    
def get_edf_plot_2(df, x):
    names = df['dist'].unique()
    fig, ax = plt.subplots()
    colors = get_default_plt_colors()
    handles = []
    for name, color in zip(names, colors):  # iterate over each distribution
        temp = df.loc[df['dist'] == name].iloc[:, :-1].to_numpy()
        hh = ax.plot(x, temp.T, c=color, alpha=0.4, label=name)
        handles.append(hh[0] if isinstance(hh, list) else hh)
    ax.legend(handles=handles)
    

##########################################
#  Empirical characteristics estimation  #
##########################################

def get_ecf(sample_dict, t):
    # samples_dict: a dictinary containing samples from different distribution including preselected parameters
    # t : array of frequencies to calculate empirical characteristic function for it's values
    df = pd.DataFrame()
    for i, (name, samples) in enumerate(sample_dict.items()):
        nr_sample =samples.shape[0]
        ecf_list = list()

        for j in range(nr_sample):
            data = samples[j,:]
            ecf = np.mean(np.exp(1j * np.outer(data, t)), axis=0)
            ecf_r = np.real(ecf)
            ecf_i = np.imag(ecf)
            ecf_list.append(np.concatenate([ecf_r,ecf_i]))

        df_per_dist = pd.DataFrame(ecf_list)
        df_per_dist['dist'] = name
        df = pd.concat([df, df_per_dist], ignore_index = True)
        
    return df


def get_ecf_plot(df, t):
    names = df['dist'].unique()
    fig, ax = plt.subplots()
    colors = get_default_plt_colors()
    handles = []
    for name, color in zip(names, colors):  # iterate over each distribution
        r_part = df.loc[df['dist'] == name].iloc[:,0:len(t)]
        i_part = df.loc[df['dist'] == name].iloc[:,len(t):-1]
        hh = ax.plot(r_part.T, i_part.T, c=color, alpha=0.4, label=name)
        handles.append(hh[0] if isinstance(hh, list) else hh)
    ax.legend(handles=handles)