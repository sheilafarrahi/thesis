import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
def get_default_plt_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


##########################################
#            methods of moments          #
##########################################

def get_moments(df, nr_moments, label= True):
    # df: a dictionary containing samples of different distribution including preselected parameters
    # nr_moments: desired number of moments to be calculated
    if label == True:
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
    else:
        X = df
    m1 = np.mean(X, axis=1)
    #m1_df = pd.DataFrame(m1, columns=['m1'])
    m1_df = pd.DataFrame(m1)
    m1_df = m1_df.reset_index(drop=True)
    moments = np.zeros((len(X), nr_moments - 1)) # array to store moments after mean
    for i in range(2,nr_moments+1): #calculate from 2nd moment
        moments[:,i-2] = stats.moment(X, i, axis=1)
    #moments_df = pd.DataFrame(moments, columns=['m'+str(j) for j in range(2,i+1)])
    moments_df = pd.DataFrame(moments)
    df = pd.concat([m1_df,moments_df], axis=1)
    df.columns = ['m'+str(j) for j in range(1,nr_moments+1)]
    if label == True:
        df['label'] = y.values.tolist()
    return df


def get_moments_no_label(df, nr_moments):
    # df: a dictionary containing samples of different distribution including preselected parameters
    # nr_moments: desired number of moments to be calculated
    m1 = np.mean(df, axis=1)
    m1_df = pd.DataFrame(m1, columns=['m1'])
    m1_df = m1_df.reset_index(drop=True)
    moments = np.zeros((len(df), nr_moments - 1)) # array to store moments after mean

    for i in range(2,nr_moments+1): #calculate from 2nd moment
        moments[:,i-2] = stats.moment(df, i, axis=1)

    moments_df = pd.DataFrame(moments, columns=['m'+str(j) for j in range(2,i+1)])
    df = pd.concat([m1_df,moments_df], axis=1)
    return df

def get_moments_partial(df, nr_moments):
    moments = list()
    m1 = np.mean(df)
    moments.append(m1)
    for i in range(2,nr_moments+1): #calculate from 2nd moment
        moments.append(stats.moment(df, i))
    return moments

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
        if type(x)==list:
            values = x[i]
        else:
            values = x
        y_estimates.append(kde(values))

    kde_df = pd.DataFrame(y_estimates)  
    kde_df['label'] = df.iloc[:,-1].tolist()

    return kde_df 
    
def get_kde_partial(df, x):
    kde = stats.gaussian_kde(df)
    return kde(x)

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

def get_edf_partial(df,y):
    ecdf = ECDF(df)
    return ecdf(y)

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

def get_edf_v2(df, y):
    # df: a dataframe containing samples from different distribution
    # y:
    x = list() 
    for i in range(len(df)):
        ecdf = ECDF(df.iloc[i,:-1])
        #inverse_ecdf = interp1d(ecdf.y, ecdf.x)
        inverse_ecdf= interp1d(ecdf.y[1:], ecdf.x[1:], bounds_error=False, fill_value=0)
        x.append(inverse_ecdf(y))

    edf_df = pd.DataFrame(x)
    edf_df['label'] = df.iloc[:,-1].tolist()
    return edf_df 



def get_edf_plot_v2(df, y):
    names = df.iloc[:,-1].unique()
    fig, ax = plt.subplots()
    colors = get_default_plt_colors()
    handles = []
    
    for name, color in zip(names, colors):  # iterate over each distribution
        temp = df.loc[df.iloc[:,-1] == name].iloc[:, :-1].to_numpy()
        hh = ax.plot(temp.T, y, c=color, alpha=0.4, label=name)
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

def get_ecf_partial(df, max_t, steps):
    t = np.linspace(0.001, int(max_t), int(steps))
    ecf = np.mean(np.exp(1j * np.outer(df, t).astype(float)), axis=0)
    ecf_r = np.real(ecf)
    ecf_i = np.imag(ecf)
    ecf_parts =np.concatenate([ecf_r,ecf_i])
    return ecf_parts

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