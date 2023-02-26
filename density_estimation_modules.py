import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF
def get_default_plt_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

##################################
#        methods of moments      #
##################################

def get_moments_df(samples_dict, nr_moments):
    # samples_dict: a dictionary containing samples of different distribution including preselected parameters
    # nr_moments: desired number of moments to be calculated

    m1 = list()
    df = pd.DataFrame()
    
    for i, (name, samples) in enumerate(samples_dict.items()):
        nr_sample = samples.shape[0]
        x = np.zeros((nr_moments - 1, nr_sample))
        m1.extend(np.mean(samples, axis=1))  # first moment

        for j in range(2,nr_moments+1): #calculate from 2nd moment
            x[j-2,:] = stats.moment(samples, j, axis=1)

        df_per_dist = pd.DataFrame(np.transpose(x), columns=['m'+str(i) for i in range(2,j+1)])
        df_per_dist['dist'] = name
        df = pd.concat([df,df_per_dist], ignore_index=True)

    m1_df = pd.DataFrame(m1, columns=['m1'])
    final_df = pd.concat([m1_df,df], axis=1)

    return final_df 


def get_histogram_of_moments(df):
    distrubtions = df['dist'].unique()
    for moment_name in df.columns[:-1]:
        fig, ax = plt.subplots()
        for distr_name in distrubtions:
            moments = df.loc[df['dist'] == distr_name, moment_name]
            ax.hist(moments, density=True, histtype='stepfilled', bins='auto', alpha=0.75, label=distr_name)
        plt.title('moment: ' + moment_name)
        ax.legend()
        

##################################
#    Kernel density estimation   #
##################################

def get_kde(samples_dict, x):
    # samples_dict: a dictinary containing samples from different distribution including preselected parameters
    # x: array to calculate empirical cdf for it's values
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


def get_kde_l(distributions_dict, nr_sample, sample_size, x, bandwidth):
    df = pd.DataFrame()
    for i, (name, distr) in enumerate(distributions_dict.items()):
        y_estimates = list()
        samples = distr.rvs(size=(nr_sample, sample_size), random_state=10)

        for j in range(nr_sample):
            X = samples[j,:]
            X = X.reshape((len(X),1))

            kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(X)

            x = x.reshape((len(x),1))
            log_density = kde.score_samples(x)
            y_estimates.append(np.exp(log_density))

        df_per_dist = pd.DataFrame(y_estimates)  
        df_per_dist['dist'] = name

        df = pd.concat([df,df_per_dist], ignore_index=True)

    return df 
            
# this works only for bounded dists
def get_kde_plot(kde_df, x):
    names = kde_df['dist'].unique()
    for name in names:
        fig, ax = plt.subplots()
        temp = kde_df.loc[kde_df['dist'] == name]
        for i in range(len(x)):
            y = temp.iloc[i]
            dist_name = y[-1:][0]
            y = y[:-1]
            ax.plot(x, y, c='#1f77b4', alpha=0.4)
            ax.set_title(dist_name)
            
            

##################################
#  Empirical density estimation  #
##################################

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