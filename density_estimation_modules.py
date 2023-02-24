import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF

##################################
#        methods of moments      #
##################################

def get_moments_df(distributions_dict, nr_moments, nr_sample, sample_size):
    # distribution dict: a dictinary containing different distribution including preselected parameters
    # nr_moments: desired number of moments to be calculated
    # nr_sample: number of samples
    # sample size: size of each sample

    m1 = list()
    x = np.zeros((nr_moments-1,nr_sample))
    df = pd.DataFrame()

    for i, (name, distr) in enumerate(distributions_dict.items()):
        samples = distr.rvs(size=(nr_sample, sample_size), random_state=10)
        m1.extend(np.mean(samples, axis=1)) # first moment

        for j in range(2,nr_moments+1): #calculate from 2nd moment
            x[j-2,:] = stats.moment(samples, j, axis=1)

        df_per_dist = pd.DataFrame(np.transpose(x), columns=['m'+str(i) for i in range(2,j+1)])
        df_per_dist['dist'] = name
        df = pd.concat([df,df_per_dist], ignore_index=True)

    m1_df = pd.DataFrame(m1, columns=['m1'])
    final_df = pd.concat([m1_df,df], axis=1)

    return final_df 


def get_histogram_of_moments(df):
    for i in range(len(df.columns)-1):
        fig, ax = plt.subplots()
        ax.hist(df.iloc[:,i], density=True, histtype='stepfilled', bins='auto')
        plt.title('moment ' + str(i))
        

##################################
#    Kernel density estimation   #
##################################

def get_kde(distributions_dict, nr_sample, sample_size, x):
    df = pd.DataFrame()
    for i, (name, distr) in enumerate(distributions_dict.items()):
        y_estimates = list()
        samples = distr.rvs(size=(nr_sample, sample_size), random_state=10)

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

            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)

            x = x.reshape((len(x),1))
            log_density = kde.score_samples(x)
            y_estimates.append(np.exp(log_density))

        df_per_dist = pd.DataFrame(y_estimates)  
        df_per_dist['dist'] = name

        df = pd.concat([df,df_per_dist], ignore_index=True)

    return df 

#this works only for bounded dists
def get_kde_plot(distributions_dict, kde_df, nr_sample, sample_size, x):
    for name,distr in (distributions_dict.items()):
        fig, ax = plt.subplots()
        temp = kde_df.loc[kde_df['dist']==name]
        for i in range(nr_sample):
            y = temp.iloc[i]
            dist_name = y['dist']
            y = y[:-1]
            ax.plot(x, y, c='#1f77b4', alpha=0.4)
            ax.set_title(dist_name)
            #plt.ylim(0,2.75)
            
            

##################################
#  Empirical density estimation  #
##################################

def get_edf(distributions_dict, nr_sample, sample_size, x):
    # distribution dict: a dictinary containing different distribution including preselected parameters
    # nr_sample: number of samples
    # sample size: size of each sample
    # x: array to calculate empirical cdf for it's values
    
    df = pd.DataFrame() # empty dataframe to store empirical CDF 
    
    for i, (name, distr) in enumerate(distributions_dict.items()): # iterate over each distribution
        cum_p = list() # empty list to store cumulative probability
        samples = distr.rvs(size=(nr_sample, sample_size), random_state=10) # get sample for each distribution
        # iterate over each sample
        for j in range(nr_sample):
            ecdf = ECDF(samples[j])
            cum_p.append(ecdf(x)) # append empirical CDF of values in x
        
        df_per_dist = pd.DataFrame(cum_p)
        df_per_dist['dist'] = name

        df = pd.concat([df, df_per_dist], ignore_index = True)
    return df 


def get_edf_plot(distributions_dict, edf_df, nr_sample, sample_size, x):
    for name,distr in (distributions_dict.items()):# iterate over each distribution
        fig, ax = plt.subplots()
        temp = edf_df.loc[edf_df['dist']==name]

        for i in range(nr_sample): # iterate over each sample
            y = temp.iloc[i]
            dist_name = y[-1:][0] 
            y = y[:-1] # excluding  the last element which is the distribution name

            ax.plot(x, y, c='#1f77b4', alpha=0.4)  
            ax.set_title(dist_name)