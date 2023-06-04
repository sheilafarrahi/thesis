import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random

def get_bounded_distribution():
    bounded_distributions = {
        "arcsine" : stats.arcsine(),
        "beta_1_2" : stats.beta(a=1, b=2),
        "powerlaw_0.3" : stats.powerlaw(a=0.3),
        "trapezoid_0.3_0.8" : stats.trapezoid(c=0.3, d=0.8),
        "traing_0.3" : stats.triang(c=0.3),
        "uniform" : stats.uniform()
    }
    return bounded_distributions

def get_heavytail_distribution():
    heavytail_distributions = {
        "cauchy" : stats.halfcauchy(),
        "lognorm_1.5" : stats.lognorm(s=1.5),
        "pareto_1.5" : stats.pareto(b=1.5),
        "weibull_min_0.4" : stats.weibull_min(c=0.4)
    }
    return heavytail_distributions

        
def get_samples(distributions_dict, nr_sample, sample_size, random_state=10, transform=False):
    # distributions_dict: a dictinary containing different distribution including preselected parameters
    # nr_sample: number of samples
    # sample size: size of each sample
    samples_dict = dict()
    for i, (name, distr) in enumerate(distributions_dict.items()):
        samples = distr.rvs(size=(nr_sample, sample_size), random_state=random_state)
        samples_dict[name] = samples
        
    if transform == True:
        transformed_sampels = dict()
        for name, samples in samples_dict.items():
            transformed_sampels[name] = np.log1p(samples)
        samples_dict = transformed_sampels

    df = pd.DataFrame()
    for i, (name, samples) in enumerate(samples_dict.items()):
        df_sample = pd.DataFrame(samples)
        df_sample['label'] = name
        df = df.append(df_sample, ignore_index=True)
    return df

def standardize_df(df):
    # standardize the given df by deducting row mean from each row and divide it by the std
    st_df = pd.DataFrame().reindex_like(df)
    row_mean = np.mean(df, axis=1)
    row_std = np.std(df, axis=1) 
    for i in range(len(df)):
        for j in range(len(df.columns)-1):
            st_df.iloc[i][j] = (df.iloc[i][j]-row_mean[i])/row_std[i]

    st_df.iloc[:,-1] = df.iloc[:,-1]    
    return st_df

def get_st_samples(distributions_dict, nr_sample, sample_size, random_state=10, transform=False):
    df = get_samples(distributions_dict, nr_sample, sample_size)
    st_df = standardize_df(df)
    return st_df

# dont check this
def plot_histograms_of_samples(df):
    dists = df['label'].unique()
    for dist in dists:
        fig, ax = plt.subplots()
        handles = []
        df_dist = df.loc[df['label']==dist].iloc[:,:-1]
        for i in range(len(df_dist)):
            sample = df_dist.iloc[i,:]
            ax.hist(sample, density=True, histtype='stepfilled', bins='auto', alpha=0.1)
            ax.set_title('Samples from %s' %dist)
            plt.ylim(0,3)
            plt.xlim(0,10)
            

def get_weights(n):
    # weights for each gaussian in the total samples taken in get_multimodal function
    random_list = [random.randint(1,20) for i in range(n)]
    weights=[]
    for i in range(n-1):
        weights.append(round(random_list[i]/sum(random_list),2))
    weights.append(round(1-sum(weights),2))
    return weights

def get_modes_vars(nr_modes, init_mode):
    # generate a vector with n (nr_modes) elements to be uses as variance in get_multimodal function
    modes = list()
    var = list()
    for i in range(nr_modes):
        modes.append(init_mode + i * (1+ random.random()))
        var.append(random.uniform(0, 1))
    return modes, var

def get_multimodal(nr_modes, nr_samples, sample_size):
    # The function generates n (sample_size) samples from m (nr_modes) gaussian distributions.
    # n is divided into m parts using a weight vector
    samples = list()
    weights = get_weights(nr_modes)
    modes, var = get_modes_vars(nr_modes, 1)

    for i in range(nr_samples):
        sample = list()
        for j in range(nr_modes):
            sample_size_ = int(weights[j] * sample_size)
            sample_ = stats.norm.rvs(size = sample_size_, loc = modes[j], scale = np.sqrt(var[j]))
            for k in range(len(sample_)): # switch negative values with mean to avoid negative values
                if sample_[k] < 0:
                    sample_[k] = modes[j]
            sample.extend(sample_)

        samples.append(sample)
    return samples, weights, modes, var

def get_multimodal_dists(nr_mm_dist, nr_sample, nr_modes, sample_size):
    samples = list()
    label_list = list()
    mean_list = list()
    var_list = list()
    
    for i in range(nr_mm_dist):
        label = 'Dist '+ str(i+1)
        samples_, weights, modes, var = get_multimodal(nr_modes, nr_sample, sample_size)
        samples.extend(samples_)
        mean_list.append(modes)
        var_list.append(var)
        df = pd.DataFrame(samples)
        for j in range(nr_sample):
            label_list.append(label)
    
    df['label']=label_list
    return df, mean_list, var_list