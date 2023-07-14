import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
#random.seed(10)

def get_bounded_distribution():
    bounded_distributions = {
        #"arcsine" : stats.arcsine(),
        #"beta_1_2" : stats.beta(a=1, b=2),
        #"powerlaw_0.3" : stats.powerlaw(a=0.3),
        #"trapezoid_0.3_0.8" : stats.trapezoid(c=0.3, d=0.8),
        #"traing_0.3" : stats.triang(c=0.3),
        #"uniform" : stats.uniform()
        "beta_1_2" : stats.beta(a=1, b=2),
        "beta_1_3" : stats.beta(a=1, b=3),
        "beta_1_4" : stats.beta(a=1, b=4),
        "beta_1_5" : stats.beta(a=1, b=5)    }
    return bounded_distributions

def get_heavytail_distribution():
    heavytail_distributions = {
        "cauchy" : stats.halfcauchy(),
        "lognorm_1" : stats.lognorm(s=1),
        "lognorm_2" : stats.lognorm(s=2),
        "pareto_0.5" : stats.pareto(b=1.5),
        #"pareto_1.5" : stats.pareto(b=1.5),
        #"weibull_min_0.4" : stats.weibull_min(c=0.4)
    }
    return heavytail_distributions

        
def get_samples(distributions_dict, nr_sample_sets, sample_size, random_state=10, transform=False):
    # distributions_dict: a dictinary containing different distribution including preselected parameters
    # nr_sample_sets: number of sample sets
    # sample size: size of each sample set
    samples_dict = dict()
    for i, (name, distr) in enumerate(distributions_dict.items()):
        samples = distr.rvs(size=(nr_sample_sets, sample_size), random_state=random_state)
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

def get_samples_flex(distributions_dict, nr_sample_sets, sample_size, min_nr_sample=1):
    df = pd.DataFrame()
    samples = list()
    labels = list()
    for i, (name, distr) in enumerate(distributions_dict.items()):
        for j in range(nr_sample_sets):
            sample = distr.rvs(random.randint(min_nr_sample,sample_size))
            samples.append(sample) 
            labels.append(name)
    df['sample_set'] = samples
    df['len'] = df['sample_set'].apply(len)
    df['label'] = labels
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

def min_max_scaled_df(df, lower_bound = 5, upper_bound = 95):
    min_ = np.percentile(df.iloc[:,:-1],lower_bound)
    max_ = np.percentile(df.iloc[:,:-1],upper_bound)
    scaler = MinMaxScaler(feature_range=(min_, max_))
    norm = scaler.fit_transform(df.iloc[:,:-1])
    norm_df = pd.DataFrame(norm)
    normalized_df = pd.concat([norm_df, df.iloc[:,-1]],axis = 1)
    return normalized_df

def get_st_samples(distributions_dict, nr_sample_sets, sample_size, random_state=10, transform=False):
    df = get_samples(distributions_dict, nr_sample_sets, sample_size, transform = transform)
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
            

def get_sample_size(n, sample_size):
    # Divides the sample_size into n parts to get number of samples per each gaussian distribution
    #random.seed(10)
    random_list = [random.randint(1,30) for i in range(n)]
    sample_size_ = []
    weights=[]
    for i in range(n-1):
        weight = random_list[i]/sum(random_list)
        sample_size_.append(int(weight * sample_size))
    # to make sure the sum will be eqaul to the original sample size in case sample_size is not divisible by n
    sample_size_.append(sample_size- sum(sample_size_)) 
    
    sample_size_sorted=np.sort(sample_size_)
    part_1 = sample_size_sorted[::2]
    part_2 = sample_size_sorted[1::2][::-1]
    final = np.append(part_1,part_2)
    return final

def get_modes(nr_modes, init_mode):
    # generate a vector with n (nr_modes) elements to be uses as mode in get_multimodal function
    modes = list()
    for i in range(nr_modes):
        #random.seed(10)
        modes.append(init_mode + i * random.uniform(2, 2.5))
    return modes

def get_vars(nr_modes):
    # generate a vector with n (nr_modes) elements to be uses as variance in get_multimodal function
    var = list()
    for i in range(nr_modes):
        #random.seed(i)
        var.append(random.uniform(0.02, 0.2))
    return var

def get_multimodal(nr_modes, nr_sample_sets, sample_size):
    # The function generates n (sample_size) samples from m (nr_modes) gaussian distributions.
    samples = list()
    sample_size_ = get_sample_size(nr_modes, sample_size)
    modes= get_modes(nr_modes, 1)
    var = get_vars(nr_modes)

    for i in range(nr_sample_sets):
        sample = list()
        for j in range(nr_modes):
            sample_ = stats.cauchy.rvs(size = sample_size_[j], loc = modes[j], scale = np.sqrt(var[j]))
            for k in range(len(sample_)):
                if abs(sample_[k]-modes[j]) > (np.sqrt(var[j]) * 6):
                    sample_[k] = modes[j]
            sample.extend(sample_)
        samples.append(sample)
    return samples, sample_size_, modes, var


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
    return df