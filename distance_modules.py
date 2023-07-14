from scipy.stats import wasserstein_distance
from scipy.stats import spearmanr
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import distribution_modules as dm
import density_estimation_modules as dem
import classification_modules as cm
import importlib

importlib.reload(dm)
importlib.reload(dem)
importlib.reload(cm)


def w_distance(df):
    labels = df.iloc[:,-1].unique()
    nr_labels = len(labels)
    d_matrix = np.zeros((nr_labels,nr_labels))
    std_matrix = np.zeros((nr_labels,nr_labels))
    for n in range(nr_labels):
        for m in range(nr_labels):
            data_part1 = df.loc[df.iloc[:,-1]==labels[n]]
            data_part2 = df.loc[df.iloc[:,-1]==labels[m]]
            distance = list()
            for i in range(0,len(data_part1)):
                for j in range(0,len(data_part2)):
                    distance.append(wasserstein_distance(data_part1.iloc[i,:-1],data_part2.iloc[j,:-1]))
            d_matrix[n][m] = np.mean(distance)
            std_matrix[n][m] = np.std(distance)
    return d_matrix, std_matrix

def w_distance_flex(df_flex):
    labels = df['label'].unique()
    nr_labels = len(labels)
    d_matrix = np.zeros((nr_labels,nr_labels))
    std_matrix = np.zeros((nr_labels,nr_labels))
    for n in range(nr_labels):
        for m in range(nr_labels):
            data_part1 = df.loc[df['label']==labels[n]]
            data_part2 = df.loc[df['label']==labels[m]]
            distance = list()
            for i in range(0,len(data_part1)):
                for j in range(0,len(data_part2)):
                    distance.append(wasserstein_distance(data_part1.iloc[i,0],data_part2.iloc[j,0]))
            d_matrix[n][m] = np.mean(distance)
            std_matrix[n][m] = np.std(distance)
    return d_matrix, std_matrix

def d_distance(df):
    labels = df.iloc[:,-1].unique()
    nr_labels = len(labels)
    d_matrix = np.zeros((nr_labels,nr_labels))
    std_matrix = np.zeros((nr_labels,nr_labels))
    for n in range(nr_labels):
        #for m in range(n,nr_labels):
        for m in range(nr_labels):
            data_part1 = df.loc[df.iloc[:,-1]==labels[n]]
            data_part2 = df.loc[df.iloc[:,-1]==labels[m]]
            distance = list()
            for i in range(0,len(data_part1)):
                for j in range(0,len(data_part2)):
                    distance.append(np.linalg.norm(np.array(data_part1.iloc[i,:-1])-np.array(data_part2.iloc[j,:-1])))
            d_matrix[n][m] = np.mean(distance)
            std_matrix[n][m] = np.std(distance)              
    return d_matrix, std_matrix

def plot_matrix(matrix, labels):
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot = True, cmap='Blues_r') 
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()
    
def corr_coef(matrix1,matrix2):
    vector1 = matrix1.flatten()
    vector2 = matrix2.flatten()
    correlation_coefficient, p_value = spearmanr(vector1, vector2)
    return correlation_coefficient, p_value

#####################################################
#                 Moments Approach                  #
#####################################################
def cv_samplesize_moments(sample_size_list, nr_moments_list, dists, nr_sample_sets, transform=False, standardize=False):
    # nr_sample_sets: number of sample sets to be generated from each distribution(dists)
    # sample_size_list: number of samples in each sample set
    # nr_moments_list: number of moments to be used in extracting statistical summary of the sample sets
    # generate samples using the given nr_sample_sets and sample_size_list
    # transform: if True then log transform the samples
    # standardize: if True the standardize each sample set
    result = list()
    for i in tqdm(sample_size_list, desc='Completed'):
        if standardize == True:
            samples = dm.get_st_samples(dists, nr_sample_sets, i, transform = transform)
        else:
            samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
        original_d, original_std = w_distance(samples)
        for j in nr_moments_list:
            moments_df = dem.get_moments(samples, j)
            moments_d, moments_std = d_distance(moments_df)
            correlation_coefficient, p_value = corr_coef(original_d,moments_d)
            result.append(dict(zip(['corr_coef','p_value','nr_moments','sample_size'],[correlation_coefficient, p_value, j, i])))
    result_df = pd.DataFrame(result)    
    return result_df

def cv_samplesize_moments_mm(sample_size_list, nr_moments_list, nr_sample_sets, nr_mm_dist, nr_modes):
    result = list()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_sample_sets, nr_modes, i)
        original_d, original_std = w_distance(samples)
        for j in nr_moments_list:
            moments_df = dem.get_moments(samples, j)
            moments_d, moments_std = d_distance(moments_df)
            correlation_coefficient, p_value = corr_coef(original_d,moments_d)
            result.append(dict(zip(['corr_coef','p_value','nr_moments','sample_size'],[correlation_coefficient, p_value, j, i])))
    result_df = pd.DataFrame(result)    
    return result_df

def plot_cv_moments(df):
    ax = sns.lineplot(data = df, x='nr_moments',y='corr_coef', hue='sample_size', ci = 'sd', legend='full', palette='muted')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
    #plt.title('Moments Approach')
    plt.xlabel('Number of moments')
    plt.ylabel('Correlation Coefficient')
    plt.grid(color='#DDDDDD')
    plt.ylim(-1,1.1)
    plt.show()
    
#####################################################
#                   KDE and EDF                     #
#####################################################
def cv_numsteps_samplesize(sample_size_list, num_steps_list, dists, nr_sample_sets, method, transform=False, standardize=False):
    result = list()
    for i in tqdm(sample_size_list, desc ='Completed'):
        if standardize == True:
            samples = dm.get_st_samples(dists, nr_sample_sets, i, transform = transform)   
        else:
            samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
        original_d, original_std = w_distance(samples)
        
        for j in num_steps_list:
            if transform == False:
                if standardize == True:
                    min_ = np.percentile(samples.iloc[:,:-1],2.5)
                    max_ = np.percentile(samples.iloc[:,:-1],97.5)
                    x = np.linspace(min_,max_,j)
                else:
                    x = np.linspace(0,1,j)
            elif transform == True :
                perc_95 = np.percentile(samples.iloc[:,:-1],95)
                x = np.linspace(0,perc_95,j)
            if method == 'kde':
                df = dem.get_kde(samples, x)
            elif method == 'edf':
                y = np.linspace(0.01,1,j)
                df = dem.get_edf_v2(samples, y)
            df_d, df_std = d_distance(df)
            correlation_coefficient, p_value = corr_coef(original_d,df_d)
            result.append(dict(zip(['corr_coef','p_value','num_steps','sample_size'],[correlation_coefficient, p_value, j, i])))
    
    result_df = pd.DataFrame(result)    
    return result_df  

def cv_numsteps_samplesize_mm(sample_size_list, num_steps_list, nr_sample_sets, nr_mm_dist, nr_modes, method):
    result = list()
    for i in tqdm(sample_size_list, desc ='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_sample_sets, nr_modes, i)
        original_d, original_std = w_distance(samples)
        for j in num_steps_list:
            if method == 'kde':
                min_ = np.percentile(samples.iloc[:,:-1],2.5)
                max_ = np.percentile(samples.iloc[:,:-1],97.5)
                x = np.linspace(min_,max_,j)
                df = dem.get_kde(samples, x)
            elif method == 'edf':
                y = np.linspace(0.01,1,j)
                df = dem.get_edf_v2(samples, y)
            df_d, df_std = d_distance(df)
            correlation_coefficient, p_value = corr_coef(original_d,df_d)
            result.append(dict(zip(['corr_coef','p_value','num_steps','sample_size'],[correlation_coefficient, p_value, j, i])))
    result_df = pd.DataFrame(result)    
    return result_df


def plot_cv_numsteps_samplesize(df):
    ax = sns.lineplot(data = df, x='num_steps',y='corr_coef', hue='sample_size', ci = 'sd', legend='full', palette='muted')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
    plt.xlabel('Number of steps')
    plt.ylabel('Correlation Coefficient')
    plt.grid(color='#DDDDDD')
    plt.ylim(-1,1.1)
    plt.show()
    
#####################################################
#                        ECF                        #
#####################################################            
def cv_ecf(sample_size_list, max_t_list, num_steps_list, dists, nr_sample_sets, transform = False, standardize=False):
    result = list()
    for i in tqdm(sample_size_list, desc ='Completed'):
        if standardize == True:
            samples = dm.get_st_samples(dists, nr_sample_sets, i, transform = transform)
            st_samples = dm.standardize_df(samples)
            original_d, original_std = w_distance(st_samples)
        else:
            samples = dm.get_samples(dists, nr_sample_sets, i, transform = transform)
            original_d, original_std = w_distance(samples)
            
        for j in num_steps_list:
            for k in max_t_list:
                t = np.linspace(k/j, k, j)
                df = dem.get_ecf(samples, t)
                df_d, df_std = d_distance(df)
                correlation_coefficient, p_value = corr_coef(original_d,df_d)
                result.append(dict(zip(['corr_coef','p_value','max_t','num_steps','sample_size'],
                                       [correlation_coefficient, p_value, k, j, i])))
            
    result_df = pd.DataFrame(result)           
    return result_df

def cv_ecf_mm(sample_size_list, max_t_list, num_steps_list, nr_sample_sets, nr_mm_dist, nr_modes):
    result = list()
    for i in tqdm(sample_size_list, desc ='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_sample_sets, nr_modes, i)
        original_d, original_std = w_distance(samples)
        for j in num_steps_list:
            for k in max_t_list:
                t = np.linspace(k/j, k, j)
                df = dem.get_ecf(samples, t)
                df_d, df_std = d_distance(df)
                correlation_coefficient, p_value = corr_coef(original_d,df_d)
                result.append(dict(zip(['corr_coef','p_value','max_t','num_steps','sample_size'],
                                       [correlation_coefficient, p_value, k, j, i])))
            
    result_df = pd.DataFrame(result)           
    return result_df

def plot_cv_ecf(clf_result):
    for i in (clf_result['sample_size'].unique()):
        fig, ax = plt.subplots()
        ax = sns.lineplot(data = clf_result.loc[clf_result['sample_size']==i], 
                          x='num_steps',y='corr_coef', hue='max_t', ci = 'sd', legend='full', palette='muted', alpha=0.7)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Max t')
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.title('ECF, Sample Size =%i' %i)
        plt.ylabel('Correlation Coefficient')
        plt.xlabel('Number of Steps')
        plt.grid(color='#DDDDDD')
        plt.ylim(-1,1.1)
        plt.show()