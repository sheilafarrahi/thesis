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
from functools import partial
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
    labels = df_flex['label'].unique()
    nr_labels = len(labels)
    d_matrix = np.zeros((nr_labels,nr_labels))
    std_matrix = np.zeros((nr_labels,nr_labels))
    for n in range(nr_labels):
        for m in range(nr_labels):
            data_part1 = df_flex.loc[df_flex['label']==labels[n]]
            data_part2 = df_flex.loc[df_flex['label']==labels[m]]
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
def cv_samplesize_moments(input_size_list, nr_features_list, dists, nr_multisets, transform=False, standardize=False, flex=False):
    result = list()
    for i in tqdm(input_size_list, desc='Completed'):
        if flex == True:
            samples = dm.get_samples_flex(dists, nr_multisets, i, 2)
            original_d, original_std = w_distance_flex(samples)
            for j in nr_features_list:
                partial_moments = partial(dem.get_moments_partial, nr_moments=j)
                moments_res = samples['sample_set'].apply(partial_moments)
                moments_df = pd.DataFrame(moments_res.tolist())
                moments_df['label'] = samples['label']
                moments_d, moments_std = d_distance(moments_df)
                correlation_coefficient, p_value = corr_coef(original_d, moments_d)
                result.append(dict(zip(['corr_coef','p_value','nr_features','input_size'],[correlation_coefficient, p_value, j, i])))
        else:
            if standardize == True:
                samples = dm.get_st_samples(dists, nr_multisets, i)
            else:
                samples = dm.get_samples(dists, nr_multisets, i)

            original_d, original_std = w_distance(samples)
            for j in nr_features_list:
                moments_df = dem.get_moments(samples, j)
                moments_d, moments_std = d_distance(moments_df)
                correlation_coefficient, p_value = corr_coef(original_d,moments_d)
                result.append(dict(zip(['corr_coef','p_value','nr_features','input_size'],[correlation_coefficient, p_value, j, i])))
            
    result_df = pd.DataFrame(result)    
    return result_df

def cv_samplesize_moments_mm(input_size_list, nr_features_list, nr_multisets, nr_mm_dist, nr_modes):
    result = list()
    for i in tqdm(input_size_list, desc='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_multisets, nr_modes, i)
        original_d, original_std = w_distance(samples)
        for j in nr_features_list:
            moments_df = dem.get_moments(samples, j)
            moments_d, moments_std = d_distance(moments_df)
            correlation_coefficient, p_value = corr_coef(original_d,moments_d)
            result.append(dict(zip(['corr_coef','p_value','nr_features','input_size'],[correlation_coefficient, p_value, j, i])))
    result_df = pd.DataFrame(result)    
    return result_df

def plot_cv_moments(df):
    input_size_list = df['input_size'].unique()
    fig, ax = plt.subplots()
    for i in range(len(input_size_list)):
        data = df.loc[df['input_size'] == input_size_list[i]]
        plt.plot(data['nr_features'], data['corr_coef'], label = input_size_list[i])
    ax.legend(loc='lower left', ncol=3, title='Input size')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Number of constructed features')
    plt.grid(color='#DDDDDD')
    plt.ylim(-1.1,1.1)
    plt.show()
    
#####################################################
#                   KDE and EDF                     #
#####################################################
def cv_numsteps_samplesize(input_size_list, num_features_list, dists, nr_multisets, method, transform=False, standardize=False, flex = False):
    result = list()
    for i in tqdm(input_size_list, desc ='Completed'):
        if flex == True:
            samples = dm.get_samples_flex(dists, nr_multisets, i, 2)
            original_d, original_std = w_distance_flex(samples)
            for j in num_features_list:
                if method == 'kde':
                    x = np.linspace(0,1,j)
                    partial_kde = partial(dem.get_kde_partial, x=x)
                    kde_res = samples['sample_set'].apply(partial_kde)
                    df = pd.DataFrame(kde_res.tolist())
                elif method == 'edf':
                    y = np.linspace(0.01,1,j)
                    partial_edf = partial(dem.get_edf_partial, y=y)
                    edf_res = samples['sample_set'].apply(partial_edf)
                    df = pd.DataFrame(edf_res.tolist())
                df['label'] = samples['label']
                df_d, df_std = d_distance(df)
                correlation_coefficient, p_value = corr_coef(original_d, df_d)
                result.append(dict(zip(['corr_coef','p_value','nr_features','input_size'],[correlation_coefficient, p_value, j, i])))
        else:        
            if standardize == True:
                samples = dm.get_st_samples(dists, nr_multisets, i)   
            else:
                samples = dm.get_samples(dists, nr_multisets, i, transform = transform)
            
            original_d, original_std = w_distance(samples)
            for j in num_features_list:
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
                correlation_coefficient, p_value = corr_coef(original_d, df_d)
                result.append(dict(zip(['corr_coef','p_value','nr_features','input_size'],[correlation_coefficient, p_value, j, i])))
    result_df = pd.DataFrame(result)    
    return result_df  

def cv_numsteps_samplesize_mm(input_size_list, num_features_list, nr_multisets, nr_mm_dist, nr_modes, method):
    result = list()
    for i in tqdm(input_size_list, desc ='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_multisets, nr_modes, i)
        original_d, original_std = w_distance(samples)
        for j in num_features_list:
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
            result.append(dict(zip(['corr_coef','p_value','nr_features','input_size'],[correlation_coefficient, p_value, j, i])))
    result_df = pd.DataFrame(result)    
    return result_df


def plot_cv_numsteps_samplesize(df):
    ax = sns.lineplot(data = df, x='nr_features',y='corr_coef', hue='input_size', ci = 'sd', legend='full', palette='muted')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
    plt.xlabel('Number of constructed features')
    plt.ylabel('Correlation Coefficient')
    plt.grid(color='#DDDDDD')
    plt.ylim(-1.1,1.1)
    plt.show()
    
#####################################################
#                        ECF                        #
#####################################################      
def cv_ecf(input_size_list, t, nr_feature_list, dists, nr_multisets, transform = False, standardize=False, flex=False):
    result = list()
    for i in tqdm(input_size_list, desc ='Completed'):
        if flex == True:
            samples = dm.get_samples_flex(dists, nr_multisets, i, 2)
            original_d, original_std = w_distance_flex(samples)
            max_t = t[0]
            for j in nr_feature_list:
                partial_ecf = partial(dem.get_ecf_partial, max_t=max_t, steps=j)
                ecf_res = samples['sample_set'].apply(partial_ecf)
                df = pd.DataFrame(ecf_res.tolist())
                df['label'] = samples['label']
                df_d, df_std = d_distance(df)
                correlation_coefficient, p_value = corr_coef(original_d, df_d)
                result.append(dict(zip(['corr_coef','p_value','nr_features','input_size'], 
                                       [correlation_coefficient, p_value, j, i])))   
        else:
            if standardize == True:
                samples = dm.get_st_samples(dists, nr_multisets, i)
                st_samples = dm.standardize_df(samples)
                original_d, original_std = w_distance(st_samples)
            else:
                samples = dm.get_samples(dists, nr_multisets, i, transform = transform)
                original_d, original_std = w_distance(samples)

            for j in nr_feature_list:
                t_interval = np.linspace(0.001, t, j)
                df = dem.get_ecf(samples, t_interval)
                df_d, df_std = d_distance(df)
                correlation_coefficient, p_value = corr_coef(original_d, df_d)
                result.append(dict(zip(['corr_coef','p_value','nr_features','input_size'], 
                                       [correlation_coefficient, p_value, j, i])))
    result_df = pd.DataFrame(result)           
    return result_df

def cv_ecf_mm(input_size_list, t, nr_feature_list, nr_multisets, nr_mm_dist, nr_modes):
    result = list()
    for i in tqdm(input_size_list, desc ='Completed'):
        samples = dm.get_multimodal_dists(nr_mm_dist, nr_multisets, nr_modes, i)
        original_d, original_std = w_distance(samples)
        for j in nr_feature_list:
            t_interval = np.linspace(0.001, t, j)
            df = dem.get_ecf(samples, t_interval)
            df_d, df_std = d_distance(df)
            correlation_coefficient, p_value = corr_coef(original_d, df_d)
            result.append(dict(zip(['corr_coef','p_value','nr_features','input_size'],[correlation_coefficient, p_value, j, i])))
    result_df = pd.DataFrame(result)           
    return result_df

def plot_cv_ecf(clf_result):
    input_size_list = clf_result['input_size'].unique()
    fig, ax = plt.subplots()
    for i in range(len(input_size_list)):
        data = clf_result.loc[clf_result['input_size'] == input_size_list[i]]
        plt.plot(data['nr_features'], data['corr_coef'], label = input_size_list[i])
        
    ax.legend(loc='lower left', ncol=3, title='Input size')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Number of constructed features')
    plt.grid(color='#DDDDDD')
    plt.ylim(-1.1,1.1)
    plt.show()
    
def plot_cv_ecf_p(clf_result):
    input_size_list = clf_result['input_size'].unique()
    fig, ax = plt.subplots()
    for i in range(len(input_size_list)):
        data = clf_result.loc[clf_result['input_size'] == input_size_list[i]]
        plt.plot(data['nr_features'], data['p_value'], label = input_size_list[i])
        
    ax.legend(loc='upper left', ncol=3, title='Input size')
    #ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Number of constructed features')
    plt.grid(color='#DDDDDD')
    #plt.ylim(-1.1,1.1)
    plt.show()