from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    correlation_coefficient, p_value = pearsonr(vector1, vector2)
    return correlation_coefficient, p_value


def cv_samplesize_moments(sample_size_list, nr_moments_list, dists, nr_sample, transform = False):
    result = list()
    for i in tqdm(sample_size_list, desc='Completed'):
        samples = dm.get_samples(dists, nr_sample, i, transform = transform)
        original_d, original_std = w_distance(samples)
        for j in nr_moments_list:
            moments_df = dem.get_moments(samples, j)
            moments_d, moments_std = w_distance(moments_df)
            correlation_coefficient, p_value = d.corr_coef(original_d,moments_d)
            result.append(dict(zip(['corr_coef','p_value','nr_moments','sample_size'],[correlation_coefficient, p_value, j, i])))
       
    result_df = pd.DataFrame(result)    
    return result_df

def plot_cv_moments(df):
    ax = sns.lineplot(data = df, x='nr_moments',y='corr_coef', hue='sample_size', ci = 'sd', legend='full', palette='muted')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title='Sample Size')
    plt.title('Moments Approach')
    plt.xlabel('Number of moments')
    plt.ylabel('Correlation Coefficient')
    plt.grid(color='#DDDDDD')
    #plt.ylim(0,1.1)
    plt.show()