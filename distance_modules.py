from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def w_distance(df):
    labels = df.iloc[:,-1].unique()
    nr_labels = len(labels)
    d_matrix = np.zeros((nr_labels,nr_labels))
    for n in range(nr_labels):
        for m in range(nr_labels):
            data_part1 = df.loc[df.iloc[:,-1]==labels[n]]
            data_part2 = df.loc[df.iloc[:,-1]==labels[m]]
            distance = list()
            for i in range(0,len(data_part1)):
                for j in range(0,len(data_part2)):
                    distance.append(wasserstein_distance(data_part1.iloc[i,:-1],data_part2.iloc[j,:-1]))
            d = np.mean(distance)
            d_matrix[n][m]=d
    return d_matrix

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