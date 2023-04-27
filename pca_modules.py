from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import colorcet as cc

def perform_pca(df, n):
    # df
    # n: number of components
    pca = PCA(n_components=n)
    x_pca = pca.fit_transform(df.iloc[:,:-1])
    result = pd.DataFrame(x_pca)
    result.columns= ['PCA'+str(i) for i in range(1,n+1)]
    result['y']=df['label']
    return result

def plot_pca(pca_result, X, Y, n_colors):
    # X, Y: which components (PCA1, PCA2, ...)
    custom_palette = sns.color_palette(cc.glasbey, n_colors=n_colors)
    sns.scatterplot(data=pca_result,x=X, y=Y,hue='y', palette = custom_palette)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, ncol=3)
    