from sklearn.decomposition import PCA
import pandas as pd

def get_pca(data, n_components):
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(data.iloc[:,:-1])
    result = pd.DataFrame(x_pca)
    result.columns=['PCA'+str(i+1) for i in range(n_components)]
    result['y']=data.iloc[:,-1]
    return result

