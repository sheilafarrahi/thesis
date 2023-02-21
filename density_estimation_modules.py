import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.neighbors import KernelDensity

##################################
#        methods of moments      #
##################################

def get_moments_df(distributions_dict, nr_moments, nr_sample, sample_size):
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
        ax.hist(moments_df.iloc[:,i], density=True, histtype='stepfilled', bins='auto')
        plt.title('moment ' + str(i))
        

##################################
#    Kernel density estimation   #
##################################

def get_kde_estimates(distributions_dict, nr_sample, sample_size, x_values, bandwidth):
    df = pd.DataFrame()
    for i, (name, distr) in enumerate(distributions_dict.items()):
        y_estimates = list()
        samples = distr.rvs(size=(nr_sample, sample_size), random_state=10)

        for j in range(nr_sample):
            X = samples[j,:]
            X = X.reshape((len(X),1))

            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)

            values = np.linspace(0,1,x_values)
            values = values.reshape((len(values),1))

            log_density = kde.score_samples(values)
            y_estimates.append(np.exp(log_density))

        df_per_dist = pd.DataFrame(y_estimates)  
        df_per_dist['dist'] = name

        df = pd.concat([df,df_per_dist], ignore_index=True)

    return df 