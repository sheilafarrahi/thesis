import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF 
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


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

def get_longtail_distribution():
    longtail_distributions = {
        "cauchy" : stats.cauchy(),
        "lognorm_1.5" : stats.lognorm(s=1.5),
        "pareto_3" : stats.pareto(b=3),
        "pareto_1" : stats.pareto(b=1),
        "weibull_min_0.5" : stats.weibull_min(c=0.5),
        "weibull_max_0.3" : stats.weibull_max(c=0.3)
    }
    return longtail_distributions
        

def plot_histograms_of_samples(distributions_dict, nr_sample, s_size):
    for i, (name, distr) in enumerate(distributions_dict.items()):
        fig, ax = plt.subplots()
        samples = distr.rvs(size=(s_size, nr_sample), random_state=10)
        ax.hist(samples, density=True, histtype='stepfilled', bins='auto', label=name, alpha=0.1)
        #ax.set_xlim(0,1)
        ax.legend()


def get_moments_df(distributions_dict, nr_moments, nr_sample, s_size):
    m1 = list()
    x = np.zeros((nr_moments,nr_sample))
    df = pd.DataFrame()

    for i, (name, distr) in enumerate(distributions_dict.items()):
        samples = distr.rvs(size=(nr_sample, s_size), random_state=10)
        m1.extend(np.mean(samples, axis=1)) # first moment

        for j in range(2,nr_moments+2): #calculate from 2nd moment
            x[j-2,:] = stats.moment(samples, j, axis=1)

        df_per_dist = pd.DataFrame(np.transpose(x))
        df_per_dist['dist'] = name
        df = pd.concat([df,df_per_dist], ignore_index=True)

    m1_df = pd.DataFrame(m1)
    final_df = pd.concat([m1_df,df], axis=1)

    return final_df 


def get_kde_estimates(distributions_dict, nr_sample, s_size, x_values, bandwidth):
    df = pd.DataFrame()
    for i, (name, distr) in enumerate(distributions_dict.items()):
        y_estimates = list()
        samples = distr.rvs(size=(nr_sample, s_size), random_state=10)

        for j in range(nr_sample):
            X = samples[j,:]
            X = X.reshape((len(X),1))

            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)

            values = np.asarray(np.linspace(0,1,x_values))
            values = values.reshape((len(values),1))

            log_density = kde.score_samples(values)
            y_estimates.append(np.exp(log_density))

        df_per_dist = pd.DataFrame(y_estimates)  
        df_per_dist['dist'] = name

        df = pd.concat([df,df_per_dist], ignore_index=True)

    return df 


def get_edf(distributions_dict, nr_sample, s_size):
    df = pd.DataFrame()
    for i, (name, distr) in enumerate(distributions_dict.items()):
        x_values = list()
        samples = distr.rvs(size=(s_size, nr_sample), random_state=10)
        for j in range(nr_sample):
            ecdf= ECDF(samples[:,j])
            x_values.append(ecdf.x)

        df_per_dist = pd.DataFrame(x_values)
        df_per_dist['dist'] = name

        df = pd.concat([df, df_per_dist], ignore_index = True)
    return df   