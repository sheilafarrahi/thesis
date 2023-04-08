import pandas as pd
import numpy as np
import scipy.stats as stats
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
        
    return samples_dict


def plot_histograms_of_samples(samples_dict):
    for i, (name, samples) in enumerate(samples_dict.items()):
        fig, ax = plt.subplots()
        ax.hist(samples, density=True, histtype='stepfilled', bins='auto', label=name, alpha=0.1)
        ax.legend()