import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def get_distributions():
    distribution_classes_on_finite_interval = {
        "arcsin": stats.beta(a=0.5, b=0.5),
        "uniform": stats.beta(a=1, b=1),
        "beta_2_2": stats.beta(a=2, b=2),
        "beta_1_4": stats.beta(a=2, b=4),
        "beta_4_1": stats.beta(a=4, b=2)
    }
    return distribution_classes_on_finite_interval


def plot_density_functions(distributions_dict):
    fig, ax = plt.subplots()
    x_values = np.linspace(0, 1, 100)
    for name, distr in distributions_dict.items():
        pdf = distr.pdf(x_values)
        ax.plot(x_values, pdf, label=name)
    ax.legend()


def plot_histograms_of_samples(distributions_dict, num_samples=500):
    num_bins = int(np.sqrt(num_samples))
    bins = np.linspace(0, 1, num_bins)
    color_fn = plt.get_cmap("tab10")

    for i, (name, distr) in enumerate(distributions_dict.items()):

        fig, ax = plt.subplots()
        samples = distr.rvs(num_samples)
        ax.hist(samples, bins=bins, label=name, color=color_fn(i), alpha=0.6)
        ax.legend()

