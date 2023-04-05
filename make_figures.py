import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from time import sleep
import matplotlib.pyplot as plt

import distribution_modules as dm
import density_estimation_modules as dem
import classification_modules as cm
import importlib

def make_ecf_plots():
    # set configuration
    sample_size = 100
    nr_sample = 20
    sample_config = [sample_size, nr_sample]

    # classificatiom
    test_size = 0.2
    cv = 5
    cv_config = [test_size, cv]

    bounded_dists = dm.get_bounded_distribution()

    num_steps_list = np.arange(2,20,2)
    step_size_list = np.arange(0.0001,2,0.05)*np.pi

    ss = [20, 50, 100, 300, 500, 750, 1000]
    for i in ss:
        sample_config = [i, nr_sample]
        acc = cm.cv_num_steps_step_size(step_size_list, num_steps_list, bounded_dists, sample_config, cv_config, 2)
            
    fig, ax = plt.subplots(figsize=(10,8))
    for accuracy, ss_ in zip(acc, ss):
        plt.plot(step_size_list, acc[i], label=str(num_steps_list[i]), alpha = 0.5)
        plt.title(f'accuracy for different step size and number of steps, sample size = {ss_}')
        plt.xlabel('step size')
        plt.ylabel('accuracy')
        
        pos = ax.get_position()
        ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5), title='number of steps')
        plt.savefig(f'ecf_cv_plot_{ss_}')
    
if __name__=='__main__':
    make_ecf_plots()
    

