# We want to determine the rank of these signals across different speeds and subjects.
# Since the magnitudes of different signals vary significantly, we cannot concatenate them directly.
# Instead, we will combine all steps from one speed within one subject, collected in one session and direction (x, y, z).
# Across rows, we will have different subjects and different speeds.
# Each matrix will represent a distinct signal type.

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def rank_check(arr):
    (_, s, _) = np.linalg.svd(np.asarray(arr)-np.mean(arr), full_matrices=False)
    s2 = np.power(s, 2)
    spectrum = np.cumsum(s2)/np.sum(s2)
    return spectrum, s2

def plot_rank_check(arr,title, axs=None):
    assert axs is None or len(axs) == 2

    spectrum, s2 = rank_check(arr)
    if axs is None:
        fig,axs = plt.subplots(2)

    axs[0].plot(spectrum, label=title)
    axs[0].set_title("Cumulative energy")
    axs[0].grid("on")
    axs[0].set_xlim(0,20)
    axs[0].set_xticks(np.arange(0, 21,2))

    
    axs[1].plot(s2, label=title)
    axs[1].grid("on")
    axs[1].set_xlabel("Ordered Singular Values") 
    axs[1].set_ylabel("Energy")
    axs[1].set_xlim(0,20)
    axs[1].set_xticks(np.arange(0,21,2))
    axs[1].set_yscale("log")


    axs[1].set_title("Singular Value Spectrum")
    axs[1].legend(loc='lower right')

    plt.tight_layout()

def create_signal_matrix(base_path):
    files = glob.glob(os.path.join(base_path, "AB*treadmill*.npz"))
    signal_names = ["shank_Accel", "shank_Gyro", "trunk_Accel", "trunk_Gyro", "GRF"]
    result = np.array([])
    for f in files:
        data = np.load(f)
        imu = data['imu']
        grf = data['grf']
        data = np.concatenate((imu.transpose(1,0,2,3,4), np.expand_dims(grf, axis=0)), axis=0)
        result = np.concatenate((result, data), axis=1) if result.size else data
        print(result.shape)

    
    for i, signal in enumerate(signal_names):
        np.savez(f"./local_results/{signal}.npz", data=result[i])

from numpy import linalg as LA
# reshape_data = lambda data: data.reshape(data.shape[0],-1)
def reshape_data(data):
    norm = LA.norm(data,ord=2, axis=2)
    print(data.shape,norm.shape)
    return norm.reshape(norm.shape[0],-1)


        
def generate_rank_plots(individual_plots=False):
    if not individual_plots:
        fig,axs = plt.subplots(2)
    for signal in ["shank_Accel", "shank_Gyro", "trunk_Accel", "trunk_Gyro", "GRF"]:
        data = np.load(f"./local_results/{signal}.npz")['data']
        if individual_plots:
            plot_rank_check(reshape_data(data), signal)
            plt.savefig(f"./local_results/{signal}.png")
            fig,axs = plt.subplots()
            axs.plot(reshape_data(data).T)
            axs.set_xlim(0,500)
            axs.set_title(signal)
            plt.savefig(f"./local_results/{signal}_raw.png")
        else:        
            plot_rank_check(reshape_data(data), signal, axs)
    plt.savefig(f"./local_results/rank_check.png")

if __name__ == '__main__':
    # create_signal_matrix("./local_results/")
    generate_rank_plots(False)
