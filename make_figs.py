##################################################################################
# make_figs.py - This creates JUST the figures for the extension work
#
# Author: Yash Patel
# University of Michigan
##################################################################################

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pickle
import imageio
from difflib import SequenceMatcher

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions to create figures ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_FR(tVec, FR, xlim=(0, 50.01), ylim=(-0.01, 2.), xlabel="Position", ylabel="Somatic activity", label=" ", alpha=1):
    """
    This function creates a plot of the firing rate over time by default. The x and y
    labels can be modified
    """
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    plt.ylim()
    plt.plot(tVec, FR, lw=2, label=label, alpha=alpha)

    plt.legend(loc="upper right")

    plt.tick_params(axis="y", labelcolor="k")
    plt.tick_params(axis="x", labelcolor="k")

def load_run_data(run_id):
    Dir = "my_runs/" + run_id + "/Data_trials/"
    fnames = os.listdir(Dir)

    fnames.sort()

    # load all the data from the file
    with open(Dir + fnames[0], "rb") as pickle_in:
        data_all = pickle.load(pickle_in)
        pickle_in.close()

    # extract the firing rates for soma and dendrites
    positions =  np.array([data_all[tr_id]["positions"] for tr_id in data_all.keys()])
    Soma_FRs = np.array([data_all[tr_id]["soma"] for tr_id in data_all.keys()])
    Dends_FRs = np.array([data_all[tr_id]["dendrites"] for tr_id in data_all.keys()])
    Syn_weights = np.array([data_all[tr_id]["Wpre_to_dend"] for tr_id in data_all.keys()])
    ExtraCurr = np.array([data_all[tr_id]["ExtraCurr"] for tr_id in data_all.keys()])

    n_laps = np.array([data_all[tr_id]["n_laps"] for tr_id in data_all.keys()])[0]

    # take an average over all trials
    positions = np.mean(positions, axis=0)
    Soma_FRs_ave = np.mean(Soma_FRs, axis=0)
    Dends_FRs_ave = np.mean(Dends_FRs, axis=0)
    Syn_weights_ave = np.mean(Syn_weights, axis=0)
    ExtraCurr_ave = np.mean(ExtraCurr, axis=0)

    # create a vector with all the time points
    tVec = np.linspace(0, data_all[0]["t_explore"], Soma_FRs.shape[1])
    
    return positions, Soma_FRs_ave, Dends_FRs_ave, Syn_weights_ave, tVec

def plot_run(sim_pars, run_id):
    positions, Soma_FRs_ave, Dends_FRs_ave, Syn_weights_ave, tVec = load_run_data(run_id)

    fig_dir = r"./Figures/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # plt.plot(tVec, positions)
    # plt.savefig(fig_dir + f"{run_id}-positions.png", dpi=300, transparent=True)
    # plt.clf() 

    fnames = []

    rendered_subset = 100
    multiplier = 1
    for i in range(rendered_subset):
        idx = multiplier * i

        dendrite_weights = Syn_weights_ave[idx,0,:]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        ax1.set_xlabel("Synapse ID")
        ax1.set_ylabel("Synaptic Weight")
        ax1.scatter(list(range(dendrite_weights.shape[-1])), dendrite_weights)

        ax2.set_xlim((0, 50))
        ax2.set_xlabel("Position")
        ax2.plot([positions[idx] % 50], [0], 'ro')
        
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Position")
        ax3.plot(tVec[:(multiplier * rendered_subset)], positions[:(multiplier * rendered_subset)])
        ax3.plot(tVec[idx], positions[idx], 'ro')

        # plt.show()

        fname = fig_dir + f"{run_id}_{idx}-dendritic_weights.png"
        plt.savefig(fname, dpi=300, transparent=True)
        plt.clf() 
        fnames.append(fname)

    with imageio.get_writer(f"acc_weight_evolution_{run_id}.gif", mode='I') as writer:
        for i, fname in enumerate(fnames):
            image = imageio.imread(fname)
            writer.append_data(image)

            if i == 0:
                writer.append_data(image)
                writer.append_data(image)
                writer.append_data(image)
                writer.append_data(image)
                writer.append_data(image)
        
    n_laps = sim_pars["n_laps"]

    points_lap = int(Soma_FRs_ave.shape[0] / n_laps)
    RF_develop = Soma_FRs_ave.reshape((-1, points_lap))
    Dend_RF = Dends_FRs_ave[:, 0].reshape((-1, points_lap))
    pos = np.linspace(0, 50, points_lap)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Plot final activation curve
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    plot_FR(pos, RF_develop[n_laps-1], xlim=(15,45), ylim=(0,1), xlabel="Position (a.u.)", ylabel="Mean activity")
    plt.savefig(fig_dir + f"{run_id}-final_activity.png", dpi=300, transparent=True)
    # plt.show()
    plt.clf()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Plot mean activity curve (over learning)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    soma = Soma_FRs_ave.reshape((n_laps, -1))
    dend = Dends_FRs_ave[:,0].reshape((n_laps, -1))

    soma_mean = np.mean(soma, axis=1)
    dend_mean = np.mean(dend, axis=1)

    plt.ylabel("Mean Activity")
    plt.xlabel("Lap")
    plt.plot(dend_mean, "k--", lw=2, label="Dendrite")
    plt.plot(soma_mean, "k-", lw=2, label="Soma")
    plt.legend()

    plt.savefig(fig_dir + f"{run_id}-mean_activity.png", dpi=300, transparent=True)
    # plt.show()
    plt.clf()
    
def plot_compilation(sim_pars, run_ids, compile_name, labels=[]):
    fig_dir = r"./Figures/"
    n_laps = sim_pars["n_laps"]
    plot_final = True

    for i, run_id in enumerate(run_ids):
        positions, Soma_FRs_ave, Dends_FRs_ave, Syn_weights_ave, tVec = load_run_data(run_id)
        
        points_lap = int(Soma_FRs_ave.shape[0] / n_laps)
        RF_develop = Soma_FRs_ave.reshape((-1, points_lap))
        Dend_RF = Dends_FRs_ave[:, 0].reshape((-1, points_lap))
        pos = np.linspace(0, 50, points_lap)

        if len(labels) == 0:
            label = run_id.split("_")[0]
        else:
            label = labels[i]

        if plot_final:
            plot_FR(pos, RF_develop[n_laps-1], xlim=(25,35), ylim=(0,1), ylabel="", label=label)
        else:
            soma = Soma_FRs_ave.reshape((n_laps, -1))
            dend = Dends_FRs_ave[:,0].reshape((n_laps, -1))

            soma_mean = np.mean(soma, axis=1)
            dend_mean = np.mean(dend, axis=1)

            plt.ylabel("Mean Activity")
            plt.xlabel("Lap")
            if i == 0:
                plt.plot(dend_mean, "k--", lw=2, label="Dendrite")
            plt.plot(soma_mean, "k-", lw=2, label=label)
            plt.legend()

    # plt.show()
    plt.savefig(fig_dir + f"{compile_name}-compiled.png", dpi=300, transparent=True)
    plt.clf()