##################################################################################
# make_figs.py - This creates JUST the figures for the extension work
#
# Author: Yash Patel
# University of Michigan
##################################################################################

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions to create figures ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_FR(tVec, FR, color, xlabel="Position", ylabel="Somatic activity", label=" ", alpha=1):
    """
    This function creates a plot of the firing rate over time by default. The x and y
    labels can be modified
    """
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim((0, 50.01))
    plt.ylim(-0.01, 2.)
    plt.plot(tVec, FR, lw=2, label=label, color=color, alpha=alpha)

    plt.tick_params(axis="y", labelcolor="k")
    plt.tick_params(axis="x", labelcolor="k")

def plot_run(sim_pars, run_id):
    Dir = "my_runs/" + run_id + "/Data_trials/"
    fnames = os.listdir(Dir)

    fnames.sort()

    # load all the data from the file
    with open(Dir + fnames[0], "rb") as pickle_in:
        data_all = pickle.load(pickle_in)
        pickle_in.close()

    # extract the firing rates for soma and dendrites
    Soma_FRs = np.array([data_all[tr_id]["soma"] for tr_id in data_all.keys()])
    Dends_FRs = np.array([data_all[tr_id]["dendrites"] for tr_id in data_all.keys()])
    Syn_weights = np.array([data_all[tr_id]["Wpre_to_dend"] for tr_id in data_all.keys()])
    ExtraCurr = np.array([data_all[tr_id]["ExtraCurr"] for tr_id in data_all.keys()])
    n_laps = np.array([data_all[tr_id]["n_laps"] for tr_id in data_all.keys()])[0]

    # take an average over all trials
    Soma_FRs_ave = np.mean(Soma_FRs, axis=0)
    Dends_FRs_ave = np.mean(Dends_FRs, axis=0)
    Syn_weights_ave = np.mean(Syn_weights, axis=0)
    ExtraCurr_ave = np.mean(ExtraCurr, axis=0)

    # create a vector with all the time points
    tVec = np.linspace(0, data_all[0]["t_explore"], Soma_FRs.shape[1])
    tVec2 = np.repeat([tVec], 50, axis=0)

    fig_dir = r"./Figures/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    FR = Soma_FRs_ave

    points_lap = int(FR.shape[0] / n_laps)
    RF_develop = FR.reshape((-1, points_lap))

    Dend_RF = Dends_FRs_ave[:, 0].reshape((-1, points_lap))

    n_laps = sim_pars["n_laps"]

    pos = np.linspace(0, 50, points_lap)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Plot final activation curve
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    plot_FR(pos, RF_develop[n_laps-1], "#FF9900", ylabel="")
    plt.savefig(fig_dir + "Run_{0:03d}-final_activity.png".format(int(run_id)), dpi=300, transparent=True)
    plt.show()
    plt.cla()

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

    plt.savefig(fig_dir + "Run_{0:03d}-learning.png".format(int(run_id)), dpi=300, transparent=True)
    plt.show()
    