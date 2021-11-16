# -----------------------------------------------------------------------
# Code designed to simulate one single neuron with 3 compartments (2 parallel
# dendrites + 1 soma) receiving place-tuned input. the following properties are 
# implemented:
# 1. Each dendritic compartment is activated following a nonlinear function of 
#    the weighted sum of inputs
# 2. Input neurons are simulated. Their firing rate will depend on the
#    animal’s location and will be determined by a non-gaussian place field.
# 3. The learning rule is local on the dendrites. It depends on presynaptic
#    activity and dendritic activity. Therefore, inputs coming to different 
#    dendrites will learn independently while inputs coming to the same dendrite 
#    will have a correlated learning rule. 
#
# -----------------------------------------------------------------------
#
# Author: Victor Pedrosa
# Imperial College London, London, UK - Jan 2020
#
# Modified by: Yash Patel
# University of Michigan

import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Some functions to be used throughout the code --------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def _rect(x): return x * (x > 0.)  # Rectification


def _upper(x, max): return x - (x - max) * (x > max)  # apply an upper bound


def _lower(x, min): return x - (x - min) * (x < min)  # apply a lower bound


def pre_layer_input(pos, N_pre, T_length, PF_amp):
    """
    This function returns the firing rate of each neuron in the presynaptic layer as a function of
    the animal"s position
    """
    sigma = 5.  # width of place fields of presynaptic neurons
    amp = PF_amp  # amplitude of place fields of presynaptic neurons
    place_input = np.zeros(N_pre)

    # Receptive fields (Feedforward Excitatory synaptic weights):
    for n in np.arange(1, N_pre + 1):
        pos0 = (n - 1) * T_length / N_pre
        dist = np.abs((pos0 - pos + T_length / 2.) % T_length - T_length / 2.)
        place_input[(n - 1):n] = amp * np.exp(-(dist) ** 2 / (2. * sigma ** 2))

    return place_input

def g_dend(x):
    """
    This is the gain function of dendritic compartments for a given input current x

    input:
        x: input current (N_dend,)
    output:
        g_dend: firing rate of the dendritic compartments (N_dend,)
    """
    linear_length = 5.
    x = 2 * x

    g1 = _rect(np.tanh(x / linear_length))
    g2 = 0.5 * np.tanh((x - linear_length) * 2.) + 0.5
    g_dend = 2 * (2 * g1 + 1 * g2) / 3.

    return g_dend

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initialization of variables --------------------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Network(object):
    """
    Network of neurons

    inputs:
        pars is a dictionary with the following keys:
            N_pre  : Number of neurons in the presynaptic layer
            N_dend : Number of dendritic compartments

    Attributes:
        Wpre_d: Synaptic weights from presynaptic layer to dendrites
        ExtraCurr: Extra currents controlling the excitability of neurons (plastic)
        RtES: [Hz] Firing rate of somatic compartment
        RtED: [Hz] Firing rates of dendritic compartments
    """

    def __init__(self, pars):
        """
        Return a Network object
        """
        self.Wpre_d = np.zeros((pars["N_dend"], pars["N_pre"]))
        # self.Wpre_d[0, 4] = 1.
        self.Wpre_d[0, :] += np.ones(pars["N_pre"]) # _rect(np.random.normal(0, 1.0, pars["N_pre"]))
        self.ExtraCurr = pars["ExtraCurr_0"]

        # time-dependent variables
        self.RtES = 0.
        self.RtED = np.zeros(pars["N_dend"])

        # Dendritic and somatic currents
        self.I_dend = np.zeros(pars["N_dend"])
        self.I_soma = 0.


def construct_net(sim_pars):
    np.random.seed()
    return Network(sim_pars)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculations to be performed at every integration step -----------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def gate_soma(hard_thresh, Vsoma, theta_prop, abruptness):
    if hard_thresh:
        return int(Vsoma > theta_prop)
    return 1 / (1 + np.exp(-abruptness * (Vsoma + theta_prop)))

def sim_step(net, Pre_input, sim_pars):
    # Simulation parameters ------------------------------------------------------------------------------
    eta_FR = sim_pars["eta_FR"]
    Nth = sim_pars["Nth"]
    eta_Extra = sim_pars["eta_Extra"]
    eta_input = sim_pars["eta_input"]
    dt = sim_pars["dt"]
    Idend_target = sim_pars["Idend_target"]
    Isoma_target = sim_pars["Isoma_target"]
    tau_nov = sim_pars["tau_nov"]
    eta_homeo = sim_pars["eta_homeo"]
    noise_input = sim_pars["noise_input"]
    theta_prop = sim_pars["theta_prop"]
    hard_thresh = sim_pars["hard_thresh"]
    abruptness = sim_pars["abruptness"]

    # Get the values from the network --------------------------------------------------------------------
    Wpre_d = net.Wpre_d
    RESin = net.RtES
    REDin = net.RtED
    ExtraCurr = net.ExtraCurr
    I_soma = net.I_soma
    I_dend = net.I_dend

    # Compute the values for the next time step ----------------------------------------------------------

    Pre_input += noise_input * np.random.normal(0, 1, Pre_input.shape[0])
    Pre_input = _rect(Pre_input)

    # Calculate the firing rate for the postsynaptic neuron (and rectify it when negative)----------------
    Vsoma = ExtraCurr - I_soma
    RtES = RESin + eta_FR * dt * (-RESin + _rect(gate_soma(hard_thresh, Vsoma, theta_prop, abruptness) * np.sum(REDin) + Vsoma - Nth))
    RtED = REDin + eta_FR * dt * (-REDin + g_dend(np.dot(Wpre_d, Pre_input) - I_dend))

    RtES = RtES * (RtES > 0.)
    RtED = RtED * (RtED > 0.)

    # Update the feedforward weights ---------------------------------------------------------------------
    R_dend_mat = REDin.reshape((-1, 1))
    target_norm = 3.
    Wpre_d = Wpre_d + eta_input * dt * (np.dot(R_dend_mat, Pre_input.reshape((1, -1))))
    Wpre_d = (Wpre_d.T - eta_homeo * dt * R_dend_mat.T * (np.sum(Wpre_d, axis=1) - target_norm)).T

    Wpre_d = Wpre_d * (Wpre_d > 0.)  # rectify synaptic weights
    Wpre_d = _upper(Wpre_d, 3)

    # Update inhibition to simulate a novel environment becoming familiar --------------------------------
    I_dend = I_dend - (1. / tau_nov) * dt * (I_dend - Idend_target)
    I_soma = I_soma - (1. / tau_nov) * dt * (I_soma - Isoma_target)

    # Return the new values for the network --------------------------------------------------------------
    net.RtES = RtES
    net.RtED = RtED
    net.ExtraCurr = ExtraCurr
    net.Wpre_d = Wpre_d
    net.I_soma = I_soma
    net.I_dend = I_dend

    return net

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define the steps to be done during the experiment ----------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plasticity_trial(sim_pars):  # place map plasticity
    """
    """

    # Extract parameters from configuration --------------------------------------------------------------
    N_pre = sim_pars["N_pre"]
    N_dend = sim_pars["N_dend"]
    T_length = sim_pars["T_length"]
    dt = sim_pars["dt"]
    Idend_0 = sim_pars["Idend_0"]
    Isoma_0 = sim_pars["Isoma_0"]
    v_run = sim_pars["v_run"]
    PF_amp = sim_pars["PF_amp"]
    movement_type = sim_pars["movement_type"]

    # Create the network ---------------------------------------------------------------------------------
    net = construct_net(sim_pars)

    # -------------------------------------------------------------------------------------------------
    # Start actual experiment and let the animal explore the environment ------------------------------
    # -------------------------------------------------------------------------------------------------

    v = v_run   # [ms^-1] speed of the animal (the unit of length is abitrary)
    n_laps = sim_pars["n_laps"]  # number of laps
    t_explore = n_laps * T_length / v  # total time of experiment
    explore_steps = int(t_explore / dt)  # number of steps for integration

    # Track all the postsynaptic firing rates ---------------------------------------------------------
    positions = np.zeros((explore_steps))  # positions over time
    RESs = np.zeros((explore_steps))  # somatic firing rate
    REDs = np.zeros((explore_steps, N_dend))  # dendritic firing rate
    Wpre_ds = np.zeros((explore_steps, N_dend, N_pre))
    ExtraCurrs = np.zeros((explore_steps))

    def pos_exp_linear(t): 
        return 10 * v * t

    def pos_exp_constant(t): 
        return 10

    def pos_exp_back_forth(t): 
        return 40 + 10 * np.sin(t / 10)

    def pos_exp_random_walk(position, t): 
        if np.random.random() > 0.5:
            return position + 1
        return position - 1

    def pos_exp(position, t):
        if movement_type == "linear":
            return pos_exp_linear(t)
        elif movement_type == "constant":
            return pos_exp_constant(t)
        elif movement_type == "back_forth":
            return pos_exp_back_forth(t)
        elif movement_type == "random_walk":
            return pos_exp_random_walk(position, t)
        return -1

    pre_input = np.zeros((0, 10))
    for pos in range(int(T_length)):
        pre_input = np.vstack((pre_input, pre_layer_input(pos, N_pre, T_length, PF_amp)))
    trans_pre_input = pre_input.transpose()
    for presynaptic_neuron in range(10):
        plt.plot(list(range(int(T_length))), trans_pre_input[presynaptic_neuron, :], label=presynaptic_neuron)
    plt.legend()
    plt.savefig("receptive_fields.png")
    plt.clf()
    
    # Initialize inhibition for novel environments (low dend inhibition + high somatic inhibition)
    net.I_dend = 0 * net.I_dend + Idend_0
    net.I_soma = 0 * net.I_soma + Isoma_0

    position = 0
    for step in range(explore_steps):
        t_bin = step * dt
        position = pos_exp(position, t_bin)
        Pre_input = pre_layer_input(position, N_pre, T_length, PF_amp)

        # Call the simulation step to calculate all the values for next time step
        net = sim_step(net, Pre_input, sim_pars)

        # Save the results
        positions[step] = position
        RESs[step] = net.RtES
        REDs[step] = net.RtED
        Wpre_ds[step] = net.Wpre_d
        ExtraCurrs[step] = net.ExtraCurr

    print("Done")
    return positions, RESs, REDs, Wpre_ds, ExtraCurrs, t_explore, n_laps


# -----------------------------------------------------------------------------------------------------
# Call the function with the experiment and save the results (define one trial)

def run_trial(sim_pars, fname):  # place map plasticity
    positions,RESs, REDs, Wpre_ds, ExtraCurrs, t_explore, n_laps = plasticity_trial(sim_pars)
    return {
        "t_explore": t_explore,
        "soma": RESs,
        "dendrites": REDs,
        "Wpre_to_dend": Wpre_ds,
        "ExtraCurr": ExtraCurrs,
        "positions": positions,
        "n_laps": n_laps
    }


# -----------------------------------------------------------------------------------------------------
# Define the main funtion which will call other functions and run it in parallel

def sim_main(sim_pars, run_id):
    exp_dir = r"./my_runs/" + run_id + "/Data_trials/"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    fname = exp_dir + "/Firing_rates_Place_cell_soma_and_dends_all_trials.pickle"

    if os.path.exists(fname):
        print("Already completed trial for: {}!".format(run_id))
        return

    print("\n Starting simulations...\n")
    data = {}

    num_trials = int(sim_pars["NTrials"])
    for trial in range(num_trials):
        data[trial] = run_trial(sim_pars, fname)

    with open(fname, "wb") as pickle_out:
        pickle.dump(data, pickle_out)
        pickle_out.close()
