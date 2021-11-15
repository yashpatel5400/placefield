##################################################################################
# run_simulations.py - Revamped version of main simulation code
#
# Author: Yash Patel
# University of Michigan
##################################################################################

from Single_CA1_neuron import sim_main
from make_figs import plot_run

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Parameters to be saved for each the experiment -------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def config():
    return {
        # Parameters ---------------------------------------------------------------------------------
        "T_length": 50.,     # Length of the track
        "N_pre": 10,         # Number of neurons in the presynaptic layer
        "Nth": 1.0,          # "Spiking threshold" (there is actually no threshold - rate-based neurons)
        "N_dend": 2,         # Number of dendritic compartments
        "ExtraCurr_0": 1.0,   # Extra current injected at the soma

        # Simulation parameters --------------------------------------------------------------
        "dt": 1.,            # [ms] Simulation time step

        # Excitatory synaptic plasticity -----------------------------------------------------
        "eta_Extra": 0.e-4,  # [ms^-1] Learning rate for the extra currents onto pyramidal neurons
        "eta_input": 2e-4,   # [ms^-1] Learning rate for the input weights from prelayer neurons
        "eta_homeo": 2e-4,   # [ms^-1] Learning rate for homeostatic plasticity

        # Novelty signal ---------------------------------------------------------------------
        "Idend_target": 8.5, # Dendritic inhibition in familiar environments
        "Isoma_target": 0.,  # Somatic inhibition in familiar environments
        "Idend_0": 0.8,      # Initial dendritic inhibition
        "Isoma_0": 1.2,      # Initial somatic inhibition
        "tau_nov": 100e3,    # [ms] Novelty signal time constant

        # Presynaptic place field ------------------------------------------------------------
        "PF_amp": 2.2,       # Amplitude of presynaptic place fields

        # Experiment parameters --------------------------------------------------------------
        "n_laps": 100,       # Number of laps the subjets runs for each trial
        "v_run": 1e-2,       # Running speed
        "noise_input": 0.05, # amplitude of noise for input neurons

        # Firing rate parameters -------------------------------------------------------------
        "eta_FR": 2.e-1,     # [ms^-1] Learning rate for the firing rates

        # Number of trials for multi-trial experiment ----------------------------------------
        "NTrials": 1,

        # Number of trials for multi-trial experiment ----------------------------------------
        "theta_prop": 0.2,
        "hard_thresh": True,

        "message": ""
    }

def current_conversion_exp():
    return {
        "0": {
            "ExtraCurr_0": 1.0,
        }, 
        "1": {
            "ExtraCurr_0": 1.0,
        }, 
        "2": {
            "ExtraCurr_0": 1.5,
        }, 
    }

if __name__ == "__main__":
    run_id_to_config = current_conversion_exp()

    for run_id in run_id_to_config:
        sim_params = config()
        for change in run_id_to_config[run_id]:
            sim_params[change] = run_id_to_config[run_id][change]

        # sim_main(sim_params, run_id=run_id)
        plot_run(sim_params, run_id=run_id)
