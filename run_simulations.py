##################################################################################
# run_simulations.py - Revamped version of main simulation code
#
# Author: Yash Patel
# University of Michigan
##################################################################################

from Single_CA1_neuron import sim_main
from make_figs import plot_run, plot_compilation

from PIL import Image

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
        "abruptness": 2.0,

        # Type of movement of rat around track ----------------------------------------
        "movement_type": "linear",

        "message": ""
    }

def current_conversion_exp():
    return [
        {
            "ExtraCurr_0": 1.0,
            "abruptness": 2.0,
            "hard_thresh": True,
        }, 
        {
            "ExtraCurr_0": 0.25,
            "abruptness": 2.0,
            "hard_thresh": True,
        }, 
        {
            "ExtraCurr_0": 0.5,
            "abruptness": 2.0,
            "hard_thresh": True,
        }, 
        {
            "ExtraCurr_0": 0.75,
            "abruptness": 2.0,
            "hard_thresh": True,
        }, 
        {
            "ExtraCurr_0": 0.90,
            "abruptness": 2.0,
            "hard_thresh": True,
        }, 
        {
            "ExtraCurr_0": 1.0,
            "abruptness": 2.0,
            "hard_thresh": True,
        }, 
        {
            "ExtraCurr_0": 1.5,
            "abruptness": 2.0,
            "hard_thresh": True,
        }, 
    ]

def movement_exp():
    return [
        {
            "movement_type": "linear",
        },
        {
            "movement_type": "constant",
        },
        {
            "movement_type": "back_forth",
        },
        {
            "movement_type": "random_walk",
        },
    ]

if __name__ == "__main__":
    merge_pictures = False
    if merge_pictures:
        movement_types = [
            "linear",
            "constant",
            "back_forth",
            "random_walk",
        ]

        for movement_type in movement_types:
            fnames = [
                "./Figures/" + f"movement_type={movement_type}_0-dendritic_weights.png",
                "./Figures/" + f"movement_type={movement_type}_25-dendritic_weights.png",
                "./Figures/" + f"movement_type={movement_type}_50-dendritic_weights.png",
                "./Figures/" + f"movement_type={movement_type}_3000-dendritic_weights.png",
                "./Figures/" + f"movement_type={movement_type}_6000-dendritic_weights.png",
                "./Figures/" + f"movement_type={movement_type}_9000-dendritic_weights.png",
            ]

            images = [Image.open(x) for x in fnames]
            widths, heights = zip(*(i.size for i in images))

            total_width = widths[0] * 2
            max_height = heights[0] * 3

            new_im = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for col in range(2):
                for row in range(3):
                    im = images[col * 3 + row]
                    new_im.paste(im, (col * widths[0], row * heights[0]))
                    
            new_im.save(f"merged_{movement_type}.png")

    run_id_to_config = movement_exp()
    run_ids = []

    for exp in run_id_to_config:
        sim_params = config()
        for change in exp:
            sim_params[change] = exp[change]

        run_id = "movement_type={}".format(sim_params["movement_type"])
        run_ids.append(run_id)

        sim_main(sim_params, run_id=run_id)
        plot_run(sim_params, run_id=run_id)
    
    compiled_name = "hthresh={}_abruptness={}_mean".format(sim_params["hard_thresh"], sim_params["abruptness"])
    plot_compilation(sim_params, run_ids=run_ids, compile_name=compiled_name)

    current = 0.75
    included_trials = [
        "hthresh=True_abruptness=0.5",
        "hthresh=False_abruptness=0.5",
        "hthresh=False_abruptness=2.0",
        "hthresh=False_abruptness=5.0",
    ]
    included_trials = [f"current={current}_" + included_trial for included_trial in included_trials]
    plot_compilation(sim_params, run_ids=included_trials, compile_name=f"current={current}_final",
        labels=["hard threshold", "abruptness=0.5", "abruptness=2.0", "abruptness=5.0"])
