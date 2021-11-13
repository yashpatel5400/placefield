# placefield
Studying the exhibited behavior of silent neurons turning into place field neurons in the hippocampus

---
## Original README

The interplay between somatic and dendritic inhibition promotes the emergence and stabilization of place fields
Victor Pedrosa and Claudia Clopath
General description
Simulates a feedforward network in which the postsynaptic cell is modelled as a two-compartment neuron as in:
  

[1]  Pedrosa, V., Clopath, C. The interplay between somatic and dendritic inhibition promotes the emergence and stabilization of place fields. 
PlosCB (2020).
  

Code written by: Victor Pedrosa 
  v.pedrosa@imperial.ac.uk
  

Imperial College London, London, UK - Jan 2020
List of files
(1) run_simulations.py
  

This file runs all the code in files 2, 3 and 4, generating the data in ‘my_runs/‘ and the 
figures in 'Figures/'.
  

(2) Single_CA1_neuron.py
  

Simulates a feedforward network of rate-based neurons with plastic excitatory
synapses and novelty signal.
  

(3) Make_figs0.py
  

Plots and save the figure generated with the data produced from (1) and (2). Figures are 
saved in Figures/. This code extracts the data and plots the evolution of the mean neuron activity.
  

(4) Make_figs1.py
  

Plots and save the figure generated with the data produced from (1) and (2). Figures are 
saved in Figures/. This code extracts the data and plots the evolution of place fields (somatic and dendritic).
  

  

To simulate the network and plot the figures
1. run (1): simulates the network, saves the results and generates the figures found in Figures;