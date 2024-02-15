# HPC_SO23
This repository contains the implementation of HPC course of Master's curriculum at UNI-Freiburg.
The code is modularized except for the parallel implementation in the cluster.  
---
## Simulations
The different simulations implemented in this repository are:

- **Milestone 1 (m1)**: Simple Lattice Boltzmann Method
- **Milestone 2 (m2)**: Simple Lattice Boltzmann Method with Collision operator
- **Milestone 3 (swd)**: Shear Wave Decay
- **Milestone 4 (cf)**: Couette Flow
- **Milestone 5 (pf)**: Poiseuille Flow
- **Milestone 6 (sls)**: Sliding Lid Serial
- **Milestone 7 (slp)**: Sliding Lid Parallelization
---
## Lattice Boltzmann
The simulation_func directory contains the implementation of the Lattice Boltzmann method along with all the boundary 
conditions and functions handled in separate functions with function descriptions for help.  
---
## Utils
The utils folder contains the files which require the utility functions for generating the sinusoidal waves and 
viscosity calculations.
---
## Experiments
All the experiments are labelled intuitively with the file names stating the functionality.
The implementation follows creating an object of class Lattice Boltzmann and performing different tasks on it.
---
## Cloning the Repository and Installing Dependencies

Please follow the below instructions to clone this repository and install the requirements.

1. Open a terminal and move to your workspace.
2. Run `git clone https://github.com/RK-UNI-Freiburg/HPC_SO23.git`.
3. Move into the cloned repository by running `cd HPC_2023`.
4. Create a virtual environment using Anaconda Prompt:
   - Open Anaconda Prompt.
   - Run `conda create --name HPC python=3.9 -y`.
   - Activate the environment with `conda activate HPC`.
5. Set up a Python Interpreter in PyCharm:
   - Open the `HPC_2023` repository via PyCharm.
   - Click **Files > Settings**.
   - Select **Project: HPC_2023**.
   - Click on **Python Interpreter > Add Interpreter > Add Local Interpreter > Conda Environment**.
   - Select `HPC` under **Use existing environment**.
   - Apply the settings.
6. Open a terminal in PyCharm - the conda virtual environment should be activated.
7. Run `pip install -r requirements.txt` to install the required dependencies.
8. Optional (for contributors): If you install any new Python libraries, update the `requirements.txt` 
9. file by running `pip freeze > requirements.txt`. Push the updated `requirements.txt` file to the Git repository.
---
## Running the code

To run the various milestones now, type in the terminal :
1. python Shear_Wave_Decay.py
2. python Couette_Flow.py
3. python Poiseuille_Flow.py
4. python Sliding_Lid.py (for serial implementation)
5. python Sliding_Lid_Parallel.py  (for parallel implementation on local devices. Not advised if system is not latest)
6. To run the code in the cluster, the code has benn simplified into a single .py file for easy handling of files on 
   the cluster. Just copy in the file Sliding_Lid_Parallel_Cluster.py to the cluster and create a .sh job file . 
   Example can be found in the Cluster folder here. Update the output file names in the job file along with specifying 
   the nodes and the number of processes.
7. The different lattice sizes can be manipulated from the file Sliding_Lid_Parallel_Cluster.py,
8. Run the .sh file on the cluster

---

## Note
The figures folder contains all the plots and simulations. 

---
### Thank you