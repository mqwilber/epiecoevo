# Epi-eco-evo model for host recovery following disease-induced declines

This repo contains the code necessary to replicate the analyses found in the manuscript
"Towards a theory of host recovery dynamics following disease-induced declines: an epi-eco-evo perspective".  Below, we describe the files necessary to replicate these
analyses and explore the E3 model.

- `environment.yml`: This file contains the details of the Python environment where all of the analyses can be performed.  To build a Python environment using the Anaconda Distribution, [see this tutorial](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).  The Anaconda Python Distribution can be freely downloaded [here](https://www.anaconda.com/products/distribution).  Note that when building the environment, install the commented-out packages after the initial conda install as follows
	- `conda install pip; conda install ipython; pip install -v jedi==0.18.1`

- `code/`: Folder that contains the functions and code necessary to replicate the analyses found in the main manuscript
	- `anatomy_of_recovery_analysis.ipynb`: A Jupyter notebook that regenerates most of the figures in the main text and specifies which scripts need to be run to replicate the other figures and analyses.  This is the place to start.
	-  `anatomy_of_recovery_analysis.html`: An HTML version of the notebook for easy viewing.
	- `compare_resistance_and_tolerance*.py`: Scripts used to answer question 2 in the manuscript regarding differences in recovery trajectories between hosts using resistance vs. tolerance strategies. One script performs the analyses with the full E3 model and one script performs the analyses with the reduced/moment-closure models.
	- `epiecoevo_function.py`: The functions used to implement and explore the the E3 model developed in the manuscript.  Functions are documented within the script.
	- `pcc_full_model_comparison.py`: A script that addresses question 1 in the manuscript using the full E3 model.  This complements a similar analysis performed in `anatomy_of_recovery_analysis.ipynb` using the reduced E3 model.