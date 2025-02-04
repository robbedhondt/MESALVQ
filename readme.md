# Multi-Event Survival Analysis-Informed Clustering
This repository hosts the code used in the publication "Multi-Event Survival Analysis-Informed Clustering with MESA-LVQ: Functional System Worsening in Multiple Sclerosis".

# Setup
- Environment setup: in the paper, Python 3.11 was used with the following packages (for more details on version specifics, see `requirements.txt`):
    ```bash
    pip install numpy pandas matplotlib seaborn torch scikit-learn scikit-survival genieclust
    ```
- Clone this repository, including SurvivalLVQ which is included as a submodule:
    ```bash
    git clone --recurse-submodules git@github.com:robbedhondt/MESALVQ.git
    ```
    Note: if you cloned without `--recurse-submodules`, you can retrieve the submodule with `git submodule update --init` or, alternatively, by cloning the SurvivalLVQ repo into the mesaLVQ folder `cd mesaLVQ; git clone git@gitlab.kuleuven.be:u0125808/survivallvq.git`.


