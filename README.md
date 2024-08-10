# Expressiveness of Line Graph Neural Networks



## Introduction

Welcome to this project, in which I researched the expressiveness of Line Graph Neural Networks! Please find the project report [here](report.pdf).
With the code in this repository, I obtained the results of the experiment at the end of the report.


## Structure

### Files

| File          | Description                                      |
| ------------- | ------------------------------------------------ |
| `code/main.ipynb`  | The main results displayed in the report, as well as the used graph figures, can be found in `code/main.ipynb`. To run this notebook yourself, follow the procedure in the section 'Running `code/main.ipynb`' of this file.      |
| `code/dataset.py`  | The code in this file was used to preprocess the datasets used in these experiments (i.e. sample subgraphs for link prediction).        |
| `code/experiment.py` | The code in this file was used to train the models.  |
| `code/models.py` | This file contains the model implementations. |
| `code/utils.py`    | This file contains utility functions for the rest of the project.       |


### Folders

| Folder          | Description                                      |
| ------------- | ------------------------------------------------ |
| `code/study` | Contains the TensorBoard logs of the three experiments (subfolder names: `PPI-hidden-dim-20`, `PPI-hidden-dim-52`, `TwitchEN-hidden-dim-20`). |
| `code/data` | Is supposed to contain the dataset. If you want to run `code/main.ipynb` yourself, you will need to fill this folder with the right data, as outlined in the next section. |
| `report` | Contains the TeX code used to generate the report. |



## Running `code/main.ipynb`

To run `code/main.ipynb`, you will need to download the datasets from the following link: https://drive.google.com/drive/folders/1KiYGXAuR-3VBO31yu82S8QLrxGLgwc_2?usp=sharing
and create the following folder structure in the `code/data` folder.
Then, everything should run!

| data <br>
|---- TwitchENDataset <br>
|---- PPIDataset
