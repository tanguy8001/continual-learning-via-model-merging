#  Model Merging for Continual Learning

We provide code for all the experiments presented in our paper.

The organization of code is as follows :
* Source code is present in `src` directory.
* Bash files\notebooks required to run some experiments along with all the hyperparameters used are in the `scripts` directory.
  

* Our model merging results:


|  ***MNIST***       | **MLPNet**           | **MLPLarge**         | **MLPHuge**          |
|----------------|-----------------------|-----------------------|-----------------------|
| **Joint Model**    | $ \pm $      | $ \pm $      | $ \pm $      |
| **Model A**    | $91.68 \pm 0.65$      | $92.11 \pm 0.36$      | $ \pm $      |
| **Model B**    | $87.56 \pm 0.21$      | $87.67 \pm 0.27$      | $ \pm $      |
| **AVG**        | $81.30 \pm 1.87$      | $85.75 \pm 0.41$      | $ \pm $       |
| **OT**         | $80.27 \pm 2.06$      | $85.42 \pm 0.56$      | $ \pm $       |
| **MPF (ours)** | **$97.32 \pm 0.07$**  | **$97.52 \pm 0.11$**  | **$ \pm $**  |


| ***CIFAR-10***     | **MLPNet**            | **MLPLarge**         | **MLPHuge**          |
|----------------|-----------------------|-----------------------|-----------------------|
| **Joint Model**    | $ \pm $      | $ \pm $      | $ \pm $      |
| **Model A**    | $ \pm $      | $35.27 \pm 34.95$      | $36.61 \pm 1.33$      |
| **Model B**    | $ \pm $      | $35.95 \pm 0.88$      | $34.00 \pm 1.59$      |
| **AVG**        | $ \pm $      | $27.06 \pm 2.32$      | $9.31 \pm 1.22$       |
| **OT**         | $ \pm $      | $27.78 \pm 1.78$      | $9.31 \pm 1.22$       |
| **MPF (ours)** | **$ \pm $**  | **$48.93 \pm 0.43$**  | **$49.21 \pm 0.44$**  |

### Requirements

The main dependencies for running the code are
* pytorch
* torchvision
* tqdm
* PIL
* numpy
* Python Optimal Transport (POT)
* tensorboard (from tensorflow to check logs)


## Running Experiments

Next, we provide detailed instructions on running each experiment.

In general, each experiment has a bash file in `scripts` directory along 
with the hyperparameters and random seeds used in the experiment. 
Corresponding command in the relevant bash file needs to be uncommented before running the experiment.
For most of the code, commands and argument names are self-explanatory.

### Training and Merging Models

First, all of the base models for fusion experiments need to be trained.
Their hyperparameters are all located in the CurveConfig class in `src/curve_merging.py`.  
The code for the model classes is in `src/models/fcmodel.py` and `src/models/mlpnet.py`.

Running training and fusion:
1. Check the CurveConfig class in `src/curve_merging.py` to modify the parameters and hyperparameters as you wish: model used, dataset, etc.  
2. Then run `src/train.py`. This script trains the base models and then merges following the AVG, OT and MPF procedures detailed in the paper.
3. The results of the trained models are located in `checkpoints/seed_{seed}/<model_{A or B}>/final_model.pth`.
4. If you already have the checkpoints of the base models and just want to merge and saved the fusion model, use `bash scripts/run_fuse_fc_models.sh`. 
To choose the type of fusion, just change the `fusion_type` variable with one of `"ot", "avg", "curve"`.
`input_dim` should be 3072 for CIFAR-10 and 784 for MNIST.
`hidden_dims` should be `400, 200, 100` for MLPNet, `800, 400, 200` for MLPLarge and `1024, 512, 256` for MLPHuge.
`model_path_list` should contain pairs of strings `{model_architecture}, model_checkpoint_path` that represent each model you want to merge. 
5. The statistics for the experiment are dumped in the tensorboard, the `model_accuracies.csv` file, and the terminal. 
Access the tensorboard using `tensorboard --logdir <logdir> --port 6006`

The model with best validation accuracy is saved as `best_val_acc_model.pth`, 
while the final model at the end of training epoch is saved as `final_model.pth`.

NOTE: We use the _final_ model for our experiments.
All the required model training and merging can be done using this script.

### Performing Continual Learning

For CL the notebook setup in `scripts` directory already has the experiments with cells executed. 
