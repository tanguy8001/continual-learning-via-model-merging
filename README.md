#  Model Merging for Continual Learning

We provide code for all the experiments presented in our paper.

The organization of code is as follows :
* Source code is present in `src` directory.
* Bash files required to run some experiments along with all the hyperparameters used are in the `scripts` directory.

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

### Training Models

First, all of the base models for fusion experiments need to be trained.
Their hyperparameters are all located in the CurveConfig class in `src/curve_merging.py`.  
The code for the model classes is in `src/models/fcmodel.py` and `src/models/mlpnet.py`.

Running training:
1. Check the CurveConfig class in `src/curve_merging.py` to modify the parameters and hyperparameters as you wish: model used, dataset, etc.  
2. Then run `src/train.py`
3. The results of the trained models would be present in `result/<experimenet_name>/<model_name>_<dataset_name>/<run_id>`, 
where `<run_id>` is a string consisting of relevant parameters used for this training like random seed etc.
4. Run `bash bin/run_train_models.sh`
5. The statistics for the experiment is dumped in tensorboard. 
Access the tensorboard using `tensorboard --logdir <logdir> --port 6006`

The trained models are saved in following dir `result/<experiment_name>/<model_name>_<dataset_name>/<run_id>/snapshots/`.
The model with best validation accuracy is saved as `best_val_acc_model.pth`, 
while the final model at the end of training epoch is saved as `final_model.pth`.

NOTE: We use the model with best validation accuracy for our experiments.
All the required model training can be done using this script.


### Fusing FC models with the same architecture

The relevant code for fusion is in `fuse_models.py`, `tlp_fusion.py`, `ad_hoc_ot_fusion.py` and `avg_fusion.py`.

Some arguments worth noting for using `fuse_models.py` are 
* `fusion_type`: The type of fusion to perfom, takes values in `tlp` for TLp fusion, 
`ot` for OT fusion, `avg` for vanilla averaging.
* `tlp_*` are the parameters for TLp fusion, `ad_hoc_ot_*` are the parameters for OT fusion
* `tlp_cost_choice`, `ad_hoc_cost_choice` denote the type of cost functions to be used for the fusion
* `tlp_ot_solver`, `ad_hoc_ot_solver` specify the type of solver to use for solving OT optimization problem

Running fusion of FC (fully connected) models with same architecture
1. Check `bin/run_fuse_fc_models.sh`
2. TLp fusion, OT fusion, Vanilla averaging have been documented in comments. 
Uncomment specific command to run it.
3. The number of models being fused can be changed 
by simply commenting the extra model paths being used for fusion. 
In case a different experiment name is being used for training the models, 
the `base_model_prefix` should be modified accordingly. 
4. The fused model is saved as `result/<experiment_name>/<model_name>_<dataset_name>/<run_id>/snapshots/fused_model.pth`
5. Note that the `<run_id>` contains all the relevant arguments as a string to identify fusion using 
a specific set of parameters.
6. After uncommenting and choosing the correct parameters run `bash bin/run_fuse_fc_models.sh`
7. Note down the validation and test accuracy of the fused model at the end of the fusion.

### Single shot distillation 

The single shot distillation experiments can be run using `bin/run_distill_models.sh`:
1. The second part of `bin/run_distill_models.sh` under `################ DISTILLATION INTO DIFFERENT ARCHITECTURE #################` 
heading contains the commands for performing distillation.
2. Note that distillation experiments are carried out for different seeds which are 
written as a for-loop in the script.
3. The entire script runs distillation for each seed and each choice of ot solver.
4. Uncomment appropriate sections to run distillation using TLp method or OT method.
5. Run using `bash bin/run_distill_models.sh`
6. The fused model is saved as `result/<experiment_name>/<model_name>_<dataset_name>/<run_id>/snapshots/fused_model.pth`
7. Note down the validation and test accuracy of the fused model at the end of each fusion.


### Fusion into different architecture

The fusion of models into different architecture is run using `bin/run_distill_models.sh`:
1. The first part of `bin/run_distill_models.sh` under `################## FUSION INTO DIFFERENT ARCHITECTURE ###################`
heading contains the commands for performing the relevant fusion.
2. The commands are TLp fusion and OT fusion can be uncommented separately to run the experiments.
3. Run `bash bin/run_distill_models.sh` for the appropriate command.
4. The fused model is saved as `result/<experiment_name>/<model_name>_<dataset_name>/<run_id>/snapshots/fused_model.pth`
5. Note down the validation and test accuracy of the fused model at the end of each fusion.


### TLp fusion for heterogeneous data distributions

The models can be trained on heterogeneous data distributions using the `bin/run_train_models.sh`.

After the models are trained the fusion can be performed using following:
1. Check `bin/run_fuse_hetero_models.sh` 
2. Note that fusion is performed for models trained from the same seed but different data distributions.
3. The weights for different models can be adjusted using `WA` and `WB` variables.
4. Model A is the model trained to recognize special digit, while Model B is the model trained for other digits.
5. Uncomment appropriate sections and run `bash bin/run_fuse_hetero_models.sh`
6. The fused model is saved as `result/<experiment_name>/<model_name>_<dataset_name>/<run_id>/snapshots/fused_model.pth`
7. Note down the validation and test accuracy of the fused model at the end of each fusion.


### Fusing deep CNNs

The deep CNN models - VGG11, ResNet18 can be trained using `bin/run_train_models.sh`.

To perform fusion run following:
1. Fusion for VGG11 models can be done using `bin/run_fuse_vgg_models.sh` 
2. For ResNet18 models, the fusion can be done using `bin/run_fuse_resnet_models.sh`
3. Uncomment the command for specific fusion type and run the scripts.
4. As usual the fused model is saved as `result/<experiment_name>/<model_name>_<dataset_name>/<run_id>/snapshots/fused_model.pth`
5. Note down the validation and test accuracy of the fused model at the end of each fusion.


For finetuning the fused and base models run following:
1. Check `bin/run_finetune_models.sh` and uncomment the appropriate section.
2. For both VGG11 and ResNet18 models, uncomment the checkpoint path corresponding 
to the finetuning and change the same in `--checkpoint_path` argument.
3. Run `bash bin/run_finetune_models.sh` for finishing finetuning and note down the accuracies.
4. Repeat for various checkpoint_paths.
