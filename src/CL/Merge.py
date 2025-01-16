from OT.wasserstein_ensemble import geometric_ensembling_modularized
from curve_merging import curve_ensembling, CurveConfig

from models.mlpnet import MlpNetBase
from CL.Utils import copy_model
from CL.Train import train_model

import torch

def naive_merge(m1,m2,input_dim,num_classes,ratio=0.5):
    """ takes two models and returns navie average possibly with weights adjusted"""
    averaged_state = {}
    # Weight for the current task
    m1_weight = ratio
    # Weight for the previous tasks combined
    m2_weight = 1 - ratio 

    for (name1, param1), (name2, param2) in zip(
        m1.state_dict().items(), m2.state_dict().items()
    ):
        averaged_state[name1] = (param1 * m1_weight) + (param2 * m2_weight)

    # Create a new model that is a copy of m1
    new_model = copy_model(
        m1,
        input_dim,
        num_classes
    )  # Get same class as m1
    new_model.load_state_dict(averaged_state)
    return new_model


def ot_merge(m1, m2, input_dim, num_classes, ratio=0.5, data_loader=None):
    """Uses code of OT and merges two models according to OT"""
    class Args:
        """Dummy class to hold arguments for wasserstein ensembling"""
        def __init__(self):
            self.geom_ensemble_type = "wts"  # Use weights-based ensembling
            self.gpu_id = -1

            self.ensemble_step = 1-ratio  # Equal weighting between models
            self.exact = True  # Use exact Wasserstein distance
            self.reg = 0.01  # Regularization parameter for Sinkhorn
            self.skip_last_layer = False
            self.unbalanced = False
            self.past_correction = False
            self.eval_aligned = False
            self.debug = False
            self.ground_metric = "euclidean"
            self.ground_metric_normalize = "log"
            self.ground_metric_eff = True
            self.normalize_wts = True
            self.activation_histograms = True
            self.dist_normalize = True
            self.act_num_samples = 100
            self.clip_gm = True
            self.clip_max = 5
            self.clip_min = 0
            self.importance = None
            self.correction = True
            self.proper_marginals = True
            self.num_models = 2
            self.width_ratio = 1
            self.model_name = "mlpnet"
            self.num_hidden_nodes = 400
            self.num_hidden_nodes1 = 400
            self.num_hidden_nodes2 = 200
            self.num_hidden_nodes3 = 100
            self.disable_bias = True
            self.enable_dropout = False
            self.dataset = "mnist"
    args = Args()

    models = [m1, m2]

    acc,new_model = geometric_ensembling_modularized(args, models, data_loader)

    return new_model


def curve_merge(m1,m2,input_dim,num_classes,data_loader):
    """Uses code of curve merging and merges two models according to curve merging"""
    config = CurveConfig()
    config.input_dim = input_dim
    config.hidden_dims = [400, 200, 100]
    models = [m1, m2]
    new_model = MlpNetBase(input_dim, num_classes)
    new_model = curve_ensembling(
        config, models, new_model, data_loader, data_loader, "cpu", num_classes, input_dim
    )
    return new_model



def seq_merge(merge, models, input_dim, num_classes, test_loader=None):
    """function to take merging scheme,list of models and merges them accordingly maybe with list of test_loaders. Is optional"""
    final_model = copy_model(models[0], input_dim, num_classes)

    for i, model in enumerate(models[1:]):
        ratio = (i + 1) / (i + 2)
        if test_loader != None:
            final_model = merge(
                final_model, model, input_dim, num_classes, ratio, test_loader
            )
        else:
            final_model = merge(final_model, model, input_dim, num_classes, ratio)

    return final_model


def seq_merge_finetune(merge, model,seq_data, input_dim, num_classes,test_loader=None):
    """function to take merging scheme,list of models and merges them accordingly with new"""
    final_model = copy_model(model, input_dim, num_classes)
    seq_data.current_task = 0
    for i in range(0,seq_data.n_tasks):
        ratio = (i ) / (i + 1)

        train_loader, _ = seq_data.get_task_data()


        if i == 0:
            final_model = copy_model(model, input_dim, num_classes)
            final_model = train_model(final_model, train_loader, num_classes)

        else:
            new_model = copy_model(final_model, input_dim, num_classes)
            new_model = train_model(new_model, train_loader, num_classes)
            if test_loader != None:
                final_model = merge(
                    final_model, new_model, input_dim, num_classes, ratio, test_loader
                )
            else:
                final_model = merge(final_model, new_model, input_dim, num_classes, ratio)

        seq_data.current_task += 1

    return final_model


def seq_merge_curve(
    curve_merge, models, input_dim, num_classes, seq_data, replay_size=500
):
    """function to take list of models and merges them using curve ensembling"""
    seq_data.current_task = 0
    final_model = copy_model(models[0], input_dim=input_dim, num_classes=num_classes)
    train_data, _ = seq_data.get_task_data()

    train_data = train_data.dataset
    # only take subset of 10% of the data
    indices = torch.randperm(len(train_data))[:replay_size]
    train_data = torch.utils.data.Subset(train_data, indices)
    for i, model in enumerate(models[1:]):
        # take the data for the epoch and add the train loader

        seq_data.current_task = i + 1
        new_train_data, _ = seq_data.get_task_data()
        new_data = new_train_data.dataset
        indices = torch.randperm(len(new_data))[:replay_size]
        new_data = torch.utils.data.Subset(new_data, indices)

        # combine the dataloaders
        train_data = torch.utils.data.ConcatDataset([train_data, new_data])
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=32, shuffle=True
        )

        final_model = curve_merge(
            final_model,
            model,
            input_dim=input_dim,
            num_classes=num_classes,
            data_loader=train_loader,
        )

    return final_model


def seq_merge_ft(
    merge, models, input_dim, num_classes, seq_data, replay_size=500,test_loader=None
):
    """function to take list of models and merges them using merge functoin and then finetune them afterwards on a replay buffer"""
    seq_data.current_task = 0
    final_model = copy_model(models[0], input_dim=input_dim, num_classes=num_classes)
    train_data, test_data = seq_data.get_task_data()

    train_data = train_data.dataset
    # only take subset of 10% of the data
    indices = torch.randperm(len(train_data))[:replay_size]
    train_data = torch.utils.data.Subset(train_data, indices)
    for i, model in enumerate(models[1:]):
        # take the data for the epoch and add the train loader
        ratio = (i + 1) / (i + 2)


        seq_data.current_task = i + 1
        new_train_data, _ = seq_data.get_task_data()
        new_data = new_train_data.dataset
        indices = torch.randperm(len(new_data))[:replay_size]
        new_data = torch.utils.data.Subset(new_data, indices)

        # combine the dataloaders
        train_data = torch.utils.data.ConcatDataset([train_data, new_data])
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=32, shuffle=True
        )

        if test_loader != None:
            final_model = merge(
                final_model, model, input_dim, num_classes, ratio, test_data
            )
        else:
            final_model = merge(final_model, model, input_dim, num_classes, ratio)
        
        final_model = train_model(final_model, train_loader, num_classes)

    return final_model


def mag_max(
        model,seq_data, input_dim, num_classes,lam,test_loader=None):
    """function to take merging scheme,list of models and merges them accordingly maybe with list of test_loaders. Is optional"""
    final_model = copy_model(model, input_dim, num_classes)
    seq_data.current_task = 0

    #make first taskvector state of the model with 0 in every entry
    running_task_vector = final_model.state_dict().copy()
    for name, param in running_task_vector.items():
        running_task_vector[name] = torch.zeros_like(param)

    task_vectors = []

    seq_data.current_task = 0
    for i in range(0, seq_data.n_tasks):
        

        train_loader, _ = seq_data.get_task_data()

        if i == 0:
            final_model = copy_model(model, input_dim, num_classes)
            final_model = train_model(final_model, train_loader, num_classes)

        else:
            new_model = copy_model(final_model, input_dim, num_classes)
            new_model = train_model(new_model, train_loader, num_classes)
            
            current_task_vector = model.state_dict().copy()
            for name, param in current_task_vector.items():
                current_task_vector[name] = param - model.state_dict()[name].clone()
            
            #update running task vector as follows. Change only the entries where the absolute value of the task_vector is bigger than the absolute value of the running_task_vector
            for name, param in current_task_vector.items():
                running_task_vector[name] = torch.where(torch.abs(param) > torch.abs(running_task_vector[name]),param,running_task_vector[name])
            
            final_model = copy_model(model, input_dim, num_classes)

            for name, param in final_model.state_dict().items():
                final_model.state_dict()[name] = model.state_dict()[name] + lam * running_task_vector[name]

            

        seq_data.current_task += 1

    return final_model
