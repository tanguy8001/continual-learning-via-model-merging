import logging
import torch

class AvgFusion:
    def __init__(self, args, base_models, target_model):
        self.args = args
        self.base_models = base_models
        self.target_model = target_model

    def fuse(self):
        logging.info('Starting model fusion')
        # Fuse the parameters
        for i in range(1, self.target_model.num_layers + 1):
            self.fuse_single_layer(i)

    def fuse_single_layer(self, layer):
        avg_weight = None
        with torch.no_grad():
            for model in self.base_models:
                if avg_weight is None:
                    avg_weight = model.get_layer_weights(layer_num=layer)
                else:
                    avg_weight += model.get_layer_weights(layer_num=layer)
            avg_weight /= len(self.base_models)

        target_weights = self.target_model.get_layer_weights(layer_num=layer)
        target_weights.data = avg_weight.data
