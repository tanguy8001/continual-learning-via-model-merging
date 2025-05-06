import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.stateless import functional_call
from collections import OrderedDict
import wandb
import os # Added for saving
from models import fcmodel # Added for clarity, though functional_call might not strictly need it

class CurveMLP(Module):
    def __init__(self, num_params, bias = True, hidden_dim= 32):
        super().__init__()  # Call parent's __init__ first
        self.hidden_dim = hidden_dim
        self.num_params = num_params

        # The MLP takes only t as input now
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=bias),  # Input is scalar t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params, bias=bias) # Output is correction vector
        )

    def forward(self,
                t: float,
                w0: torch.Tensor,   # shape (num_params,)
                w1: torch.Tensor    # shape (num_params,)
               ) -> torch.Tensor:
        """
        Returns:
          w_interp: torch.Tensor of shape (num_params,)
                     = (1−t)*w0 + t*w1 + t*(1−t)*MLP(t)
        """
        # Ensure t is on the same device and has the right shape for MLP input
        # Assume w0 and w1 are already on the correct device
        t_tensor = torch.as_tensor(t, dtype=w0.dtype, device=w0.device).view(1, 1)

        # w0_flat = w0.view(-1) # No need to view if already flat
        # w1_flat = w1.view(-1)
        lin = (1.0 - t) * w0 + t * w1        # shape (num_params,)

        # MLP takes only t as input
        mlp_out = self.mlp(t_tensor).view(-1) # Ensure output is flat (num_params,)

        corr = mlp_out * (t * (1.0 - t))  # shape (num_params,)
        return lin + corr

    def get_model_weights(self, model: Module):
        sd = model.state_dict()
        all_weights = torch.cat([
            v.view(-1)
            for k, v in sd.items()
            if "weight" in k
        ])
        return all_weights

    def fit(self, train_loader, test_loader, config, model1: Module, model2: Module):
        """
        Trains the CurveMLP.

        Args:
            train_loader: DataLoader for the training set (fused).
            test_loader: DataLoader for the test set.
            config (dict): Dictionary containing training configuration:
                'epochs', 'learning_rate', 'momentum', 'weight_decay',
                'save_path', 'dataset', 'input_dim', 'hidden_dims', 'output_dim',
                'batch_size'.
            model1 (Module): The first base model (e.g., model_A). Assumed to be on the target device.
            model2 (Module): The second base model (e.g., model_B). Assumed to be on the target device.
        """
        # Determine device from model1 parameters
        device = next(model1.parameters()).device
        self.to(device)
        print(f"CurveMLP training on device: {device}")

        optimizer = optim.SGD(
            self.mlp.parameters(),           # Update only the MLP parameters
            lr=config['learning_rate'],      # Use dictionary access
            momentum=config['momentum'],     # Use dictionary access
            weight_decay=config['weight_decay'] # Use dictionary access
        )
        criterion = nn.CrossEntropyLoss()

        # Define interpolation points (can be made configurable if needed)
        interpolation_points = torch.tensor([0.5], device=device) # Example: Midpoint

        # Get flat parameter vectors (assuming models are already on the correct device)
        with torch.no_grad():
            flat1 = parameters_to_vector(model1.parameters()).detach()
            flat2 = parameters_to_vector(model2.parameters()).detach()
        specs = [(n, p.numel()) for n, p in model1.named_parameters()]

        # Use model1 as the prototype for functional_call
        # Requires model1 config ('input_dim', 'hidden_dims', 'output_dim') from main script config
        # We pass model1 itself which functional_call uses for architecture/buffers
        prototype_model = model1

        # --- WandB Setup ---
        # Filter config for logging to avoid logging large lists/complex objects if any
        wandb_config = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
        wandb_config.update({
             "optimizer": "SGD",
             "interpolation_points": interpolation_points.cpu().tolist(), # Log points used
             "num_params_mlp": sum(p.numel() for p in self.mlp.parameters()),
             "num_params_base": flat1.numel()
        })

        run = wandb.init(
            entity = "Continual_Learning-DAL", # TODO: Make configurable?
            project="Model Path Fusion for Continual Learning", # TODO: Make configurable?
            config=wandb_config # Log the filtered config
        )
        wandb.watch(self.mlp, log_freq=100) # Watch the MLP model parameters

        print("Starting CurveMLP training...")
        history = {'train_loss': [], 'train_acc': []} # Store history

        for epoch in range(config['epochs']): # Use dictionary access
            self.train() # Set CurveMLP to training mode
            prototype_model.train() # Ensure prototype is in train mode for functional_call if it matters (e.g., dropout)

            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            num_batches = len(train_loader)

            # Use tqdm for progress bar
            # from tqdm import tqdm # Consider adding import at top if tqdm is used
            # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

            # for batch_idx, (inputs, targets) in enumerate(progress_bar):
            for batch_idx, (inputs, targets) in enumerate(train_loader): # Original loop
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                batch_loss = 0.0
                # Consider drawing random t per batch or epoch if desired
                # current_t_points = torch.rand(1, device=device) # Example: Random t per batch
                current_t_points = interpolation_points # Use fixed points for now

                for t in current_t_points:
                    # Get the interpolated weights for the base model
                    w_interp = self(t, flat1, flat2)
                    # Build the state dict for the base model using interpolated weights
                    interp_state = build_state_dict(w_interp, specs, prototype_model)
                    # Perform a forward pass using the base model architecture but with interpolated weights
                    # functional_call handles non-parameter buffers correctly
                    outputs = functional_call(prototype_model, interp_state, (inputs,))
                    loss = criterion(outputs, targets)
                    batch_loss += loss # Accumulate loss for all t points in this batch

                    # Calculate accuracy (only for the last t point, or average?)
                    # Let's average accuracy across t points as well for consistency
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(targets).sum().item()
                    # total_samples is counted per t-point pass, adjust if needed

                total_samples += targets.size(0) * len(current_t_points) # Correct sample count

                # Average loss over interpolation points used in this batch
                batch_loss = batch_loss / len(current_t_points)

                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()

                # Update tqdm progress bar if used
                # progress_bar.set_postfix(loss=batch_loss.item())

                # Optional: Log batch loss less frequently
                # if batch_idx % 100 == 0:
                #     print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{num_batches}, Loss: {batch_loss.item():.4f}')

            avg_loss = total_loss / num_batches
            # Accuracy needs careful calculation depending on how samples/correct are counted
            # If total_samples is targets.size(0) * num_batches * len(t_points), then this is correct:
            accuracy = 100. * total_correct / total_samples

            history['train_loss'].append(avg_loss)
            history['train_acc'].append(accuracy)

            print(f'Epoch: {epoch+1}/{config["epochs"]}, Avg Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%')

            wandb.log({
                "epoch": epoch + 1,
                "train/average_loss": avg_loss,
                "train/accuracy": accuracy
            })

            # --- Optional: Evaluation Phase (on test_loader) ---
            # If evaluation per epoch is desired, add it here.
            # self.eval()
            # prototype_model.eval()
            # test_loss = 0.0
            # test_correct = 0
            # test_samples = 0
            # with torch.no_grad():
            #     for inputs, targets in test_loader:
            #         inputs, targets = inputs.to(device), targets.to(device)
            #         eval_batch_loss = 0.0
            #         eval_batch_correct = 0
            #         # Evaluate at specific t, e.g., t=0.5
            #         t_eval = torch.tensor([0.5], device=device)
            #         w_interp_eval = self(t_eval[0], flat1, flat2)
            #         interp_state_eval = build_state_dict(w_interp_eval, specs, prototype_model)
            #         outputs_eval = functional_call(prototype_model, interp_state_eval, (inputs,))
            #         loss_eval = criterion(outputs_eval, targets)
            #         eval_batch_loss += loss_eval.item()
            #
            #         _, predicted_eval = outputs_eval.max(1)
            #         eval_batch_correct += predicted_eval.eq(targets).sum().item()
            #
            #     test_loss += eval_batch_loss # Sum loss across batches
            #     test_correct += eval_batch_correct
            #     test_samples += targets.size(0)
            #
            # avg_test_loss = test_loss / len(test_loader)
            # test_accuracy = 100. * test_correct / test_samples
            # print(f'Epoch: {epoch+1}, Avg Test Loss: {avg_test_loss:.4f}, Test Accuracy (t=0.5): {test_accuracy:.2f}%')
            # wandb.log({
            #     "epoch": epoch + 1,
            #     "test/average_loss_t0.5": avg_test_loss,
            #     "test/accuracy_t0.5": test_accuracy
            # })


        # --- Save Final Checkpoint ---
        save_path = config.get('save_path') # Use .get for safety
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save CurveMLP state_dict and relevant config
            # Mirror the structure used in test_curve_merging.py save_checkpoint
            checkpoint = {
                'model_state_dict': self.state_dict(), # Save CurveMLP state
                'config': { # Store relevant config for reloading CurveMLP
                    'num_params': self.num_params,
                    'hidden_dim': self.hidden_dim,
                    'bias': True, # Assuming bias was True during init, make dynamic if needed
                    # Include base model config needed for evaluation later?
                    'base_model_config': {
                         'input_dim': config.get('input_dim'),
                         'hidden_dims': config.get('hidden_dims'),
                         'output_dim': config.get('output_dim')
                    },
                    # Store training config used for this MLP?
                    'training_config': {
                        'epochs': config['epochs'],
                        'learning_rate': config['learning_rate'],
                        'momentum': config['momentum'],
                        'weight_decay': config['weight_decay']
                    }
                }
            }
            torch.save(checkpoint, save_path)
            print(f"CurveMLP checkpoint saved to {save_path}")
            # Log saved artifact to wandb
            artifact = wandb.Artifact(f'curvemlp-{config["dataset"]}', type='model')
            artifact.add_file(save_path)
            run.log_artifact(artifact)

        wandb.finish()
        print("CurveMLP training finished.")
        return history # Return training history

# --- Helper Function ---
# (Keep outside class or make staticmethod if preferred)
def build_state_dict(w_flat: torch.Tensor,
                     specs: list[tuple[str,int]],
                     prototype_model: torch.nn.Module
                    ) -> OrderedDict:
    """
    Convert a flat parameter vector back into a OrderedDict matching prototype_model.

    Args:
      w_flat:    1D Tensor of length = sum(p.numel() for p in prototype_model.parameters())
      specs:     list of (name, numel) for each parameter in prototype_model.named_parameters()
      prototype_model: any nn.Module whose .state_dict() naming/param shapes you want to match

    Returns:
      OrderedDict{name → Tensor.view_as(original_param)}
    """
    state = OrderedDict()
    offset = 0
    # Get parameters from the prototype model to determine shapes
    named_params = dict(prototype_model.named_parameters())
    # Also handle buffers if necessary for functional_call
    named_buffers = dict(prototype_model.named_buffers())

    param_offset = 0
    for name, numel in specs:
        chunk = w_flat[param_offset : param_offset + numel]
        param_offset += numel
        if name in named_params:
             orig_param = named_params[name]
             state[name] = chunk.view_as(orig_param).clone() # Use clone to avoid modifying source tensor
        # else:
        #     # Handle cases where spec might include things not in named_parameters (unlikely)
        #     print(f"Warning: Spec name {name} not found in prototype model parameters.")

    # Add buffers needed by functional_call (they are not interpolated)
    for name, buf in named_buffers.items():
        if name not in state: # Only add if not already handled (e.g., if buffer names clash with params)
            state[name] = buf.clone() # Add buffers directly

    # Ensure all required keys for functional_call are present
    # This might involve adding buffers that weren't part of the parameter specs
    # for name, buf in prototype_model.named_buffers():
    #     if name not in state:
    #         state[name] = buf

    return state