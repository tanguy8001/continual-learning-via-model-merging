import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import yaml # Add YAML import
from models import fcmodel
# from curves_MLP import CurveMLP, build_state_dict # Import necessary components - Check if this import path is correct
try:
    from curves_MLP import CurveMLP, build_state_dict
except ImportError:
    # Assuming curves_MLP.py is in the same directory (src)
    from .curves_MLP import CurveMLP, build_state_dict

# Imports for CurveNet (from curve_merging/plane.py context)
import curves
from models import fcmodel as fcmodel_curve # Avoid name clash if different base classes
from curves import CurveNet

from torch.nn.utils import parameters_to_vector
from torch.nn.utils.stateless import functional_call
from data import double_loaders, create_fused_loader


def normalize_direction(direction, w_ref):
    """Normalizes a direction vector relative to a reference weight vector."""
    # Avoid division by zero if direction or w_ref is zero vector
    norm_direction = torch.norm(direction)
    norm_w_ref = torch.norm(w_ref)
    if norm_direction == 0 or norm_w_ref == 0:
        return direction # Or handle as an error/warning
    return direction / norm_direction * norm_w_ref

def plot_loss_surface_mlp(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_ids else "cpu") # Use gpu_ids arg

    data_loaders, num_classes = double_loaders(
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_name=args.transform,
        digit=args.mnist_digit,
        cifar_class=args.cifar_class
    )
    testloader = data_loaders['test']

    # --- 1. Load Models and Weights ---
    print("Loading models...")
    # Load model_A state_dict
    # Use weights_only=True for security if possible, requires saving differently or trusting the source.
    # For now, keeping False based on user's log, but added warning suppression.
    import warnings
    warnings.filterwarnings("ignore", message=".*weights_only=False.*")
    state_dict_A = torch.load(args.model_a_path, map_location=device)

    # Determine config for model A
    if 'config' in state_dict_A:
        config_A = state_dict_A['config']
    else:
        print("Warning: 'config' key not found in model_A checkpoint. Using command-line arguments for architecture.")
        config_A = {
            'input_dim': args.input_dim,
            'hidden_dims': args.hidden_dims,
            'output_dim': args.output_dim
        }
        # Check if the loaded state dict is just the model state
        if 'model_state_dict' not in state_dict_A:
             # Assume the entire loaded object is the state_dict
            model_state_dict_A = state_dict_A
        else:
             model_state_dict_A = state_dict_A['model_state_dict']


    model_A = fcmodel.FCModelBase(**config_A) # Use derived config
    # Load state dict - handle case where it might be the root object
    if 'model_state_dict' in state_dict_A:
        model_A.load_state_dict(state_dict_A['model_state_dict'])
    else:
         model_A.load_state_dict(model_state_dict_A) # Use the separated state dict
    model_A.to(device).eval()
    w0 = parameters_to_vector(model_A.parameters()).detach()

    # Load model_B state_dict
    state_dict_B = torch.load(args.model_b_path, map_location=device)
     # Determine config for model B (assuming same architecture if not provided)
    if 'config' in state_dict_B:
        config_B = state_dict_B['config']
    else:
        print("Warning: 'config' key not found in model_B checkpoint. Using command-line arguments for architecture.")
        config_B = config_A # Assume same architecture as A if B's config is missing
        # Check if the loaded state dict is just the model state
        if 'model_state_dict' not in state_dict_B:
             # Assume the entire loaded object is the state_dict
            model_state_dict_B = state_dict_B
        else:
             model_state_dict_B = state_dict_B['model_state_dict']


    model_B = fcmodel.FCModelBase(**config_B) # Use derived config
    # Load state dict - handle case where it might be the root object
    if 'model_state_dict' in state_dict_B:
        model_B.load_state_dict(state_dict_B['model_state_dict'])
    else:
        model_B.load_state_dict(model_state_dict_B) # Use the separated state dict

    model_B.to(device).eval()
    w1 = parameters_to_vector(model_B.parameters()).detach()

    print("Loading CurveMLP...")
    state_dict_curve = torch.load(args.curve_mlp_path, map_location=device)
    num_params = w0.numel()

    # Determine config for CurveMLP
    curve_hidden_dim = 32 # Default
    if isinstance(state_dict_curve, dict) and 'config' in state_dict_curve and 'hidden_dim' in state_dict_curve['config']:
        curve_hidden_dim = state_dict_curve['config']['hidden_dim']
    elif isinstance(state_dict_curve, dict) and 'hidden_dim' in state_dict_curve: # Check if hidden_dim is top-level key
         curve_hidden_dim = state_dict_curve['hidden_dim']
    else:
         print(f"Warning: Could not find 'hidden_dim' in CurveMLP checkpoint config. Using default: {curve_hidden_dim}")


    curve_mlp = CurveMLP(num_params=num_params, hidden_dim=curve_hidden_dim)
    # Load state dict - handle case where it might be the root object
    if isinstance(state_dict_curve, dict) and 'model_state_dict' in state_dict_curve:
        curve_mlp.load_state_dict(state_dict_curve['model_state_dict'])
    elif isinstance(state_dict_curve, dict):
         # Maybe the dict itself is the state_dict (if no 'model_state_dict' key)
         try:
             curve_mlp.load_state_dict(state_dict_curve)
             print("Loaded CurveMLP state_dict directly from checkpoint object.")
         except RuntimeError as e:
             print(f"Error loading CurveMLP state_dict directly: {e}")
             print("Please ensure the CurveMLP checkpoint contains the model state under the key 'model_state_dict' or is the state_dict itself.")
             raise # Re-raise the error after printing info
    else:
         # If it's not a dict, assume it's the state_dict itself
         try:
            curve_mlp.load_state_dict(state_dict_curve)
            print("Loaded CurveMLP state_dict directly from checkpoint object (non-dict).")
         except Exception as e:
             print(f"Error loading CurveMLP state_dict from non-dict object: {e}")
             raise # Re-raise the error

    curve_mlp.to(device).eval()

    # --- Load Optional CurveNet ---
    curve_net_model = None
    if args.curve_net_ckpt and CurveNet: # Check if path provided and import succeeded
        print("Loading CurveNet...")
        try:
            # Determine architecture for CurveNet (use model_A's config as default)
            curve_net_config = config_A
            container_name = args.curve_net_base_model.replace('Base', '') # Simple approach: FCModelBase -> FCModel
            try:
                model_container = getattr(fcmodel_curve, container_name) # e.g., fcmodel_curve.FCModel
            except AttributeError:
                 # Fallback or better error handling: maybe the arg IS the container name already
                 print(f"Warning: Could not find container '{container_name}'. Trying '{args.curve_net_base_model}' directly.")
                 try:
                     model_container = getattr(fcmodel_curve, args.curve_net_base_model)
                 except AttributeError:
                     raise AttributeError(f"Could not find model container class '{container_name}' or '{args.curve_net_base_model}' in models.fcmodel")

            # Ensure the retrieved object has a 'curve' attribute
            if not hasattr(model_container, 'curve'):
                raise AttributeError(f"Model container '{model_container.__name__}' does not have a 'curve' attribute.")
            curve_net_arch_func = model_container.curve

            curve_def = getattr(curves, args.curve_type)

            curve_net_model = CurveNet(
                num_classes=curve_net_config['output_dim'], # Assuming output_dim is num_classes
                curve=curve_def,
                architecture=curve_net_arch_func, # Pass the curve architecture function/class
                num_bends=args.curve_net_num_bends,
                architecture_kwargs=curve_net_config, # Pass the base model config
            )
            curve_net_model.to(device)
            checkpoint = torch.load(args.curve_net_ckpt, map_location=device)
            # Check for 'model_state_dict' key in checkpoint
            if 'model_state_dict' in checkpoint:
                curve_net_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                 # Assume the checkpoint itself is the state dict
                print("Warning: 'model_state_dict' key not found in CurveNet checkpoint. Attempting to load the root object as state_dict.")
                curve_net_model.load_state_dict(checkpoint)
            curve_net_model.eval()
            print("CurveNet loaded successfully.")
        except Exception as e:
            print(f"Error loading CurveNet from {args.curve_net_ckpt}: {e}")
            print("Continuing without CurveNet plot.")
            curve_net_model = None # Ensure it's None if loading failed
    elif args.curve_net_ckpt and not CurveNet:
        print("CurveNet checkpoint provided, but necessary modules could not be imported. Skipping CurveNet plot.")


    # --- 2. Define Plane Directions ---
    print("Defining plane...")
    t_mid = torch.tensor(0.5, device=device)
    # Ensure w0 and w1 are passed correctly if using a class method
    # If curve_mlp is just a function, this is fine. If it's a module, ensure forward is correct.
    try:
        w_mid = curve_mlp(t_mid, w0, w1).detach()
    except TypeError:
         # Maybe the CurveMLP expects only 't'? Adapt as needed.
         # This depends heavily on the CurveMLP implementation.
         # Assuming it takes t, w0, w1 based on previous context.
         print("Error calling curve_mlp. Ensure its forward method signature is correct.")
         # Example alternative if it only takes t:
         # w_mid = curve_mlp(t_mid).detach() # This would require curve_mlp to internally know w0, w1
         raise # Re-raise the error after suggestion


    u_direction = w1 - w0
    v_direction_raw = w_mid - (0.5 * w0 + 0.5 * w1)

    # Orthogonalize v w.r.t. u (Gram-Schmidt)
    dot_uv_raw = torch.dot(u_direction, v_direction_raw)
    dot_uu = torch.dot(u_direction, u_direction)
    # Avoid division by zero if u_direction is zero vector
    if dot_uu == 0:
        print("Warning: u_direction (w1 - w0) is zero vector. Cannot orthogonalize.")
        v_direction = v_direction_raw # Or handle differently
    else:
        v_direction = v_direction_raw - (dot_uv_raw / dot_uu) * u_direction

    # Normalize directions (optional but often helpful for consistent scaling)
    u_direction = normalize_direction(u_direction, w0) # Scale relative to w0 norm
    v_direction = normalize_direction(v_direction, w0)

    # --- 3. Setup Grid and Dataloader ---
    print("Setting up grid and data...")
    # Define grid range and resolution
    X = np.linspace(args.xmin, args.xmax, args.resolution)
    Y = np.linspace(args.ymin, args.ymax, args.resolution)
    loss_surface = np.empty((args.resolution, args.resolution))

    criterion = torch.nn.CrossEntropyLoss()

    # Create a base model instance for evaluation using the derived config
    eval_model = fcmodel.FCModelBase(**config_A).to(device)
    # Get parameter specs for build_state_dict
    # Ensure eval_model is correctly initialized before getting named_parameters
    try:
        specs = [(n, p.numel()) for n, p in eval_model.named_parameters()] # Use numel for build_state_dict
    except Exception as e:
        print(f"Error getting specs from eval_model: {e}")
        raise

    # --- 4. Evaluate Loss on Grid ---
    print("Evaluating loss surface...")
    eval_model.eval()
    with torch.no_grad():
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                # Calculate weight vector for this grid point
                # Ensure u_direction and v_direction have compatible dimensions with w_mid
                try:
                    w = w_mid + x * u_direction + y * v_direction
                except RuntimeError as e:
                    print(f"Error calculating w at grid point ({i},{j}): {e}")
                    print(f"w_mid shape: {w_mid.shape}, u_direction shape: {u_direction.shape}, v_direction shape: {v_direction.shape}")
                    raise


                # Build state dict from flat vector
                try:
                    # build_state_dict might need shapes, not numel. Adjust if necessary based on its implementation.
                    current_state = build_state_dict(w, specs, eval_model) # Pass eval_model instance
                except Exception as e:
                     print(f"Error calling build_state_dict: {e}")
                     print(f"w shape: {w.shape}")
                     # print(f"specs: {specs}") # Can be very long
                     raise


                # Evaluate loss using functional_call
                total_loss = 0.0
                total_samples = 0
                batches_processed = 0 # Keep track of batches used for this grid point
                max_batches_per_point = 5 # Define the limit

                # Loop through batches, but stop after max_batches_per_point
                for k, (inputs, targets) in enumerate(testloader):
                    # --- Check if we have processed enough batches --- 
                    if batches_processed >= max_batches_per_point:
                        break # Stop processing batches for this grid point
                    # --------------------------------------------------

                    inputs, targets = inputs.to(device), targets.to(device)

                    # --- Manually flatten the input for functional_call ---
                    original_shape = inputs.shape
                    inputs_flat = inputs.view(inputs.size(0), -1)
                    # --- ----------------------------------------------- ---

                    # Use functional_call for efficiency
                    try:
                        # Check the FLATTENED input shape
                        if inputs_flat.shape[1] != config_A['input_dim']:
                             print(f"Flattened input shape mismatch in batch {k}: Expected {config_A['input_dim']}, Got {inputs_flat.shape[1]}. Original shape: {original_shape}")
                             continue # Skip batch or handle error

                        # Pass the FLATTENED input to functional_call
                        outputs = functional_call(eval_model, current_state, (inputs_flat,))
                        loss = criterion(outputs, targets)
                        total_loss += loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)
                    except Exception as e:
                         print(f"Error during functional_call or loss calculation in batch {k}: {e}")
                         print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}, Output shape: {outputs.shape if 'outputs' in locals() else 'N/A'}")
                         raise

                    batches_processed += 1 # Increment the counter

                if total_samples == 0:
                     loss_surface[j, i] = float('nan') # Or some indicator of no data
                     print(f"Warning: No samples processed for grid point ({i},{j})")
                else:
                    loss_surface[j, i] = total_loss / total_samples # Store average loss
            print(f"Column {i+1}/{args.resolution} evaluated.")


    # --- 5. Calculate and Project Curve ---
    print("Projecting CurveMLP path...")
    ts = torch.linspace(0, 1, 100, device=device)
    curve_mlp_points_x = []
    curve_mlp_points_y = []
    # Pre-calculate norms and dot products for efficiency and safety
    norm_u_sq = torch.dot(u_direction, u_direction)
    norm_v_sq = torch.dot(v_direction, v_direction)
    norm_u = torch.sqrt(norm_u_sq) if norm_u_sq > 0 else 0
    norm_v = torch.sqrt(norm_v_sq) if norm_v_sq > 0 else 0
    norm_w0 = torch.norm(w0)

    if norm_u_sq == 0 or norm_v_sq == 0 or norm_w0 == 0:
        print("Warning: A direction or reference vector norm is zero. Projection scaling might be inaccurate.")


    with torch.no_grad():
        for t in ts:
            # Ensure curve_mlp call is correct
            try:
                 w_t = curve_mlp(t, w0, w1)
            except TypeError:
                 print("Error calling curve_mlp in projection loop. Ensure its forward method signature is correct.")
                 # Example alternative: w_t = curve_mlp(t)
                 raise
            delta = w_t - w_mid
            x_proj = torch.dot(delta, u_direction) / norm_u_sq if norm_u_sq > 0 else 0
            y_proj = torch.dot(delta, v_direction) / norm_v_sq if norm_v_sq > 0 else 0

            # Apply scaling based on normalization factor used earlier (norm_w0)
            x_scaled = x_proj * norm_u / norm_w0 if norm_w0 > 0 else 0
            y_scaled = y_proj * norm_v / norm_w0 if norm_w0 > 0 else 0

            curve_mlp_points_x.append(x_scaled.item())
            curve_mlp_points_y.append(y_scaled.item())


    # Project w0, w1, w_mid as well
    delta0 = w0 - w_mid
    x0 = torch.dot(delta0, u_direction) / norm_u_sq if norm_u_sq > 0 else 0
    y0 = torch.dot(delta0, v_direction) / norm_v_sq if norm_v_sq > 0 else 0
    x0_scaled = x0 * norm_u / norm_w0 if norm_w0 > 0 else 0
    y0_scaled = y0 * norm_v / norm_w0 if norm_w0 > 0 else 0

    delta1 = w1 - w_mid
    x1 = torch.dot(delta1, u_direction) / norm_u_sq if norm_u_sq > 0 else 0
    y1 = torch.dot(delta1, v_direction) / norm_v_sq if norm_v_sq > 0 else 0
    x1_scaled = x1 * norm_u / norm_w0 if norm_w0 > 0 else 0
    y1_scaled = y1 * norm_v / norm_w0 if norm_w0 > 0 else 0


    # w_mid projects to (0, 0) by definition of the coordinate system
    xmid_scaled, ymid_scaled = 0.0, 0.0

    # --- 5b. Calculate and Project Optional CurveNet Path ---
    curve_net_points_x = []
    curve_net_points_y = []
    if curve_net_model:
        print("Projecting CurveNet path...")
        with torch.no_grad():
            for t in ts:
                # Call weights method, returns numpy array
                w_t_cn_np = curve_net_model.weights(torch.tensor([t], device=device))
                # Convert numpy array to torch tensor on the correct device
                w_t_cn = torch.from_numpy(w_t_cn_np).to(device)

                delta_cn = w_t_cn - w_mid # Project relative to the same origin w_mid
                x_proj_cn = torch.dot(delta_cn, u_direction) / norm_u_sq if norm_u_sq > 0 else 0
                y_proj_cn = torch.dot(delta_cn, v_direction) / norm_v_sq if norm_v_sq > 0 else 0

                # Apply the same scaling as used for CurveMLP and axes
                x_scaled_cn = x_proj_cn * norm_u / norm_w0 if norm_w0 > 0 else 0
                y_scaled_cn = y_proj_cn * norm_v / norm_w0 if norm_w0 > 0 else 0

                curve_net_points_x.append(x_scaled_cn.item())
                curve_net_points_y.append(y_scaled_cn.item())

        # --- Print CurveNet coordinate range for debugging ---
        if curve_net_points_x and curve_net_points_y:
            print(f"DEBUG: CurveNet X range: [{min(curve_net_points_x):.4f}, {max(curve_net_points_x):.4f}]")
            print(f"DEBUG: CurveNet Y range: [{min(curve_net_points_y):.4f}, {max(curve_net_points_y):.4f}]")
            print(f"DEBUG: Plot X limits: [{args.xmin}, {args.xmax}]")
            print(f"DEBUG: Plot Y limits: [{args.ymin}, {args.ymax}]")
        else:
            print("DEBUG: CurveNet coordinate lists are empty. Was projection successful?")
        # ----------------------------------------------------

    # --- 6. Plot ---
    print("Plotting...")
    plt.figure(figsize=(10, 8))

    # Filter out NaN values before plotting contours if any occurred
    valid_loss = loss_surface[~np.isnan(loss_surface)]
    if len(valid_loss) > 0:
         min_loss = np.min(valid_loss)
         max_loss = np.max(valid_loss)
         # Adjust levels for contour plot - avoid error if min=max
         levels = np.linspace(min_loss, max_loss, 20) if min_loss != max_loss else [min_loss]
         # Use contourf for filled contours, which might look better
         # CS = plt.contour(X, Y, loss_surface, levels=levels, cmap='viridis')
         CS = plt.contourf(X, Y, loss_surface, levels=levels, cmap='viridis', extend='both') # extend handles values outside range
         plt.colorbar(CS, label='Loss')
    else:
         print("Warning: Loss surface contains only NaN values. Skipping contour plot.")

    # Plot the projected curve
    plt.plot(curve_mlp_points_x, curve_mlp_points_y, marker='.', linestyle='-', color='red', label='CurveMLP Path')

    # Plot the projected CurveNet path if available
    if curve_net_points_x and curve_net_points_y: # Check if projection was successful
        print("Plotting CurveNet path...")
        plt.plot(curve_net_points_x, curve_net_points_y, marker='x', linestyle='--', color='purple', label='CurveNet Path')

    # Plot projected model points
    plt.scatter([x0_scaled], [y0_scaled], marker='o', s=100, color='blue', label='Model A (w0)')
    plt.scatter([x1_scaled], [y1_scaled], marker='o', s=100, color='green', label='Model B (w1)')
    plt.scatter([xmid_scaled], [ymid_scaled], marker='*', s=150, color='orange', label='Midpoint w(0.5)')

    plt.title('Loss Surface with CurveMLP Path')
    plt.xlabel(f'Direction 1 (w1 - w0) (scaled by {norm_u/norm_w0:.2f})')
    plt.ylabel(f'Direction 2 (Orthogonal Deviation) (scaled by {norm_v/norm_w0:.2f})')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    # Ensure plot limits cover the data range
    plt.xlim(args.xmin, args.xmax)
    plt.ylim(args.ymin, args.ymax)
    plt.gca().set_aspect('equal', adjustable='box') # Make axes scales equal if desired

    plt.savefig(args.output_plot_path)
    print(f"Plot saved to {args.output_plot_path}")
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Loss Surface for CurveMLP')

    # --- Add config file argument ---
    parser.add_argument('--config_path', type=str, default='src/config.yaml', help='Path to the YAML configuration file')
    # --- ----------------------- ---

    parser.add_argument('--model_a_path', type=str, required=True, help='Path to model A state dict (.pth)')
    parser.add_argument('--model_b_path', type=str, required=True, help='Path to model B state dict (.pth)')
    parser.add_argument('--curve_mlp_path', type=str, required=True, help='Path to trained CurveMLP state dict (.pth)')

    # Arguments needed if 'config' not in checkpoints
    parser.add_argument('--input_dim', type=int, default=784, help='Input dimension (required if config missing)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', help='Hidden layer dimensions (required if config missing)')
    parser.add_argument('--output_dim', type=int, default=10, help='Output dimension (required if config missing)')

    # Dataloader arguments (add if get_dataloaders_for_plotting is used)
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name (used for placeholder dataloader info)')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to data (if needed for dataloader)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')

    # Plotting arguments
    parser.add_argument('--xmin', type=float, default=-3.0, help='Min x range')
    parser.add_argument('--xmax', type=float, default=3.0, help='Max x range')
    parser.add_argument('--ymin', type=float, default=-3.2, help='Min y range')
    parser.add_argument('--ymax', type=float, default=3.2, help='Max y range')
    parser.add_argument('--resolution', type=int, default=20, help='Grid resolution')
    parser.add_argument('--output_plot_path', type=str, default='curvemlp_loss_surface.png', help='Output path for the plot')
    parser.add_argument('--gpu_ids', type=str, default=None, help='GPU ID(s) to use (e.g., "0" or "0,1"). None uses CPU.')

    # Optional arguments for plotting CurveNet path
    parser.add_argument('--curve_net_ckpt', type=str, default=None, help='Path to trained CurveNet state dict (.pth) (optional)')
    parser.add_argument('--curve_type', type=str, default='Bezier', help='Type of curve used for CurveNet (e.g., Bezier)')
    parser.add_argument('--curve_net_num_bends', type=int, default=3, help='Number of bends for CurveNet')
    parser.add_argument('--curve_net_base_model', type=str, # default='FCModelBase', # Default loaded from config if available
                        help='Base model class name used for CurveNet architecture (must exist in models.fcmodel)')

    # --- Argument for num_workers (can be loaded from config) ---
    parser.add_argument('--num_workers', type=int, default=None, # Default to None, will be loaded from config
                        help='Number of data loading workers (overrides config if set)')
    # --- ---------------------------------------------------- ---

    # --- Argument for transform (can be loaded from config) ---
    parser.add_argument('--transform', type=str, default=None, # Default to None, will be loaded from config
                        help='Transform type to use (e.g., MLPNET, default from config)')
    # --- ---------------------------------------------------- ---

    # --- Argument for mnist_digit (can be loaded from config) ---
    parser.add_argument('--mnist_digit', type=int, default=None, # Default to None, will be loaded from config
                        help='MNIST digit for split loading (default from config)')
    # --- ---------------------------------------------------- ---

    # --- Argument for cifar_class (can be loaded from config) ---
    parser.add_argument('--cifar_class', type=int, default=None, # Default to None, will be loaded from config
                        help='CIFAR class index for split loading (default from config)')
    # --- ---------------------------------------------------- ---


    # --- Initial parse to get config path ---
    # We parse known args first to get the config file path
    temp_args, unknown = parser.parse_known_args()

    # --- Load config from YAML file ---
    config = {}
    if os.path.exists(temp_args.config_path):
        print(f"Loading configuration from: {temp_args.config_path}")
        with open(temp_args.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(f"Error loading YAML config: {exc}")
    else:
        print(f"Warning: Config file not found at {temp_args.config_path}. Using command-line defaults only.")
    # --- -------------------------- ---

    # --- Set defaults from config before final parse ---
    # Important: Only set parser defaults if the corresponding config key exists
    # This prevents errors if config keys are missing
    defaults_from_config = {}
    config_mapping = {
        'input_dim': 'input_dim',
        'hidden_dims': 'hidden_dims',
        'output_dim': 'output_dim',
        'dataset': 'dataset',
        'data_path': 'data_path',
        'batch_size': 'batch_size',
        'num_workers': 'num_workers',
        'transform': 'transform',
        'mnist_digit': 'mnist_digit',
        'cifar_class': 'cifar_class',
        'curve': 'curve_type', # YAML 'curve' maps to arg 'curve_type'
        'bezier_num_bends': 'curve_net_num_bends', # YAML 'bezier_num_bends' maps to arg 'curve_net_num_bends'
        'model': 'curve_net_base_model', # YAML 'model' maps to 'curve_net_base_model'
        # Add other mappings as needed
    }

    for config_key, arg_name in config_mapping.items():
        if config_key in config:
            defaults_from_config[arg_name] = config[config_key]

    parser.set_defaults(**defaults_from_config)
    # --- --------------------------------------- ---

    # --- Final parse with config defaults applied ---
    args = parser.parse_args()
    # --- ---------------------------------------- ---

    # Convert hidden_dims from list of strings/ints to list of ints if needed
    if args.hidden_dims and isinstance(args.hidden_dims, list):
        args.hidden_dims = [int(d) for d in args.hidden_dims]

    print("Running with arguments:", args)
    plot_loss_surface_mlp(args)
