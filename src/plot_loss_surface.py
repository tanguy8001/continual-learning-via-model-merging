import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from models import fcmodel # Assuming FCModelBase is the base architecture
# from curves_MLP import CurveMLP, build_state_dict # Import necessary components - Check if this import path is correct
try:
    from curves_MLP import CurveMLP, build_state_dict
except ImportError:
    # Assuming curves_MLP.py is in the same directory (src)
    from .curves_MLP import CurveMLP, build_state_dict

from torch.nn.utils import parameters_to_vector
from torch.nn.utils.stateless import functional_call
# Import your dataloader function (e.g., from train_models or data)
# from data import get_dataloaders_for_plotting


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

    # Load trained CurveMLP
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

    # Get dataloader for evaluation (use test set)
    # TODO: Replace placeholder with actual dataloader
    print("Using placeholder dataloader!")
    input_dim = config_A['input_dim']
    output_dim = config_A['output_dim']
    testloader = [(torch.randn(args.batch_size, input_dim), torch.randint(0, output_dim, (args.batch_size,))) for _ in range(5)]


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
    eval_model.eval() # Ensure model is in eval mode
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
                for k, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    # Use functional_call for efficiency
                    try:
                        # Ensure inputs match the expected input dim
                        if inputs.shape[1] != config_A['input_dim']:
                             print(f"Input shape mismatch in batch {k}: Expected {config_A['input_dim']}, Got {inputs.shape[1]}")
                             continue # Skip batch or handle error
                        outputs = functional_call(eval_model, current_state, (inputs,))
                        loss = criterion(outputs, targets)
                        total_loss += loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)
                    except Exception as e:
                         print(f"Error during functional_call or loss calculation in batch {k}: {e}")
                         print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}, Output shape: {outputs.shape if 'outputs' in locals() else 'N/A'}")
                         raise


                if total_samples == 0:
                     loss_surface[j, i] = float('nan') # Or some indicator of no data
                     print(f"Warning: No samples processed for grid point ({i},{j})")
                else:
                    loss_surface[j, i] = total_loss / total_samples # Store average loss
            print(f"Column {i+1}/{args.resolution} evaluated.")


    # --- 5. Calculate and Project Curve ---
    print("Projecting curve...")
    ts = torch.linspace(0, 1, 100, device=device)
    curve_points_x = []
    curve_points_y = []
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

            curve_points_x.append(x_scaled.item())
            curve_points_y.append(y_scaled.item())


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
    plt.plot(curve_points_x, curve_points_y, marker='.', linestyle='-', color='red', label='CurveMLP Path')

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
    parser.add_argument('--xmin', type=float, default=-1.0, help='Min x range')
    parser.add_argument('--xmax', type=float, default=1.0, help='Max x range')
    parser.add_argument('--ymin', type=float, default=-1.0, help='Min y range')
    parser.add_argument('--ymax', type=float, default=1.0, help='Max y range')
    parser.add_argument('--resolution', type=int, default=20, help='Grid resolution')
    parser.add_argument('--output_plot_path', type=str, default='curvemlp_loss_surface.png', help='Output path for the plot')
    parser.add_argument('--gpu_ids', type=str, default=None, help='GPU ID(s) to use (e.g., "0" or "0,1"). None uses CPU.')


    args = parser.parse_args()

    # Convert hidden_dims from list of strings/ints to list of ints if needed (though argparse nargs='+' should handle ints)
    if args.hidden_dims:
        args.hidden_dims = [int(d) for d in args.hidden_dims]

    # Basic check for required config args if config is likely missing (can't know for sure without loading)
    # Note: This check isn't perfect as we haven't loaded yet. The logic inside plot_loss_surface_mlp handles the actual check.
    print("Running with arguments:", args)


    plot_loss_surface_mlp(args)
