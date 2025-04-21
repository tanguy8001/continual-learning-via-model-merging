# filename: ode_merging_julia.jl

using Flux
using Flux: @epochs, destructure, Chain, Dense, ReLU, logitcrossentropy, params, state, loadmodel! # Add necessary Flux components
using DifferentialEquations
using DiffEqFlux
using MLDatasets: MNIST
using Statistics: mean
using CUDA # Optional GPU support
using Random: shuffle, seed!
using ProgressMeter: @showprogress
using Parameters: @with_kw # For keyword arguments
using Zygote # Ensure Zygote is available for AD

# --- Configuration ---
seed!(1234)

@with_kw struct Config
    dataset::String = "MNIST"
    digit_A::Int = 0
    digit_B::Int = 1
    data_path::String = "./data"
    batch_size::Int = 128
    base_model_epochs::Int = 3 # Fewer epochs for demonstration
    node_epochs::Int = 5      # Fewer epochs for demonstration
    node_lr::Float64 = 0.005
    base_model_lr::Float64 = 0.01
    hidden_dims::Vector{Int} = [100, 50] # Smaller model for demo
    input_dim::Int = 28 * 28
    output_dim::Int = 10
    t_samples::Int = 5      # Number of time points to sample in loss
    t_eval_grid::Int = 21   # Number of points for final evaluation
    endpoint_weight::Float64 = 0.1 # Weight for endpoint loss
    use_gpu::Bool = CUDA.functional() # Automatically check for GPU
end

cfg = Config()

# --- Device Handling ---
if cfg.use_gpu
    @info "Using CUDA GPU"
    CUDA.allowscalar(false)
    device = gpu
else
    @info "Using CPU"
    device = cpu
end

# --- Data Loading ---
function get_mnist_data(digit, batch_size, data_path)
    mkpath(data_path)
    # Load MNIST dataset
    train_x, train_y = MNIST(split=:train, dir=data_path)[:]
    test_x, test_y = MNIST(split=:test, dir=data_path)[:]

    # Filter by digit
    train_indices = findall(==(digit), train_y)
    test_indices = findall(==(digit), test_y)

    # Select subset based on indices and flatten images
    train_x_digit = Flux.flatten(train_x[:, :, train_indices])
    train_y_digit = Flux.onehotbatch(train_y[train_indices], 0:9) # Use original labels for onehot

    test_x_digit = Flux.flatten(test_x[:, :, test_indices])
    test_y_digit = Flux.onehotbatch(test_y[test_indices], 0:9)

    # Create DataLoaders
    train_loader = Flux.DataLoader((train_x_digit, train_y_digit), batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader((test_x_digit, test_y_digit), batchsize=batch_size)

    return train_loader, test_loader
end

# Create separate dataloaders for two digits and a fused loader
train_loader_A, _ = get_mnist_data(cfg.digit_A, cfg.batch_size, cfg.data_path)
train_loader_B, _ = get_mnist_data(cfg.digit_B, cfg.batch_size, cfg.data_path)
_, test_loader_full = MNIST(split=:test, dir=cfg.data_path)[:] # Use full test set for evaluation
test_loader_full_dl = Flux.DataLoader((Flux.flatten(test_loader_full), Flux.onehotbatch(test_loader_full |> MNIST.labels, 0:9)), batchsize=cfg.batch_size) |> device


# Combine data from both loaders for NODE training (simple concatenation for demo)
all_train_x = hcat(collect(train_loader_A.data[1]), collect(train_loader_B.data[1]))
all_train_y = hcat(collect(train_loader_A.data[2]), collect(train_loader_B.data[2]))
fused_loader = Flux.DataLoader((all_train_x, all_train_y), batchsize=cfg.batch_size, shuffle=true) |> device
train_loader_A = train_loader_A |> device
train_loader_B = train_loader_B |> device

# --- Model Definition ---
function create_fc_model(input_dim, hidden_dims, output_dim)
    layers = Any[Dense(input_dim, hidden_dims[1], relu)]
    for i in 1:(length(hidden_dims)-1)
        push!(layers, Dense(hidden_dims[i], hidden_dims[i+1], relu))
    end
    push!(layers, Dense(hidden_dims[end], output_dim))
    return Chain(layers...)
end

# --- Base Model Training Function ---
function train_base_model!(model, loader, opt, epochs, device)
    model = model |> device
    ps = params(model)
    @info "Training base model..."
    for epoch in 1:epochs
        @showprogress desc="Epoch $epoch/$(epochs)..." for (x, y) in loader
            gs = gradient(ps) do
                y_hat = model(x)
                logitcrossentropy(y_hat, y)
            end
            Flux.update!(opt, ps, gs)
        end
    end
    return model |> cpu # Move back to CPU after training
end

# --- Accuracy Evaluation ---
function evaluate_accuracy(model, loader, device)
    model = model |> device
    correct = 0
    total = 0
    for (x, y) in loader
        y_hat = model(x)
        correct += sum(Flux.onecold(y_hat |> cpu) .== Flux.onecold(y |> cpu))
        total += size(y, 2)
    end
    return correct / total * 100
end


# --- Train Base Models ---
model_A = create_fc_model(cfg.input_dim, cfg.hidden_dims, cfg.output_dim)
model_B = create_fc_model(cfg.input_dim, cfg.hidden_dims, cfg.output_dim)
opt_base = ADAM(cfg.base_model_lr)

model_A = train_base_model!(model_A, train_loader_A, opt_base, cfg.base_model_epochs, device)
model_B = train_base_model!(model_B, train_loader_B, opt_base, cfg.base_model_epochs, device)

acc_A = evaluate_accuracy(model_A, test_loader_full_dl, device)
acc_B = evaluate_accuracy(model_B, test_loader_full_dl, device)
@info "Model A Accuracy (Full Test): $(acc_A)%"
@info "Model B Accuracy (Full Test): $(acc_B)%"

# --- Neural ODE Setup ---
theta_0, restructure_model = destructure(model_A)
theta_1, _ = destructure(model_B)
theta_0 = theta_0 |> device # Ensure start/end points are on the correct device
theta_1 = theta_1 |> device
param_dim = length(theta_0)

# Define the dynamics network f(θ, p, t)
# Input: θ (current parameters), Output: dθ/dt
# Note: DiffEqFlux often expects `f(du, u, p, t)` for inplace,
# or `f(u, p, t)` for out-of-place. NeuralODE typically uses out-of-place.
dudt_net = Chain(
    Dense(param_dim, 128, tanh), # Smaller hidden layer for ODE net
    Dense(128, param_dim)
) |> device

# Neural ODE definition
# `p` contains the parameters of `dudt_net`
node = NeuralODE(
    dudt_net,
    (0.0f0, 1.0f0), # Time span t=0 to t=1
    Tsit5(),       # ODE Solver Algorithm
    saveat=range(0.0f0, 1.0f0, length=cfg.t_samples), # Sample points for loss calculation
    reltol=1e-3, abstol=1e-3 # Relax tolerances for speed/memory
)
ps_node = params(node) |> device # Get parameters of the NeuralODE (which includes dudt_net params)

# --- Loss Function for Neural ODE ---
function node_loss(data_loader, node_model, p_node, base_model_restructure, θ_start, θ_end, endpoint_weight, device)
    total_task_loss = 0.0f0
    num_batches = 0

    # Limit batches for faster loss calculation during training demo
    max_batches_per_epoch = 10
    current_batch = 0

    for (x, y) in data_loader
        if current_batch >= max_batches_per_epoch
            break
        end
        x, y = x |> device, y |> device

        # Solve the ODE for the current batch with current NODE parameters `p_node`
        # Output `pred_path` is a matrix: rows=param_dim, cols=t_samples
        pred_path = node_model(θ_start, p_node)

        batch_task_loss = 0.0f0
        # Calculate loss at each sampled time point
        for i in 1:size(pred_path, 2) # Iterate through time samples (columns)
            θ_t = pred_path[:, i]
            temp_model = base_model_restructure(θ_t)
            y_hat = temp_model(x)
            batch_task_loss += logitcrossentropy(y_hat, y)
        end

        total_task_loss += batch_task_loss / size(pred_path, 2) # Average over time samples
        num_batches += 1
        current_batch += 1
    end

    # Final prediction to calculate endpoint loss
    # Need solution at t=1 specifically
    final_θ = node_model(θ_start, p_node, Val(false), save_idxs=nothing, tstops=[1.0f0])[:, end] # Efficiently get t=1 prediction

    # Endpoint Loss (encourage path to end at θ_1)
    # Start point is implicitly handled by the ODE initial condition
    loss_endpoint = sum(abs2, final_θ .- θ_end)

    # Average task loss over batches
    avg_task_loss = total_task_loss / num_batches

    # Combined loss
    combined_loss = avg_task_loss + endpoint_weight * loss_endpoint
    return combined_loss
end

# --- Neural ODE Training Loop ---
opt_node = ADAM(cfg.node_lr)

@info "Training Neural ODE..."
for epoch in 1:cfg.node_epochs
    epoch_loss = 0.0
    batch_count = 0
    @showprogress desc="NODE Epoch $epoch/$(cfg.node_epochs)..." for batch in fused_loader
         # Calculate loss and gradients
        loss, grads = Flux.withgradient(ps_node) do
             node_loss(fused_loader, node, ps_node, restructure_model, theta_0, theta_1, cfg.endpoint_weight, device)
        end

        # Handle potential NaN gradients
        if isnan(loss) || any(isnan, grads[ps_node[1]]) # Check grads of first parameter set in node
            @warn "NaN loss or gradient detected, skipping update."
            continue
        end

        # Update NODE parameters
        Flux.update!(opt_node, ps_node, grads)
        epoch_loss += loss
        batch_count +=1

        # Limit batches per epoch for faster demo run
         if batch_count >= 20
            break
        end
    end
    avg_loss = epoch_loss / batch_count
    @info "Epoch: $epoch, Avg Loss: $avg_loss"

     # Clear GPU memory if applicable
     if cfg.use_gpu
         GC.gc(); CUDA.reclaim()
     end
end

# --- Evaluation: Find Best Point on the Learned Path ---
@info "Evaluating learned ODE path..."
node = node |> device # Ensure NODE is on device for final solve
ps_node_final = params(node) # Use the final trained parameters

# Solve ODE over a finer grid for evaluation
eval_t_points = range(0.0f0, 1.0f0, length=cfg.t_eval_grid)
final_path = node(theta_0, ps_node_final, saveat=eval_t_points) |> cpu # Move path to CPU

best_acc = -1.0
best_t = -1.0
best_model_state = nothing

eval_loader_cpu = Flux.DataLoader((test_loader_full_dl.data[1] |> cpu, test_loader_full_dl.data[2] |> cpu), batchsize=cfg.batch_size) # Eval on CPU

@showprogress desc="Evaluating path..." for i in 1:cfg.t_eval_grid
    t = eval_t_points[i]
    θ_t_eval = final_path[:, i]
    eval_model = restructure_model(θ_t_eval) # Model is on CPU

    acc = evaluate_accuracy(eval_model, eval_loader_cpu, cpu) # Evaluate on CPU

    if acc > best_acc
        best_acc = acc
        best_t = t
        best_model_state = state(eval_model) # Save the state of the best model
    end
    # Optional: Print accuracy at each point
    # println("t = $(round(t, digits=3)), Acc = $(round(acc, digits=2))%")
end

@info "--- Results ---"
@info "Model A Accuracy: $(round(acc_A, digits=2))%"
@info "Model B Accuracy: $(round(acc_B, digits=2))%"
@info "Best Accuracy on ODE Path: $(round(best_acc, digits=2))% at t = $(round(best_t, digits=3))"

# You can now reconstruct the best model if needed:
# best_model = create_fc_model(cfg.input_dim, cfg.hidden_dims, cfg.output_dim)
# loadmodel!(best_model, best_model_state)

println("Julia script finished.")
