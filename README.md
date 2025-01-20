# Model Merging for Continual Learning

We provide the code for all experiments presented in our paper.

## Code Organization

- Source code is in the `src` directory.
- Bash scripts for running experiments and their associated hyperparameters are in the `scripts` directory.

## Results

Our model merging results are summarized below:

### MNIST

|  **Method**       | **MLPNet**           | **MLPLarge**         | **MLPHuge**          |
|-------------------|-----------------------|-----------------------|-----------------------|
| **Joint Model**    | $96.97 \pm 0.20$      | $97.44 \pm 0.18$      | $97.22 \pm 0.21$      |
| **Model A**        | $91.68 \pm 0.65$      | $92.11 \pm 0.36$      | $91.92 \pm 0.66$      |
| **Model B**        | $87.56 \pm 0.21$      | $87.67 \pm 0.27$      | $87.81 \pm 0.15$      |
| **AVG**            | $81.30 \pm 1.87$      | $85.75 \pm 0.41$      | $86.19 \pm 0.27$      |
| **OT**             | $80.27 \pm 2.06$      | $85.42 \pm 0.56$      | $85.96 \pm 0.36$      |
| **MPF (ours)**     | **$97.32 \pm 0.07$**  | **$97.52 \pm 0.11$**  | **$97.76 \pm 0.07$**  |

### CIFAR-10

|  **Method**       | **MLPNet**           | **MLPLarge**         | **MLPHuge**          |
|-------------------|-----------------------|-----------------------|-----------------------|
| **Joint Model**    | $34.12 \pm 1.29$     | $33.94 \pm 1.37$      | $35.22 \pm 0.66$      |
| **Model A**        | $34.52 \pm 2.45$     | $35.27 \pm 2.04$      | $34.32 \pm 1.98$      |
| **Model B**        | $33.42 \pm 0.40$     | $35.95 \pm 0.88$      | $35.02 \pm 1.45$      |
| **AVG**            | $23.93 \pm 0.60$     | $27.06 \pm 2.32$      | $27.96 \pm 1.89$      |
| **OT**             | $25.29 \pm 1.04$     | $27.78 \pm 1.78$      | $27.87 \pm 0.90$      |
| **MPF (ours)**     | **$48.02 \pm 0.25$** | **$48.93 \pm 0.43$**  | **$49.62 \pm 0.39$**  |

Our analysis shows that MPF effectively integrates knowledge from both Model A and Model B, demonstrating robust synthesis without the catastrophic forgetting seen in baseline methods.

---

## Requirements

The main dependencies for running the code are:

- `pytorch`
- `torchvision`
- `tqdm`
- `Pillow`
- `numpy`
- `Python Optimal Transport (POT)`
- `tensorboard`

You can install these packages using the following command:

```bash
pip install torch torchvision tqdm pillow numpy pot tensorboard
