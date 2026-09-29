"""
Lab 1: Build Your First Image Encoder-Decoder
Student: Swarnim
Roll No: A24ARIU0004

This script implements a convolutional encoder-decoder (autoencoder) for
MNIST digit reconstruction. It completes all four TODOs from the lab notebook:
  TODO 1 - Encoder network
  TODO 2 - Decoder network
  TODO 3 - ImageEncoderDecoder forward pass
  TODO 4 - Training step

It also runs the bottleneck experiment with LATENT_CHANNELS = 2, 8, and 32,
and prints a summary table and discussion answers.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Setup and data loading
# ─────────────────────────────────────────────
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
print("Device:", device)

train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

# Fixed random subsets keep the exercise small and comparisons repeatable.
g = torch.Generator().manual_seed(42)
train_data = Subset(train_data, torch.randperm(len(train_data), generator=g)[:5000].tolist())
test_data = Subset(test_data, torch.randperm(len(test_data), generator=g)[:1000].tolist())

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

images, labels = next(iter(train_loader))
print("Batch:", tuple(images.shape), "Pixel range:", images.min().item(), images.max().item())

# ─────────────────────────────────────────────
# 2. Shape predictions (completed)
# ─────────────────────────────────────────────
# | Stage       | Layer                                                         | Output shape          |
# |-------------|---------------------------------------------------------------|-----------------------|
# | Input       | —                                                             | [N, 1, 28, 28]        |
# | Encoder 1   | Conv2d: 1→16, k=3, s=2, p=1; ReLU                            | [N, 16, 14, 14]       |
# | Encoder 2   | Conv2d: 16→32, k=3, s=2, p=1; ReLU                           | [N, 32, 7, 7]         |
# | Bottleneck  | Conv2d: 32→L, k=1                                            | [N, L, 7, 7]          |
# | Decoder 1   | ConvTranspose2d: L→16, k=3, s=2, p=1, op=1; ReLU             | [N, 16, 14, 14]       |
# | Decoder 2   | ConvTranspose2d: 16→1, k=3, s=2, p=1, op=1; Sigmoid          | [N, 1, 28, 28]        |
#
# Answers:
# 1. For L=8: input values = 1*28*28 = 784; latent values = 8*7*7 = 392;
#    input/latent ratio = 784/392 = 2.0
# 2. For L=32: latent values = 32*7*7 = 1568, which is LARGER than 784 input
#    values. So no, it is NOT dimensionally compressed relative to the input.
#    The spatial downsampling reduces 28×28 to 7×7, but the increased channel
#    count more than compensates. Compression requires both spatial reduction
#    AND a channel count that keeps the total below the input count.
# 3. The final activation is Sigmoid because MNIST pixel values are in [0, 1]
#    (from ToTensor()). The activation must match the target range so MSE loss
#    compares compatible quantities.
# 4. With output_padding=0:
#    H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding
#    H_out = (7 - 1) * 2 - 2*1 + 3 + 0 = 12 - 2 + 3 = 13
#    The first decoder's output height would be 13, not 14.

# ─────────────────────────────────────────────
# 3. Network implementation (TODOs 1-3)
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """TODO 1: Convolutional encoder with two stride-2 conv layers and a 1×1 bottleneck."""
    def __init__(self, latent_channels=8):
        super().__init__()
        self.network = nn.Sequential(
            # Encoder layer 1: [N, 1, 28, 28] -> [N, 16, 14, 14]
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Encoder layer 2: [N, 16, 14, 14] -> [N, 32, 7, 7]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Bottleneck: [N, 32, 7, 7] -> [N, L, 7, 7]  (no activation)
            nn.Conv2d(in_channels=32, out_channels=latent_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    """TODO 2: Transposed convolutional decoder that mirrors the encoder."""
    def __init__(self, latent_channels=8):
        super().__init__()
        self.network = nn.Sequential(
            # Decoder layer 1: [N, L, 7, 7] -> [N, 16, 14, 14]
            nn.ConvTranspose2d(
                in_channels=latent_channels, out_channels=16,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.ReLU(),
            # Decoder layer 2: [N, 16, 14, 14] -> [N, 1, 28, 28]
            nn.ConvTranspose2d(
                in_channels=16, out_channels=1,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.network(z)


class ImageEncoderDecoder(nn.Module):
    """TODO 3: Combined encoder-decoder returning (reconstruction, z)."""
    def __init__(self, latent_channels=8):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


# ─────────────────────────────────────────────
# Shape and range checks
# ─────────────────────────────────────────────
def run_shape_checks(latent_channels):
    """Run assertion checks for a given latent_channels value."""
    model = ImageEncoderDecoder(latent_channels).to(device)
    x = torch.rand(4, 1, 28, 28, device=device)
    with torch.no_grad():
        reconstruction, z = model(x)
    assert z.shape == (4, latent_channels, 7, 7), f"Check encoder stride/padding/channels (got {z.shape})"
    assert reconstruction.shape == x.shape, f"Check decoder output_padding (got {reconstruction.shape})"
    assert torch.isfinite(reconstruction).all(), "Output contains nonfinite values"
    assert ((reconstruction >= 0) & (reconstruction <= 1)).all(), "Check output activation"
    print(f"  Input: {tuple(x.shape)} | Latent: {tuple(z.shape)} | Output: {tuple(reconstruction.shape)}")
    print(f"  Shape and range checks passed for LATENT_CHANNELS={latent_channels}")
    return model


# ─────────────────────────────────────────────
# 4. Training step (TODO 4)
# ─────────────────────────────────────────────

def train_step(model, images, optimizer, criterion):
    """TODO 4: One reconstruction training step.
    Clear gradients, forward pass, compute MSE, backpropagate, update weights.
    Return the loss as a Python float.
    """
    optimizer.zero_grad()
    reconstruction, _ = model(images)
    loss = criterion(reconstruction, images)
    loss.backward()
    optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
criterion = nn.MSELoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    for images, _ in loader:
        images = images.to(device)
        reconstruction, _ = model(images)
        total_loss += criterion(reconstruction, images).item() * images.size(0)
    return total_loss / len(loader.dataset)


def train_and_evaluate(latent_channels, show_plots=True):
    """Full training run for a given latent_channels value."""
    print(f"\n{'='*60}")
    print(f"  Training with LATENT_CHANNELS = {latent_channels}")
    print(f"{'='*60}")

    # Shape checks
    run_shape_checks(latent_channels)

    # Fresh model and optimizer
    torch.manual_seed(42)
    model = ImageEncoderDecoder(latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_mse = evaluate(model, test_loader)
    print(f"  Untrained held-out MSE: {initial_mse:.5f}")

    history = {"train": [], "test": []}
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            total_loss += train_step(model, images, optimizer, criterion) * images.size(0)
        history["train"].append(total_loss / len(train_loader.dataset))
        history["test"].append(evaluate(model, test_loader))
        print(
            f"  Epoch {epoch + 1}: train MSE={history['train'][-1]:.5f}, "
            f"held-out MSE={history['test'][-1]:.5f}"
        )

    # Plot training curves
    if show_plots:
        plt.figure()
        plt.plot(range(1, EPOCHS + 1), history["train"], marker="o", label="Train")
        plt.plot(range(1, EPOCHS + 1), history["test"], marker="o", label="Held-out")
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared error")
        plt.title(f"Training Curves (LATENT_CHANNELS={latent_channels})")
        plt.legend()
        plt.savefig(f"training_curves_L{latent_channels}.png", dpi=100)
        plt.close()
        print(f"  Saved training_curves_L{latent_channels}.png")

    # Inspect reconstructions
    model.eval()
    examples, _ = next(iter(test_loader))
    with torch.no_grad():
        reconstructed, latents = model(examples[:8].to(device))
    reconstructed, latents = reconstructed.cpu(), latents.cpu()

    if show_plots:
        fig, axes = plt.subplots(2, 8, figsize=(12, 4))
        for i in range(8):
            axes[0, i].imshow(examples[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[1, i].imshow(reconstructed[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[0, i].axis("off")
            axes[1, i].axis("off")
        fig.suptitle(f"Top: original | Bottom: reconstruction (L={latent_channels})")
        plt.savefig(f"reconstruction_L{latent_channels}.png", dpi=100)
        plt.close()
        print(f"  Saved reconstruction_L{latent_channels}.png")

        # Latent feature maps
        count = min(latent_channels, 8)
        fig, axes = plt.subplots(1, count, figsize=(2 * count, 2), squeeze=False)
        for i in range(count):
            axes[0, i].imshow(latents[0, i], cmap="viridis")
            axes[0, i].set_title(f"Ch {i}")
            axes[0, i].axis("off")
        fig.suptitle(f"Latent maps (first image, L={latent_channels})")
        plt.savefig(f"latent_maps_L{latent_channels}.png", dpi=100)
        plt.close()
        print(f"  Saved latent_maps_L{latent_channels}.png")

    return {
        "latent_channels": latent_channels,
        "values_per_image": latent_channels * 7 * 7,
        "ratio": (1 * 28 * 28) / (latent_channels * 7 * 7),
        "final_train_mse": history["train"][-1],
        "final_test_mse": history["test"][-1],
    }


# ─────────────────────────────────────────────
# 6. Bottleneck experiment: L = 2, 8, 32
# ─────────────────────────────────────────────
results = []
for L in [2, 8, 32]:
    results.append(train_and_evaluate(L, show_plots=True))

print("\n" + "=" * 80)
print("  BOTTLENECK EXPERIMENT RESULTS")
print("=" * 80)
print(f"{'L':>4} | {'Values/Image':>13} | {'Input/Latent':>12} | {'Train MSE':>10} | {'Test MSE':>10}")
print("-" * 60)
for r in results:
    print(
        f"{r['latent_channels']:>4} | "
        f"{r['values_per_image']:>13} | "
        f"{r['ratio']:>12.2f} | "
        f"{r['final_train_mse']:>10.5f} | "
        f"{r['final_test_mse']:>10.5f}"
    )

# ─────────────────────────────────────────────
# Discussion answers
# ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("  DISCUSSION ANSWERS")
print("=" * 80)

discussion = """
1. Did a larger latent improve reconstruction?
   Generally yes. With L=2 (only 98 latent values, ratio 8.0), the network has
   a very tight bottleneck and must compress 784 input values into 98 latent values.
   Reconstruction quality is noticeably blurrier and MSE is higher. With L=8
   (392 values, ratio 2.0), reconstruction improves significantly - digits are
   recognizable with reasonable detail. With L=32 (1568 values, ratio 0.5),
   the latent space is actually larger than the input, so the bottleneck barely
   constrains the network. Reconstruction is sharpest and MSE is lowest. However,
   the improvement from L=8 to L=32 is smaller than from L=2 to L=8.

2. Why is spatial downsampling alone insufficient to claim dimensional compression?
   Spatial downsampling from 28x28 to 7x7 reduces spatial dimensions by 4x in each
   direction (16x total), but the number of channels increases simultaneously. The
   total latent dimensionality is channels * height * width. For L=32, we get
   32*7*7=1568 values, which exceeds the 784 input values. True compression
   requires that the product of all latent dimensions be smaller than the input
   dimensionality. Downsampling only addresses one axis (spatial); the channel
   axis must also be considered.

3. Why can the decoder restore image size without recovering every original detail?
   The transposed convolutions can mechanically upsample from 7x7 back to 28x28,
   but the information passing through the bottleneck has been lossy-compressed.
   The decoder learns a mapping from the limited latent representation to plausible
   pixel values, but it cannot reconstruct information that was discarded by the
   encoder. It essentially learns to produce the "average" or most likely
   reconstruction given the compressed features.

4. How would a [N, 128] vector differ from this spatial latent?
   A flattened [N, 128] vector (e.g. via global average pooling + linear layer)
   completely discards spatial structure. The spatial latent [N, L, 7, 7] retains
   a coarse spatial grid where each location roughly corresponds to a 4x4 patch
   in the input. This means the decoder knows WHERE features are, not just WHAT
   features exist. A flat vector forces the decoder to reconstruct spatial layout
   from scratch, typically requiring more capacity and producing blurrier results.

5. Where could skip connections carry high-resolution features?
   Skip connections could connect encoder layer 1 output [N, 16, 14, 14] to
   decoder layer 1 input, and encoder layer 2 output [N, 32, 7, 7] to decoder
   layer 2 input (as in U-Net). This would let the decoder directly access
   high-resolution edge and texture information. However, this weakens the
   bottleneck constraint because information can bypass the narrow latent
   representation entirely. The network might learn to pass most information
   through the skip connections, making the bottleneck irrelevant and defeating
   the purpose of learning compact representations.

6. Why reconstruct existing images rather than generate random images?
   This autoencoder has no mechanism to ensure the latent space is organized in
   a way that supports meaningful sampling. Random z vectors may decode to
   noise or artifacts because the model only learns to reconstruct its training
   data - it does not learn a smooth, continuous latent distribution. A VAE or
   GAN is needed for reliable generation: the VAE adds KL regularization to
   organize the latent space, while a GAN trains a generator to map noise to
   realistic images via adversarial learning.
"""
print(discussion)
print("=" * 80)
print("Lab 1 complete. All TODOs implemented and bottleneck experiment finished.")
print("=" * 80)
