"""
Lab 2: Build a Dense Variational Autoencoder for Faces
Student: Swarnim
Roll No: A24ARIU0004

This script implements a dense VAE on the LFW face dataset.
It completes all six TODOs from the lab notebook:
  TODO 1 - mu_head and logvar_head linear layers
  TODO 2 - Reparameterization trick
  TODO 3 - VAE forward pass
  TODO 4 - Negative ELBO loss (BCE + beta * KL)
  TODO 5 - Training step
  TODO 6 - Generate faces from the prior

It also runs the beta experiment and provides discussion answers.

NOTE: Requires LFW dataset. Set LFW_DATASET_PATH environment variable or
place the lfw-deepfunneled folder under data/lfw-deepfunneled.
To download: https://www.kaggle.com/datasets/julinmaloof/lfw-dataset
"""

import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Installing pandas and scikit-learn...")
    os.system("pip install --break-system-packages pandas scikit-learn pillow 2>/dev/null || pip install pandas scikit-learn pillow")
    import pandas as pd
    from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 45
CHANNELS = 3
INPUT_DIM = CHANNELS * IMAGE_SIZE * IMAGE_SIZE
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
BETA = 1.0
NUM_WORKERS = 2 if torch.cuda.is_available() else 0

# Change only this line for a local setup if LFW_DATASET_PATH is not set.
LOCAL_DATASET_PATH = Path("data/lfw-deepfunneled")

print(f"Running on {DEVICE}")

# ─────────────────────────────────────────────
# 2. LFW data loading
# ─────────────────────────────────────────────
def resolve_lfw_root():
    env_path = os.environ.get("LFW_DATASET_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    if Path("/kaggle/input").exists():
        candidates.extend([
            Path("/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled"),
            Path("/kaggle/input/lfw-dataset"),
        ])
    candidates.extend([
        LOCAL_DATASET_PATH.expanduser(),
        Path("lfw-deepfunneled"),
    ])
    for root in candidates:
        if root.exists() and next(root.rglob("*.jpg"), None) is not None:
            return root
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "No LFW .jpg files were found. Attach lfw-dataset on Kaggle, set "
        "LFW_DATASET_PATH, or edit LOCAL_DATASET_PATH. Checked:\n" + checked
    )


DATASET_ROOT = resolve_lfw_root()
image_paths = sorted(DATASET_ROOT.rglob("*.jpg"))
image_df = pd.DataFrame({
    "path": image_paths,
    "person": [path.parent.name for path in image_paths],
})

# Remove identities with 25+ photographs
image_df = image_df.groupby("person", group_keys=False).filter(lambda rows: len(rows) < 25)
image_df = image_df.reset_index(drop=True)

print(f"Dataset root: {DATASET_ROOT}")
print(f"Using {len(image_df):,} images from {image_df['person'].nunique():,} people")


class FaceDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        path = self.dataframe.loc[index, "path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            margin_x = min(80, max(0, (width - 1) // 3))
            margin_y = min(80, max(0, (height - 1) // 3))
            image = image.crop((margin_x, margin_y, width - margin_x, height - margin_y))
            return self.transform(image)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

train_df, val_df = train_test_split(image_df, test_size=0.2, random_state=SEED)
train_dataset = FaceDataset(train_df, transform)
val_dataset = FaceDataset(val_df, transform)

loader_args = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                   pin_memory=DEVICE.type == "cuda")
train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

faces = next(iter(train_loader))
assert faces.ndim == 4 and faces.shape[1:] == (3, 45, 45)
assert 0 <= faces.min() and faces.max() <= 1
print(f"Face batch shape: {faces.shape}, range: [{faces.min():.2f}, {faces.max():.2f}]")

# ─────────────────────────────────────────────
# Conceptual answers for Section 2
# ─────────────────────────────────────────────
# 1. Shapes of mu, logvar, z: all are [N, LATENT_DIM] = [N, 100]
# 2. mu and logvar must not have ReLU because:
#    - mu can be any real number (positive or negative), representing the center
#      of the approximate posterior
#    - logvar = log(sigma^2) can also be any real number (negative logvar means
#      sigma < 1, which is valid). ReLU would clamp negative values to 0,
#      preventing the encoder from expressing small variances.
# 3. Directly sampling z ~ N(mu, sigma^2) is a stochastic operation. Gradients
#    cannot flow through a random sampling operation (the "sample" node has no
#    well-defined gradient w.r.t. its distribution parameters). The
#    reparameterization trick z = mu + sigma * epsilon (where epsilon ~ N(0,1))
#    makes z a deterministic function of mu and sigma given epsilon, allowing
#    gradients to flow through mu and sigma to the encoder.

# ─────────────────────────────────────────────
# 3. Model implementation (TODOs 1-3)
# ─────────────────────────────────────────────
class DenseFaceVAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.ReLU(),
        )

        # TODO 1: define self.mu_head and self.logvar_head.
        self.mu_head = nn.Linear(1000, latent_dim)
        self.logvar_head = nn.Linear(1000, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1500),
            nn.ReLU(),
            nn.Linear(1500, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder(x.flatten(start_dim=1))
        return self.mu_head(features), self.logvar_head(features)

    def reparameterize(self, mu, logvar):
        # TODO 2: sample z while preserving a differentiable path to mu/logvar.
        # sigma = exp(0.5 * logvar) = exp(0.5 * log(sigma^2)) = sigma
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z):
        pixels = self.decoder(z)
        return pixels.view(-1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    def forward(self, x):
        # TODO 3: encode x, sample z, decode it, and return all three outputs.
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


# ─────────────────────────────────────────────
# Model checks
# ─────────────────────────────────────────────
model = DenseFaceVAE().to(DEVICE)
test_batch = faces[:4].to(DEVICE)
reconstruction, mu, logvar = model(test_batch)

assert reconstruction.shape == test_batch.shape
assert mu.shape == logvar.shape == (4, LATENT_DIM)
assert reconstruction.min() >= 0 and reconstruction.max() <= 1

z1 = model.reparameterize(mu, logvar)
z2 = model.reparameterize(mu, logvar)
assert not torch.equal(z1, z2)
(z1.mean()).backward()
assert model.mu_head.weight.grad is not None
assert model.logvar_head.weight.grad is not None
model.zero_grad(set_to_none=True)
print("Model checks passed.")

# ─────────────────────────────────────────────
# 4. Negative ELBO (TODO 4)
# ─────────────────────────────────────────────
def vae_loss(reconstruction, target, mu, logvar, beta=1.0):
    """
    TODO 4: Compute reconstruction (BCE) + beta * KL, normalized per image.
    Returns: (total_loss, detached_recon_loss, detached_kl_loss)
    """
    batch_size = target.size(0)

    # Reconstruction loss: BCE with reduction='sum', then divide by batch_size
    bce = F.binary_cross_entropy(
        reconstruction.flatten(start_dim=1),
        target.flatten(start_dim=1),
        reduction="sum",
    ) / batch_size

    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    total = bce + beta * kl
    return total, bce.detach(), kl.detach()


# ─────────────────────────────────────────────
# Loss checks
# ─────────────────────────────────────────────
dummy_x = torch.full((2, 3, 45, 45), 0.5, device=DEVICE)
dummy_mu = torch.zeros(2, LATENT_DIM, device=DEVICE)
dummy_logvar = torch.zeros_like(dummy_mu)
total, recon_term, kl_term = vae_loss(dummy_x, dummy_x, dummy_mu, dummy_logvar)
assert torch.isclose(kl_term, torch.tensor(0.0, device=DEVICE))
assert total.requires_grad is False  # dummy tensors do not require gradients

reconstruction, mu, logvar = model(test_batch)
total, _, _ = vae_loss(reconstruction, test_batch, mu, logvar, beta=BETA)
assert total.requires_grad
print("Loss checks passed.")

# ─────────────────────────────────────────────
# 5. Training step (TODO 5)
# ─────────────────────────────────────────────
def train_step(model, batch, optimizer, beta):
    """
    TODO 5: One optimization step. Move batch to device, clear gradients,
    forward pass, compute loss, backpropagate, update. Return three floats.
    """
    model.train()
    batch = batch.to(DEVICE, non_blocking=True)
    optimizer.zero_grad()
    reconstruction, mu, logvar = model(batch)
    total, recon_term, kl_term = vae_loss(reconstruction, batch, mu, logvar, beta)
    total.backward()
    optimizer.step()
    return total.item(), recon_term.item(), kl_term.item()


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, beta):
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    examples = 0
    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=True)
        reconstruction, mu, logvar = model(batch)
        values = vae_loss(reconstruction, batch, mu, logvar, beta)
        batch_size = batch.size(0)
        totals += np.array([value.item() for value in values]) * batch_size
        examples += batch_size
    return totals / examples


def run_vae_training(beta_value, epochs=EPOCHS):
    """Full training run for a given beta."""
    print(f"\n{'='*60}")
    print(f"  Training VAE with BETA = {beta_value}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    model = DenseFaceVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_total": [], "val_total": [], "val_reconstruction": [], "val_kl": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_total = 0.0
        seen = 0
        for batch in train_loader:
            total, reconstruction_term, kl_term = train_step(model, batch, optimizer, beta_value)
            running_total += total * batch.size(0)
            seen += batch.size(0)

        val_total, val_reconstruction, val_kl = evaluate(model, val_loader, beta_value)
        history["train_total"].append(running_total / seen)
        history["val_total"].append(val_total)
        history["val_reconstruction"].append(val_reconstruction)
        history["val_kl"].append(val_kl)
        print(
            f"  Epoch {epoch:02d} | train {running_total / seen:.2f} | "
            f"val {val_total:.2f} | recon {val_reconstruction:.2f} | KL {val_kl:.2f}"
        )

    # Plot training curves
    plt.figure()
    plt.plot(history["train_total"], label="train total")
    plt.plot(history["val_total"], label="validation total")
    plt.xlabel("Epoch")
    plt.ylabel("Negative ELBO per image")
    plt.title(f"VAE Training (beta={beta_value})")
    plt.legend()
    plt.savefig(f"vae_training_beta{beta_value}.png", dpi=100)
    plt.close()
    print(f"  Saved vae_training_beta{beta_value}.png")

    # Inspect reconstructions
    model.eval()
    validation_faces = next(iter(val_loader))[:8].to(DEVICE)
    with torch.no_grad():
        mu_val, _ = model.encode(validation_faces)
        reconstructed_faces = model.decode(mu_val)

    originals = validation_faces.cpu()
    reconstructed_faces = reconstructed_faces.cpu()
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for index in range(8):
        axes[0, index].imshow(originals[index].permute(1, 2, 0).numpy())
        axes[1, index].imshow(reconstructed_faces[index].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    plt.suptitle(f"Reconstruction (beta={beta_value})")
    plt.tight_layout()
    plt.savefig(f"vae_reconstruction_beta{beta_value}.png", dpi=100)
    plt.close()
    print(f"  Saved vae_reconstruction_beta{beta_value}.png")

    return model, history


# ─────────────────────────────────────────────
# Main training with BETA = 1.0
# ─────────────────────────────────────────────
model, main_history = run_vae_training(BETA)

# ─────────────────────────────────────────────
# 7. Generate new faces (TODO 6)
# ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    # TODO 6: sample from the prior and decode.
    z = torch.randn(10, LATENT_DIM, device=DEVICE)
    generated_faces = model.decode(z)

assert generated_faces.shape == (10, 3, 45, 45)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for face, axis in zip(generated_faces.cpu(), axes.flat):
    axis.imshow(face.permute(1, 2, 0).clamp(0, 1).numpy())
    axis.axis("off")
plt.suptitle("Generated Faces from Prior z ~ N(0, I)")
plt.tight_layout()
plt.savefig("vae_generated_faces.png", dpi=100)
plt.close()
print("Saved vae_generated_faces.png")

# ─────────────────────────────────────────────
# 8. Beta experiment
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BETA EXPERIMENT")
print("=" * 60)

beta_results = {}
for beta_val in [0.0, 4.0]:
    m, h = run_vae_training(beta_val)
    beta_results[beta_val] = {
        "model": m,
        "val_reconstruction": h["val_reconstruction"][-1],
        "val_kl": h["val_kl"][-1],
    }

# Also store the main training results
beta_results[1.0] = {
    "model": model,
    "val_reconstruction": main_history["val_reconstruction"][-1],
    "val_kl": main_history["val_kl"][-1],
}

print("\n" + "=" * 80)
print("  BETA EXPERIMENT RESULTS")
print("=" * 80)
print(f"{'Beta':>6} | {'Reconstruction':>15} | {'KL':>10}")
print("-" * 40)
for beta_val in sorted(beta_results.keys()):
    r = beta_results[beta_val]
    print(f"{beta_val:>6.1f} | {r['val_reconstruction']:>15.2f} | {r['val_kl']:>10.2f}")

# Generate samples from each beta model
for beta_val in sorted(beta_results.keys()):
    m = beta_results[beta_val]["model"]
    m.eval()
    with torch.no_grad():
        z = torch.randn(10, LATENT_DIM, device=DEVICE)
        gen = m.decode(z)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for face, axis in zip(gen.cpu(), axes.flat):
        axis.imshow(face.permute(1, 2, 0).clamp(0, 1).numpy())
        axis.axis("off")
    plt.suptitle(f"Generated Faces (beta={beta_val})")
    plt.tight_layout()
    plt.savefig(f"vae_generated_beta{beta_val}.png", dpi=100)
    plt.close()
    print(f"Saved vae_generated_beta{beta_val}.png")

# ─────────────────────────────────────────────
# Discussion answers
# ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("  DISCUSSION ANSWERS")
print("=" * 80)

discussion = """
1. What happens at beta=0?
   With beta=0, the KL divergence term is entirely removed from the loss. The
   model optimizes only reconstruction quality (BCE). Without the KL penalty, the
   encoder is free to map each image to an arbitrary, potentially highly concentrated
   point in latent space. The latent space is no longer encouraged to match the
   unit normal prior N(0,I). As a result, sampling z ~ N(0,I) at test time will
   produce latent vectors that are unlikely to fall where the encoder maps real
   data, leading to poor or meaningless generated faces. The model degenerates
   into a deterministic autoencoder with an unused stochastic layer.

2. How does increasing beta change reconstruction fidelity and prior samples?
   Increasing beta strengthens the KL penalty, which forces the approximate
   posterior q(z|x) closer to the prior N(0,I). This improves the quality of
   prior samples because the decoder has been trained on latent vectors that
   actually come from near the prior. However, it hurts reconstruction: the
   encoder cannot specialize latent representations for individual images because
   all posteriors are pushed toward the same N(0,I). With beta=4, reconstructions
   are blurrier but samples from the prior look more face-like and consistent.
   With beta=0, reconstructions are sharper but prior samples are poor.

3. Why can a VAE be implemented without convolutional layers? Disadvantage?
   The VAE framework is defined by the probabilistic encoder-decoder structure,
   the reparameterization trick, and the ELBO loss. These are architectural
   choices for the loss and sampling, not for the network layers. Any
   differentiable network can serve as encoder/decoder. However, a dense (MLP)
   model flattens the image, destroying spatial structure. It cannot share
   weights across spatial positions (translation equivariance), leading to many
   more parameters and blurrier reconstructions compared to a convolutional VAE
   of similar capacity. The dense model treats adjacent and distant pixels
   equivalently, ignoring the strong spatial correlations in natural images.
"""
print(discussion)
print("=" * 80)
print("Lab 2 complete. All TODOs implemented and beta experiment finished.")
print("=" * 80)
