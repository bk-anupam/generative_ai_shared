import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

print("Running on", DEVICE)


def resolve_lfw_root():
    candidates = []

    if os.environ.get("LFW_DATASET_PATH"):
        candidates.append(Path(os.environ["LFW_DATASET_PATH"]))

    candidates.append(Path("/kaggle/input/lfw-dataset/lfw-deepfunneled"))
    candidates.append(Path("/kaggle/input/lfw-dataset"))
    candidates.append(Path("data/lfw-deepfunneled"))
    candidates.append(Path("lfw-deepfunneled"))

    for root in candidates:
        if root.exists():
            jpg_files = list(root.rglob("*.jpg"))
            if len(jpg_files) > 0:
                return root

    raise FileNotFoundError(
        "No LFW .jpg files were found. "
        "Set LFW_DATASET_PATH or place lfw-deepfunneled in data/."
    )


LFW_ROOT = resolve_lfw_root()

print("LFW dataset:", LFW_ROOT)


class FaceDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)

        all_images = list(self.root.rglob("*.jpg"))

        identity_counts = {}

        for image_path in all_images:
            identity = image_path.parent.name
            identity_counts[identity] = identity_counts.get(identity, 0) + 1

        valid_identities = {
            identity
            for identity, count in identity_counts.items()
            if count < 25
        }

        self.image_paths = [
            image_path
            for image_path in all_images
            if image_path.parent.name in valid_identities
        ]

        self.transform = transforms.Compose([
            transforms.CenterCrop(150),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image


dataset = FaceDataset(LFW_ROOT)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("Total images:", len(dataset))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


class DenseVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_DIM, 1000),
            nn.ReLU()
        )

        self.mu_head = nn.Linear(1000, LATENT_DIM)
        self.logvar_head = nn.Linear(1000, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 1000),
            nn.ReLU(),
            nn.Linear(1000, INPUT_DIM),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        h = self.encoder(x)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decoder(z)
        reconstruction = reconstruction.view(
            -1,
            CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        return reconstruction, mu, logvar


model = DenseVAE().to(DEVICE)

print(model)

x = next(iter(train_loader))
x = x.to(DEVICE)

with torch.no_grad():
    reconstruction, mu, logvar = model(x)

print("Input shape:", x.shape)
print("Reconstruction shape:", reconstruction.shape)
print("Mu shape:", mu.shape)
print("Logvar shape:", logvar.shape)

assert x.shape == reconstruction.shape
assert mu.shape[1] == LATENT_DIM
assert logvar.shape[1] == LATENT_DIM

print("Shape checks passed.")

def vae_loss(reconstruction, x, mu, logvar, beta=BETA):
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction,
        x,
        reduction="sum"
    ) / x.size(0)

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / x.size(0)

    total_loss = reconstruction_loss + beta * kl_loss

    return (
        total_loss,
        reconstruction_loss.detach(),
        kl_loss.detach()
    )

with torch.no_grad():
    test_reconstruction, test_mu, test_logvar = model(x)

    total_loss, reconstruction_loss, kl_loss = vae_loss(
        test_reconstruction,
        x,
        test_mu,
        test_logvar,
        BETA
    )

print("Total loss:", total_loss.item())
print("Reconstruction loss:", reconstruction_loss.item())
print("KL loss:", kl_loss.item())


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


def train_step(model, batch, optimizer, beta):
    batch = batch.to(DEVICE)

    optimizer.zero_grad()

    reconstruction, mu, logvar = model(batch)

    loss, reconstruction_loss, kl_loss = vae_loss(
        reconstruction,
        batch,
        mu,
        logvar,
        beta
    )

    loss.backward()
    optimizer.step()

    return (
        loss.item(),
        reconstruction_loss.item(),
        kl_loss.item()
    )

def validate(model, loader, beta):
    model.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)

            reconstruction, mu, logvar = model(batch)

            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                batch,
                mu,
                logvar,
                beta
            )

            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_kl += kl_loss.item()
            total_batches += 1

    return (
        total_loss / total_batches,
        total_reconstruction / total_batches,
        total_kl / total_batches
    )

train_losses = []
train_reconstruction_losses = []
train_kl_losses = []

val_losses = []
val_reconstruction_losses = []
val_kl_losses = []


for epoch in range(EPOCHS):
    model.train()

    epoch_loss = 0.0
    epoch_reconstruction = 0.0
    epoch_kl = 0.0
    num_batches = 0

    for batch in train_loader:
        loss, reconstruction_loss, kl_loss = train_step(
            model,
            batch,
            optimizer,
            BETA
        )

        epoch_loss += loss
        epoch_reconstruction += reconstruction_loss
        epoch_kl += kl_loss
        num_batches += 1

    epoch_loss /= num_batches
    epoch_reconstruction /= num_batches
    epoch_kl /= num_batches

    val_loss, val_reconstruction, val_kl = validate(
        model,
        val_loader,
        BETA
    )

    train_losses.append(epoch_loss)
    train_reconstruction_losses.append(epoch_reconstruction)
    train_kl_losses.append(epoch_kl)

    val_losses.append(val_loss)
    val_reconstruction_losses.append(val_reconstruction)
    val_kl_losses.append(val_kl)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Train Loss: {epoch_loss:.4f} "
        f"Train Recon: {epoch_reconstruction:.4f} "
        f"Train KL: {epoch_kl:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Recon: {val_reconstruction:.4f} "
        f"Val KL: {val_kl:.4f}"
    )

    epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("vae_learning_curve.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    train_reconstruction_losses,
    label="Train Reconstruction"
)
plt.plot(
    epochs,
    val_reconstruction_losses,
    label="Validation Reconstruction"
)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("Reconstruction Loss")
plt.legend()
plt.tight_layout()
plt.savefig("vae_reconstruction_loss.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(epochs, train_kl_losses, label="Train KL")
plt.plot(epochs, val_kl_losses, label="Validation KL")
plt.xlabel("Epoch")
plt.ylabel("KL Loss")
plt.title("KL Divergence")
plt.legend()
plt.tight_layout()
plt.savefig("vae_kl_loss.png")
plt.show()

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("vae_learning_curve.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    train_reconstruction_losses,
    label="Train Reconstruction"
)
plt.plot(
    epochs,
    val_reconstruction_losses,
    label="Validation Reconstruction"
)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("Reconstruction Loss")
plt.legend()
plt.tight_layout()
plt.savefig("vae_reconstruction_loss.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(epochs, train_kl_losses, label="Train KL")
plt.plot(epochs, val_kl_losses, label="Validation KL")
plt.xlabel("Epoch")
plt.ylabel("KL Loss")
plt.title("KL Divergence")
plt.legend()
plt.tight_layout()
plt.savefig("vae_kl_loss.png")
plt.show()

model.eval()

with torch.no_grad():
    original_batch = next(iter(val_loader)).to(DEVICE)
    reconstructed_batch, _, _ = model(original_batch)

original_batch = original_batch.cpu()
reconstructed_batch = reconstructed_batch.cpu()

num_images = 8

plt.figure(figsize=(16, 4))

for i in range(num_images):
    plt.subplot(2, num_images, i + 1)
    plt.imshow(original_batch[i].permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(2, num_images, num_images + i + 1)
    plt.imshow(reconstructed_batch[i].permute(1, 2, 0))
    plt.axis("off")

plt.tight_layout()
plt.savefig("vae_reconstructions.png")
plt.show()

model.eval()

with torch.no_grad():
    z = torch.randn(10, LATENT_DIM).to(DEVICE)

    generated = model.decoder(z)

    generated = generated.view(
        10,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

generated = generated.cpu()

plt.figure(figsize=(15, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated[i].permute(1, 2, 0))
    plt.axis("off")

plt.tight_layout()
plt.savefig("vae_generated_faces.png")
plt.show()

def train_model_for_beta(beta):
    beta_model = DenseVAE().to(DEVICE)

    beta_optimizer = torch.optim.Adam(
        beta_model.parameters(),
        lr=LEARNING_RATE
    )

    beta_train_losses = []
    beta_val_losses = []
    beta_val_reconstruction_losses = []
    beta_val_kl_losses = []

    for epoch in range(EPOCHS):
        beta_model.train()

        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            loss, reconstruction_loss, kl_loss = train_step(
                beta_model,
                batch,
                beta_optimizer,
                beta
            )

            total_loss += loss
            num_batches += 1

        total_loss /= num_batches

        val_loss, val_reconstruction, val_kl = validate(
            beta_model,
            val_loader,
            beta
        )

        beta_train_losses.append(total_loss)
        beta_val_losses.append(val_loss)
        beta_val_reconstruction_losses.append(val_reconstruction)
        beta_val_kl_losses.append(val_kl)

        print(
            f"Beta {beta} | "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {total_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Recon: {val_reconstruction:.4f} | "
            f"Val KL: {val_kl:.4f}"
        )

    return (
        beta_model,
        beta_train_losses,
        beta_val_losses,
        beta_val_reconstruction_losses,
        beta_val_kl_losses
    )


beta_0_results = train_model_for_beta(0.0)

beta_4_results = train_model_for_beta(4.0)


model_beta_0 = beta_0_results[0]
train_beta_0 = beta_0_results[1]
val_beta_0 = beta_0_results[2]
recon_beta_0 = beta_0_results[3]
kl_beta_0 = beta_0_results[4]

model_beta_4 = beta_4_results[0]
train_beta_4 = beta_4_results[1]
val_beta_4 = beta_4_results[2]
recon_beta_4 = beta_4_results[3]
kl_beta_4 = beta_4_results[4]

print()
print("Beta Experiment Results")
print()
print("Beta = 0")
print("Validation Reconstruction Loss:", recon_beta_0[-1])
print("Validation KL:", kl_beta_0[-1])

print()
print("Beta = 4")
print("Validation Reconstruction Loss:", recon_beta_4[-1])
print("Validation KL:", kl_beta_4[-1])

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    recon_beta_0,
    label="Beta = 0"
)

plt.plot(
    range(1, EPOCHS + 1),
    recon_beta_4,
    label="Beta = 4"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Reconstruction Loss")
plt.title("Beta Experiment - Reconstruction Loss")
plt.legend()
plt.tight_layout()
plt.savefig("beta_reconstruction_comparison.png")
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    kl_beta_0,
    label="Beta = 0"
)

plt.plot(
    range(1, EPOCHS + 1),
    kl_beta_4,
    label="Beta = 4"
)

plt.xlabel("Epoch")
plt.ylabel("Validation KL")
plt.title("Beta Experiment - KL Divergence")
plt.legend()
plt.tight_layout()
plt.savefig("beta_kl_comparison.png")
plt.show()


model_beta_0.eval()

with torch.no_grad():
    images = next(iter(val_loader)).to(DEVICE)
    reconstructions, _, _ = model_beta_0(images)

images = images.cpu()
reconstructions = reconstructions.cpu()

plt.figure(figsize=(16, 4))

for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(images[i].permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(2, 8, 8 + i + 1)
    plt.imshow(reconstructions[i].permute(1, 2, 0))
    plt.axis("off")

plt.tight_layout()
plt.savefig("beta_0_reconstructions.png")
plt.show()


model_beta_4.eval()

with torch.no_grad():
    images = next(iter(val_loader)).to(DEVICE)
    reconstructions, _, _ = model_beta_4(images)

images = images.cpu()
reconstructions = reconstructions.cpu()

plt.figure(figsize=(16, 4))

for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(images[i].permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(2, 8, 8 + i + 1)
    plt.imshow(reconstructions[i].permute(1, 2, 0))
    plt.axis("off")

plt.tight_layout()
plt.savefig("beta_4_reconstructions.png")
plt.show()

model_beta_0.eval()

with torch.no_grad():
    z = torch.randn(10, LATENT_DIM).to(DEVICE)
    generated_beta_0 = model_beta_0.decoder(z)

generated_beta_0 = generated_beta_0.view(
    10,
    CHANNELS,
    IMAGE_SIZE,
    IMAGE_SIZE
).cpu()

plt.figure(figsize=(15, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_beta_0[i].permute(1, 2, 0))
    plt.axis("off")

plt.tight_layout()
plt.savefig("beta_0_generated_faces.png")
plt.show()


model_beta_4.eval()

with torch.no_grad():
    z = torch.randn(10, LATENT_DIM).to(DEVICE)
    generated_beta_4 = model_beta_4.decoder(z)

generated_beta_4 = generated_beta_4.view(
    10,
    CHANNELS,
    IMAGE_SIZE,
    IMAGE_SIZE
).cpu()

plt.figure(figsize=(15, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_beta_4[i].permute(1, 2, 0))
    plt.axis("off")

plt.tight_layout()
plt.savefig("beta_4_generated_faces.png")
plt.show()

print()
print("Final Beta Comparison")
print("=" * 50)

print(
    f"Beta = 0 | "
    f"Validation Reconstruction = {recon_beta_0[-1]:.6f} | "
    f"Validation KL = {kl_beta_0[-1]:.6f}"
)

print(
    f"Beta = 4 | "
    f"Validation Reconstruction = {recon_beta_4[-1]:.6f} | "
    f"Validation KL = {kl_beta_4[-1]:.6f}"
)