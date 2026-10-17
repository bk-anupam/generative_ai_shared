import torch 
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image
import matplotlib.pyplot as plt

transform1 = transforms.Compose([
    transforms.ToTensor(),
])

#mnist
mnist_train = MNIST(root='./data', train=True, download=False, transform=transform1)
mnist_val = MNIST(root='./data', train=False, download=False, transform=transform1)



train_dataloader=DataLoader(mnist_train, batch_size=32, shuffle=True)
val_dataloader=DataLoader(mnist_val, batch_size=32, shuffle=False)
#encoder

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=16,kernel_size=1,stride=1),
            
        )


    def forward(self,x):
            x=self.encoder(x)
            return x

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(in_channels=16,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid()
        )
    def forward(self,x):
            x=self.decoder(x)
            return x
        



class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.decoder=decoder()
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x


model1 = model()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    model1.parameters(),
    lr=0.001
)
epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
    model1.train()
    total_train_loss = 0

    for batch_idx, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        noisy_x = x + 0.1 * torch.randn_like(x)
        output = model1(noisy_x)
        loss = loss_fn(output, x)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_dataloader)
    model1.eval()

    total_val_loss = 0

    with torch.no_grad():

        for x_val, y_val in val_dataloader:
            noisy_x_val = x_val + 0.1 * torch.randn_like(x_val)

            val_output = model1(noisy_x_val)

            val_loss = loss_fn(val_output, x_val)

            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_dataloader)
    train_losses.append(average_train_loss)
    val_losses.append(average_val_loss)
   
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Validation Loss: {average_val_loss:.4f}"
    )
torch.save(
    model1.state_dict(),
    "autoencoder.pth"
)
plt.figure(figsize=(8, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()
model1.eval()

with torch.no_grad():
    x_val, _ = next(iter(val_dataloader))
    reconstructed = model1(x_val)
    num_images = 5
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(
            x_val[i][0].cpu().numpy(),
            cmap="gray"
        )
        plt.title("Original")
        plt.axis("off")
        plt.subplot(2, num_images, i + 1 + num_images)

        plt.imshow(
            reconstructed[i][0].cpu().numpy(),
            cmap="gray"
        )
        plt.title("Reconstructed")
        plt.axis("off")
    plt.tight_layout()
    plt.show()