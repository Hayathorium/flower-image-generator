import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, z_dim = 64, 100
ckpt_path = "model_state.pth"

# 2. Data
transform = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
dataset = datasets.Flowers102(root="data", split="train", download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 3. Models & Optimizers
from model import Generator, Discriminator
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)
opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

# 4. Resume logic (Load if exists)
start_epoch = 0
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state['G'])
    D.load_state_dict(state['D'])
    opt_G.load_state_dict(state['opt_G'])
    opt_D.load_state_dict(state['opt_D'])
    start_epoch = state['epoch']
    print(f"Resuming from epoch {start_epoch}")

# 5. Training Loop
epoch = start_epoch
while True:
    for i, (real_imgs, _) in enumerate(loader):
        real_imgs = real_imgs.to(device)
        b_size = real_imgs.size(0)
        
        # Train D
        D.zero_grad()
        loss_D = criterion(D(real_imgs).view(-1), torch.ones(b_size, device=device)) + \
                 criterion(D(G(torch.randn(b_size, z_dim, 1, 1, device=device)).detach()).view(-1), torch.zeros(b_size, device=device))
        loss_D.backward(); opt_D.step()

        # Train G
        G.zero_grad()
        loss_G = criterion(D(G(torch.randn(b_size, z_dim, 1, 1, device=device))).view(-1), torch.ones(b_size, device=device))
        loss_G.backward(); opt_G.step()
    with torch.no_grad():
        fake_imgs = G(torch.randn(64, z_dim, 1, 1, device=device))
        save_image(fake_imgs, f"samples/epoch_{epoch+1}.png", normalize=True)
    print(
        f"Epoch [{epoch+1:>4}]"
        f"Loss_D: {loss_D.item():.4f}  "
        f"Loss_G: {loss_G.item():.4f}"
    )
        
    # Save checkpoint at the end of every 10 epoch
    if epoch % 10 == 0:
        torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(), 'epoch': epoch + 1}, ckpt_path)
        print(f"Epoch {epoch+1} completed and saved.")
    epoch += 1