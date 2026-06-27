import torch
from torchvision.utils import save_image
from model import Generator

# 1. Configuration
output_file = "generated_flowers.png"
n_images = 64
z_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Model
G = Generator(z_dim=z_dim).to(device)
state = torch.load("model_state.pth", map_location=device)
G.load_state_dict(state['G'])
G.eval()

# 3. Generate
with torch.no_grad():
    z = torch.randn(n_images, z_dim, 1, 1, device=device)
    fake_imgs = G(z)
    save_image(fake_imgs, output_file, normalize=True)

print(f"Generated {n_images} images and saved to {output_file}")