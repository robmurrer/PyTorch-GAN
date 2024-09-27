import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def train(model, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def sample(model, num_samples, device):
    with torch.no_grad():
        noise = torch.randn(num_samples, 20).to(device)
        samples = model.decode(noise).view(-1, 1, 28, 28)
    return samples

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def generate_mixed_input_(vae_model, num_samples, mix_ratio, device, output_size=128):
    """
    Generate inputs for a GAN by mixing pure noise with VAE samples, reshaped to the specified output size.
    
    :param vae_model: Trained VAE model
    :param num_samples: Number of samples to generate
    :param mix_ratio: Ratio of VAE sample to pure noise (0 to 1)
    :param device: Device to run the computation on
    :param output_size: Size of the output tensor (default 128)
    :return: Tensor of mixed inputs, reshaped to (num_samples, output_size)
    """
    if not 0 <= mix_ratio <= 1:
        raise ValueError("mix_ratio must be between 0 and 1")

    # Determine the latent dimension from the VAE model
    latent_dim = vae_model.fc_mu.out_features

    # Generate pure noise
    pure_noise = torch.randn(num_samples, latent_dim, device=device)
    
    # Generate VAE samples
    with torch.no_grad():
        vae_model.eval()
        # Encode random MNIST-like inputs
        random_input = torch.randn(num_samples, 784, device=device)
        mu, logvar = vae_model.encode(random_input)
        # Sample from the VAE's latent space
        vae_samples = vae_model.reparameterize(mu, logvar)
    
    # Mix VAE samples with pure noise
    mixed_input = mix_ratio * vae_samples + (1 - mix_ratio) * pure_noise
    
    # Reshape to the desired output size
    mixed_input_reshaped = mixed_input.view(num_samples, -1)
    
    # If the reshaped size is larger than the output_size, truncate it
    if mixed_input_reshaped.size(1) >= output_size:
        return mixed_input_reshaped[:, :output_size]
    
    # If it's smaller, pad with zeros
    else:
        padding_size = output_size - mixed_input_reshaped.size(1)
        return torch.cat([mixed_input_reshaped, torch.zeros(num_samples, padding_size, device=device)], dim=1)

def generate_mixed_input(vae_model, num_samples, mix_ratio, device, output_size=128):
    if not 0 <= mix_ratio <= 1:
        raise ValueError("mix_ratio must be between 0 and 1")

    latent_dim = vae_model.fc_mu.out_features
    pure_noise = torch.randn(num_samples, latent_dim, device=device)
    
    with torch.no_grad():
        vae_model.eval()
        random_input = torch.randn(num_samples, 784, device=device)
        mu, logvar = vae_model.encode(random_input)
        vae_samples = vae_model.reparameterize(mu, logvar)
    
    mixed_input = mix_ratio * vae_samples + (1 - mix_ratio) * pure_noise
    mixed_input_reshaped = mixed_input.view(num_samples, -1)
    
    if mixed_input_reshaped.size(1) >= output_size:
        return mixed_input_reshaped[:, :output_size]
    else:
        padding_size = output_size - mixed_input_reshaped.size(1)
        return torch.cat([mixed_input_reshaped, torch.zeros(num_samples, padding_size, device=device)], dim=1)

#def get_vae_mnist_noise_model():
def get_vae_model(device):
    #loaded_model = load_model(VAE(784, 20), 'vae_model.pth', device) 
    loaded_model = load_model(VAE(784, 20), 'vae_model_100iter.pth', device) 
    return loaded_model

def main_train(name='vae_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = VAE(784, 20).to(device)
    optimizer = optim.Adam(model.parameters())
    
    train(model, dataloader, optimizer, epochs=100, device=device)
    
    save_model(model, name) 
    
    loaded_model = load_model(VAE(784, 20), name, device)
    
    new_samples = sample(loaded_model, num_samples=32, device=device)
    
    os.makedirs('generated_samples', exist_ok=True)
    for i, sample_tensor in enumerate(new_samples):
        sample_image = transforms.ToPILImage()(sample_tensor.cpu())
        sample_image.save(f'generated_samples/{name}_sample_{i}.png')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaded_model = get_vae_model(device) 
    
    # Generate mixed inputs for different mix ratios
    mix_ratios = [0, 0.25, 0.5, 0.75, 1]
    num_samples = 16
    output_size = 128

    for mix_ratio in mix_ratios:
        mixed_inputs = generate_mixed_input(loaded_model, num_samples, mix_ratio, device, output_size)
        
        print(f"Generated mixed inputs for ratio {mix_ratio}")
        print(f"Shape of mixed inputs: {mixed_inputs.shape}")
        
        # Since we can't directly visualize 128-dimensional data, we'll print some statistics
        print(f"Mean: {mixed_inputs.mean():.4f}")
        print(f"Std Dev: {mixed_inputs.std():.4f}")
        print(f"Min: {mixed_inputs.min():.4f}")
        print(f"Max: {mixed_inputs.max():.4f}")
        print("---")

if __name__ == "__main__":
    #main()
    main_train("vae_model_100iter.pth")