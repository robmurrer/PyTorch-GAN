import argparse
import os
import secrets
import time
from datetime import datetime, timedelta
import numpy as np
import math
from collections import deque
import torchvision.transforms as transforms
from torchvision.utils import save_image
import json

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt

import vae_image_sampler as vaes

DEBUG = False
#DEBUG = True #uncomment this line to debug
iters = 1
epochs = 200 

if DEBUG:
    epochs = 2 
    iters = 3 # corner analysis

def minutes_in_day():
    now = datetime.now()
    minutes = now.hour * 60 + now.minute
    return f"{minutes:04d}"

ruid_ = datetime.now().strftime("%Y.%m.%d.") + minutes_in_day()
ruid_dir = "doe/vae/" + ruid_ + "/"

os.makedirs(ruid_dir, exist_ok=False) # throw if we collide :)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=epochs, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--vae_mix_ratio", type=float, default=1.0, help="ratio of VAE samples to pure noise")
opt = parser.parse_args()
print(opt)
with open(file=os.path.join(ruid_dir, "hyper.params"), mode="w") as f:
    f.write(str(opt))
    f.close()   

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Lists to store loss values for plotting
g_losses = []
d_losses = []
iterations = []
epoch_times = []


#g_losses = deque(maxlen=10 * len(dataloader))
#d_losses = deque(maxlen=10 * len(dataloader))
best_g_loss = float('inf')
best_d_loss = float('inf')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model = vaes.get_vae_model(device)
for epoch in range(opt.n_epochs):
    epoch_start_time = time.time()
    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Tensor(imgs.size(0), 1).fill_(1.0)
        fake = Tensor(imgs.size(0), 1).fill_(0.0)
        real_imgs = imgs.type(Tensor)
        optimizer_G.zero_grad()

        # the gambit
        #z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
        z = vaes.generate_mixed_input(vae_model, imgs.size(0), mix_ratio=opt.vae_mix_ratio, device=device, output_size=opt.latent_dim)
        
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()


         # Calculate and save epoch time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        batches_done = epoch * len(dataloader) + i
        
        # Store losses and iteration number
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        iterations.append(batches_done)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        if batches_done % opt.sample_interval == 0:
            # Create and save the plot
            plt.figure(figsize=(10, 5))
            plt.plot(iterations, g_losses, label='Generator Loss')
            plt.plot(iterations, d_losses, label='Discriminator Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Generator and Discriminator Loss')
            plt.legend()
            
            # Generate a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plot_iter_{batches_done}_{timestamp}.png"
            
            plt.savefig(ruid_dir + filename)
            plt.close()

            # Save generated images (as before)
            save_image(gen_imgs.data[:25], ruid_dir + "%d.png" % batches_done, nrow=5, normalize=True)

            # Save best models
            if np.mean(list(g_losses)[-len(dataloader):]) < best_g_loss:
                best_g_loss = np.mean(list(g_losses)[-len(dataloader):])
                torch.save(generator.state_dict(), ruid_dir + 'best_generator.pth')

            if np.mean(list(d_losses)[-len(dataloader):]) < best_d_loss:
                best_d_loss = np.mean(list(d_losses)[-len(dataloader):])
                torch.save(discriminator.state_dict(), ruid_dir + 'best_discriminator.pth')



# save it all!
plt.figure(figsize=(10, 5))
plt.plot(iterations, g_losses, label='Generator Loss')
plt.plot(iterations, d_losses, label='Discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Loss (Full Training)')
plt.legend()

filename = f"plot_final_{ruid_}.png"
plt.savefig(ruid_dir + filename)
plt.close()

avg_epoch_time = np.mean(epoch_times)


# Calculate average loss over last 10 epochs
avg_g_loss = np.mean(list(g_losses))
avg_d_loss = np.mean(list(d_losses))

# Prepare results
results = {
    "avg_generator_loss": avg_g_loss,
    "avg_discriminator_loss": avg_d_loss,
    "best_generator_loss": best_g_loss,
    "best_discriminator_loss": best_d_loss,
    "best_generator_path": "best_generator.pth",
    "best_discriminator_path": "best_discriminator.pth",
    "avg_epoch_time_seconds": avg_epoch_time,
}

# Save results to JSON file
with open(ruid_dir + 'gan_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Write timing information to text file
    with open(ruid_dir + 'training_times.txt', 'w') as f:
        f.write(f"Average iter time: {timedelta(seconds=avg_epoch_time)}\n")
        f.write("Individual iter times:\n")
        for i, t in enumerate(epoch_times):
            f.write(f"iter {i}: {timedelta(seconds=t)}\n")

    print("Training completed. Results saved to gan_results.json")
    print(f"Best Generator saved to: {results['best_generator_path']}")
    print(f"Best Discriminator saved to: {results['best_discriminator_path']}")
    print(f"Training times saved to: training_times.txt")
    print(f"Average epoch time: {timedelta(seconds=avg_epoch_time)}")
