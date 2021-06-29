import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from pytorch_msssim import MS_SSIM

from src.model import VariationalAutoencoder
from src.losses import KullbackLeiblerDivergence
from src.utils import ArtDataset, ReverseNormalization, Logger



def main():

    image_size = (256, 256)
    imgs_path = "klimt"
    batch_size = 256
    num_epochs = 1500
    learning_rate = 2*1e-3
    init_channels = 64
    hidden_size = 200
    latent_size = 64

    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # loading the data
    data = ArtDataset(
        main_dir=imgs_path, 
        transform=transform)
    # creating a dataloader for the training process
    data_loader = DataLoader(
        dataset=data, 
        batch_size=batch_size, 
        shuffle=True)

    # instantiating the variational autoencoder
    vae = VariationalAutoencoder(
        init_channels=init_channels, 
        hidden_size=hidden_size, 
        latent_size=latent_size)
    
    # the VAE loss is composed of: KL divergence + reconstruction error
    kl_div_loss = KullbackLeiblerDivergence()
    rec_l1_loss = nn.L1Loss(reduction="sum")
    
    # any optimizer should work, Adam just works better (on average)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # initialize the logger to collect training metrics
    historian = Logger(num_epochs=num_epochs, latent_size=latent_size)

    # training loop
    for e in range(1, num_epochs+1):

        epoch_kl = 0
        epoch_rec = 0

        for img_batch in data_loader:
            optimizer.zero_grad()
            mu, logvar, x_tilde = vae(img_batch)
            
            loss1 = kl_div_loss(mu, logvar)
            loss2 = rec_l1_loss(x_tilde, img_batch) / img_batch.size(0)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            epoch_kl += loss1.item()
            epoch_rec += loss2.item()

        historian.add_epoch_loss(epoch_kl / len(data), epoch_rec / len(data))
        historian.print_epoch_loss(e)

        with torch.no_grad():    
            historian.add_snapshot(vae.decoder(historian.random_latent))
            grid_img = torchvision.utils.make_grid(historian.last_snapshot, nrow=4)
            trans = transforms.ToPILImage()
            show_img = trans(grid_img)
            show_img.save(f"output/images/{e}.png")

        if e % 100 == 0:
            torch.save(vae, f"output/checkpoints/{e}.pth")
    
    historian.plot_history("test")
            


if __name__ == "__main__":
    main()