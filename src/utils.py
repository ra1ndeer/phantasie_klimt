import os
import torch
import natsort
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset


class ArtDataset(Dataset):
    """
    Implements a custom Pytorch Dataset comprised
    of images of paintings by Gustav Klimt
    """
    def __init__(self, main_dir, transform):
        super(ArtDataset, self).__init__()

        self.main_dir = main_dir 
        self.transform = transform
        self.images = natsort.natsorted(os.listdir(main_dir))

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns a given item of the dataset
        """
        img_loc = os.path.join(self.main_dir, self.images[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image



class ReverseNormalization(object):
    """
    Implements a simple transform to reverse Normalization
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Applies 'denormalization' to a tensor
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t = (t * s) + m
        return tensor



class Logger(object):
    """
    A simple class to log the results of the training process
    """
    def __init__(self, num_epochs, latent_size):
        self.num_epochs = num_epochs
        self.epochs = [i for i in range(1, self.num_epochs+1)]
        self.kl_loss = list()
        self.rec_loss = list()
        self.last_snapshot = None
        self.random_latent = torch.randn(16, latent_size)
    

    def add_epoch_loss(self, kl, rec):
        """
        Adds a new epoch to the loss
        """
        self.kl_loss.append(kl)
        self.rec_loss.append(rec)


    def print_epoch_loss(self, epoch):
        """
        Prints the a given epoch's training losses
        """
        print("-----------------------------------------------------")
        print(f" Epoch {epoch}; KL loss: {self.kl_loss[epoch-1]:.2f}; Rec loss: {self.rec_loss[epoch-1]:.2f}")
        print("-----------------------------------------------------")


    def add_snapshot(self, snapshot):
        """
        Adds a snapshot to the list
        """
        self.last_snapshot = snapshot


    def plot_history(self, figname):
        """
        Makes a pretty plot with the training history
        """
        fig, ax1 = plt.subplots(figsize=(14, 5))

        # plot the values with twin axes
        ax1.plot(self.epochs, self.kl_loss, marker=" ", linestyle="--", label="KL loss", color="#046E8F", alpha=0.7)
        ax2=ax1.twinx()
        ax2.plot(self.epochs, self.rec_loss, marker=" ", linestyle="--", label="Rec loss", color="#4A7C59", alpha=0.7)
        
        # labelling and tuning the range of values for y1
        ax1.set_ylim((0, max(self.kl_loss)*1.05))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean Absolute Error", color="#046E8F")
        ax1.set_xlim((0, self.num_epochs+1))
        
        # remove ticks from the plot
        ax1.yaxis.set_ticks_position("none")
        ax1.xaxis.set_ticks_position("none")
        ax2.yaxis.set_ticks_position('none') 
        
        # spines are useless and ugly
        for spine, _ in zip(ax1.spines, ax2.spines):
            ax1.spines[spine].set_visible(False)
            ax2.spines[spine].set_visible(False)
        
        # labelling and tuning the range of values for y2
        ax2.set_ylim((0, max(self.rec_loss)*1.05))
        ax2.set_ylabel("Kullback-Leibler Divergence", color="#4A7C59", rotation=-90, labelpad=13)

        ax1.grid(True, which="major", axis="y", linestyle=":", color="#046E8F")
        ax2.grid(True, which="major", axis="y", linestyle=":", color="#4A7C59")

        fig.savefig("output/plots/" + figname + ".png")