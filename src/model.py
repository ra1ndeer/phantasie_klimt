import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    """
    Implements a convolutional Variational Autoencoder
    """
    def __init__(
        self, 
        image_channels=3, 
        init_channels=64, 
        kernel_size=4, 
        stride=2, 
        padding=1, 
        use_bias=False, 
        hidden_size=64, 
        latent_size=16):
        
        super(VariationalAutoencoder, self).__init__()

        self.image_channels = image_channels
        self.init_channels = init_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = InferenceNetwork(
            image_channels=self.image_channels,
            init_channels=self.init_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            use_bias=self.use_bias,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size
        )

        self.decoder = RecognitionNetwork(
            latent_size=self.latent_size,
            init_channels=self.init_channels * 8,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            use_bias=self.use_bias
        )


    def reparametrize(self, mu, logvar):
        """
        The reparametrization trick
        """
        eps = torch.randn_like(logvar)
        return mu + (eps * torch.exp(logvar))


    def forward(self, x):
        """
        VAE forward step
        """
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_tilde = self.decoder(z)
        return mu, logvar, x_tilde



class InferenceNetwork(nn.Module):
    """
    Implements a simple convolutional encoder for a VAE, meaning 
    that it outputs two vectors instead of a single vector.
    """
    def __init__(self, image_channels, init_channels, kernel_size, stride, padding, use_bias, hidden_size, latent_size):
        super(InferenceNetwork, self).__init__()

        self.image_channels = image_channels
        self.init_channels = init_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv_net = nn.Sequential(
            nn.Conv2d(
                in_channels=self.image_channels, 
                out_channels=self.init_channels, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels),
            nn.LeakyReLU(
                negative_slope=0.2, 
                inplace=True),
            nn.Conv2d(
                in_channels=self.init_channels, 
                out_channels=self.init_channels*2, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels*2),
            nn.LeakyReLU(
                negative_slope=0.2, 
                inplace=True),
            nn.Conv2d(
                in_channels=self.init_channels*2, 
                out_channels=self.init_channels*4, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels*4),
            nn.LeakyReLU(
                negative_slope=0.2, 
                inplace=True),
            nn.Conv2d(
                in_channels=self.init_channels*4, 
                out_channels=self.init_channels*8, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels*8),
            nn.LeakyReLU(
                negative_slope=0.2, 
                inplace=True),
            nn.Conv2d(
                in_channels=self.init_channels*8, 
                out_channels=self.init_channels*16, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels*16),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True),
            nn.Conv2d(
                in_channels=self.init_channels*16, 
                out_channels=8, 
                kernel_size=self.kernel_size, 
                stride=1, 
                padding=0, 
                bias=self.use_bias),
            nn.Flatten()
        )

        self.fc = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.hidden_size,
            bias=True)
        self.hidden2mu = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.latent_size, 
            bias=True)
        self.hidden2logvar = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.latent_size,
            bias=True)


    def forward(self, x):
        """
        Encoder forward step
        """
        hidden = torch.tanh(self.fc(self.conv_net(x)))
        return self.hidden2mu(hidden), self.hidden2logvar(hidden)



class RecognitionNetwork(nn.Module):
    """
    Implements a simple convolutional decoder for a VAE.
    """
    def __init__(self, latent_size, init_channels, kernel_size, stride, padding, use_bias):
        super(RecognitionNetwork, self).__init__()
        
        self.latent_size = latent_size
        self.init_channels = init_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias

        self.network = nn.Sequential(
            nn.Unflatten(-1, (self.latent_size, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=self.latent_size, 
                out_channels=self.init_channels, 
                kernel_size=self.kernel_size, 
                stride=1, 
                padding=0, 
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.init_channels, 
                out_channels=self.init_channels // 2, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.init_channels // 2, 
                out_channels=self.init_channels // 4, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.init_channels // 4, 
                out_channels=self.init_channels // 8, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.init_channels // 8, 
                out_channels=self.init_channels // 16, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels // 16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.init_channels // 16,
                out_channels=self.init_channels // 32,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.use_bias),
            nn.BatchNorm2d(num_features=self.init_channels // 32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.init_channels // 32,
                out_channels=3,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.use_bias),
            nn.Sigmoid()
        )


    def forward(self, z):
        """
        Decoder forward step
        """
        return self.network(z)