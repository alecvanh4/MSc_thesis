import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(approximate="tanh"),
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(approximate="tanh"),
        )

    def forward(self, x):
        return self.block(x)


class FiberSeqVAE(nn.Module):
    def __init__(
        self,
        in_channels=2,
        input_length=1024,
        latent_dim=8,
        num_classes=4,
        hidden_dims=(16, 32, 64, 128),
        kernel_sizes=(9, 7, 5, 3),
    ):
        super().__init__()

        assert len(hidden_dims) == 4
        assert len(kernel_sizes) == 4
        assert input_length % 16 == 0

        h1, h2, h3, h4 = hidden_dims
        k1, k2, k3, k4 = kernel_sizes

        self.in_channels = in_channels
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.enc1 = ConvBlock1D(in_channels, h1, kernel_size=k1, stride=2)
        self.enc2 = ConvBlock1D(h1, h2, kernel_size=k2, stride=2)
        self.enc3 = ConvBlock1D(h2, h3, kernel_size=k3, stride=2)
        self.enc4 = ConvBlock1D(h3, h4, kernel_size=k4, stride=2)

        self.feature_channels = h4
        self.feature_length = input_length // 16
        self.flatten_dim = h4 * self.feature_length

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Linear classifier directly on latent space
        self.classifier = nn.Linear(latent_dim, num_classes)

        # Learnable class centers in latent space
        self.class_centers = nn.Parameter(
            torch.randn(num_classes, latent_dim)
        )

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.dec1 = DeconvBlock1D(h4, h3)
        self.dec2 = DeconvBlock1D(h3, h2)
        self.dec3 = DeconvBlock1D(h2, h1)
        self.dec4 = DeconvBlock1D(h1, h1)

        self.final_conv_5mc = nn.Conv1d(h1, 1, kernel_size=1)
        self.final_conv_m6a = nn.Conv1d(h1, 1, kernel_size=1)

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(z.size(0), self.feature_channels, self.feature_length)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        x_5mc = self.final_conv_5mc(x)
        x_m6a = self.final_conv_m6a(x)

        x = torch.cat([x_5mc, x_m6a], dim=1)

        if x.size(-1) > self.input_length:
            x = x[..., :self.input_length]
        elif x.size(-1) < self.input_length:
            x = F.pad(x, (0, self.input_length - x.size(-1)))

        return x

    def center_loss(self, mu, labels):
        labels = labels.view(-1)
        labeled_mask = labels > 0

        if not labeled_mask.any():
            return torch.tensor(0.0, device=mu.device)

        y = labels[labeled_mask].long() - 1
        mu_labeled = mu[labeled_mask]

        centers = self.class_centers[y]

        return ((mu_labeled - centers) ** 2).sum(dim=1).mean()

    def forward(self, x, labels=None):
        mu, logvar = self.encode(x)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        x_recon = self.decode(z)

        class_logits = self.classifier(mu)

        if labels is not None:
            center_loss = self.center_loss(mu, labels)
        else:
            center_loss = None

        return x_recon, class_logits, mu, logvar, center_loss