import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(Generator, self).__init__()
        # The generator transforms the embedding into a privacy-preserving embedding
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # We add the transformed features to the original embedding (residual connection)
        # to ensure it retains some structural properties, then normalize
        noise_component = self.net(x)
        obfuscated = x + noise_component
        # Re-normalize to unit hypersphere
        return F.normalize(obfuscated, p=2, dim=1)

class PrivacyGAN:
    def __init__(self, input_dim=512, device='cpu'):
        self.device = device
        self.generator = Generator(input_dim=input_dim).to(device)
        self.discriminator = Discriminator(input_dim=input_dim).to(device)
        
        # Load weights here if we had a trained model
        # For demonstration, we'll initialize with random weights and set to eval mode
        self.generator.eval()
        self.discriminator.eval()

    def obfuscate(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Passes the original embedding through the Generator.
        Args:
            embedding (torch.Tensor): Original embedding (batch_size, 512)
        Returns:
            torch.Tensor: Obfuscated embedding (batch_size, 512)
        """
        embedding = embedding.to(self.device)
        with torch.no_grad():
            obfuscated_emb = self.generator(embedding)
        return obfuscated_emb

# Note: In a real scenario, there would be a training script with:
# - Adversarial loss (BCE) for the discriminator
# - Cosine similarity loss for the generator to preserve identity (1 - cos_sim(orig, obf))
