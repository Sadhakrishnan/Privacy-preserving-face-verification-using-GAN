import torch
import torch.nn.functional as F
import numpy as np

class DifferentialPrivacyModule:
    def __init__(self, sensitivity=2.0):
        # Sensitivity of cosine similarity in normalized embeddings is typically bounded
        # Max difference between two vectors in L2 distance on a unit hypersphere is 2
        self.sensitivity = sensitivity

    def add_noise(self, embedding: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
        """
        Adds Gaussian noise to the embedding based on the epsilon parameter.
        Higher epsilon = less noise (less privacy, higher accuracy).
        Lower epsilon = more noise (more privacy, lower accuracy).
        
        Args:
            embedding (torch.Tensor): Input embedding (batch_size, 512).
            epsilon (float): Privacy budget parameter.
            
        Returns:
            torch.Tensor: Noisy, normalized embedding.
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be greater than 0.")
            
        # Standard deviation for Gaussian noise in $(\epsilon, \delta)$-DP
        # Simplified assumption for scale: scale ~ sensitivity / epsilon
        # A true Gaussian mechanism would also depend on delta, but we use a simplified version here.
        scale = self.sensitivity / (epsilon + 1e-5)
        
        # We selectively apply noise. In a real scenario, we might apply it to non-critical dimensions
        # or globally. Here, we apply globally scaled Gaussian noise.
        noise = torch.normal(mean=0.0, std=scale, size=embedding.size(), device=embedding.device)
        
        # Add noise
        noisy_embedding = embedding + noise
        
        # Re-normalize to ensure it remains a valid cosine similarity embedding
        noisy_embedding = F.normalize(noisy_embedding, p=2, dim=1)
        
        return noisy_embedding
