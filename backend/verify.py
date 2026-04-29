import torch
import torch.nn.functional as F

class VerificationModule:
    def __init__(self, threshold=0.35):
        self.threshold = threshold

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Computes the cosine similarity between two embeddings.
        Args:
            emb1, emb2 (torch.Tensor): Embeddings of shape (1, 512)
        Returns:
            float: Cosine similarity score [-1, 1]
        """
        # F.cosine_similarity returns a tensor, we want the scalar float
        sim = F.cosine_similarity(emb1, emb2, dim=1).item()
        return sim

    def verify(self, emb1: torch.Tensor, emb2: torch.Tensor) -> dict:
        """
        Verifies if two embeddings belong to the same identity.
        Args:
            emb1, emb2 (torch.Tensor): Embeddings of shape (1, 512)
        Returns:
            dict: Contains 'match' (bool) and 'score' (float)
        """
        score = self.compute_similarity(emb1, emb2)
        match = score >= self.threshold
        return {
            "match": match,
            "score": score
        }
