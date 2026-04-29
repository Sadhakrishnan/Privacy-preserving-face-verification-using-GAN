import torch
import torch.nn.functional as F
import numpy as np
# insightface uses onnxruntime under the hood for its inference
from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self, device='cpu'):
        # For GPU, device could be something like 'cuda:0'. 
        # Insightface FaceAnalysis uses providers=['CUDAExecutionProvider'] or ['CPUExecutionProvider']
        provider = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
        
        # Initialize ArcFace model from InsightFace
        # We only need the recognition (embedding) part, not detection since MTCNN handles that
        # But FaceAnalysis is a convenient wrapper. We'll disable detection inside it if possible,
        # or use a raw arcface model. For simplicity, let's use the default recognizer.
        self.app = FaceAnalysis(name='buffalo_l', providers=provider)
        # Prepare without detector to avoid double-detection overhead if we just want embeddings.
        # Actually buffalo_l contains both. Let's just use it to extract embedding from the cropped face.
        self.app.prepare(ctx_id=0 if device == 'cpu' else 1, det_size=(112, 112))

    def get_embedding(self, face_image_np: np.ndarray) -> torch.Tensor:
        """
        Extracts a 512-dimensional embedding using InsightFace.
        Args:
            face_image_np (np.ndarray): Cropped face image as a numpy array in BGR format (H, W, 3).
                                        InsightFace expects BGR images.
        Returns:
            torch.Tensor: Normalized embedding of shape (1, 512)
        """
        # insightface app.get() expects a BGR numpy array
        # Since we use MTCNN for cropping, we might need to directly pass the crop to the recognizer.
        # However, FaceAnalysis expects full images to run detection first.
        # To bypass detection and directly get embedding:
        # We can extract the recognizer model from the app.
        recognizer = self.app.models['recognition']
        
        # Recognizer expects input in BGR, transposed to (1, 3, 112, 112), normalized
        # The exact preprocessing is usually handled by the model wrapper
        embedding = recognizer.get_feat(face_image_np)
        
        # Convert to torch tensor
        embedding_tensor = torch.tensor(embedding).unsqueeze(0)
        
        # Normalize just in case
        embedding_tensor = F.normalize(embedding_tensor, p=2, dim=1)
        
        return embedding_tensor
