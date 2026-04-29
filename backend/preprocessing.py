import torch
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image

class Preprocessor:
    def __init__(self, device='cpu'):
        self.device = device
        # Initialize MTCNN for face detection and alignment
        # image_size=112 is required for InsightFace/ArcFace
        self.mtcnn = MTCNN(
            image_size=112, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], 
            factor=0.709, 
            post_process=True,
            device=self.device
        )

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """
        Detects a face in the image, aligns it, resizes to 112x112, and normalizes it.
        Args:
            image (PIL.Image): Input image.
        Returns:
            torch.Tensor: Preprocessed face tensor of shape (1, 3, 112, 112)
        """
        # Ensure image is in RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # mtcnn returns a normalized tensor of shape (3, 112, 112)
        face_tensor = self.mtcnn(image)
        
        if face_tensor is None:
            raise ValueError("No face detected in the image.")
            
        # Add batch dimension
        return face_tensor.unsqueeze(0)
