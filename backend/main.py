from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PIL import Image
import numpy as np
import torch
import cv2

# Import custom modules
from backend.preprocessing import Preprocessor
from backend.embedding import FaceEmbedder
from backend.gan_model import PrivacyGAN
from backend.dp_module import DifferentialPrivacyModule
from backend.verify import VerificationModule

app = FastAPI(title="Privacy-Preserving Face Verification API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
preprocessor = Preprocessor(device=device)
embedder = FaceEmbedder(device=device)
gan_model = PrivacyGAN(device=device)
verify_module = VerificationModule(threshold=0.35)

def process_image_to_embedding(image_bytes, epsilon: float):
    # 1. Read Image
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    # 2. Preprocess (MTCNN)
    # MTCNN returns a tensor (1, 3, 112, 112) normalized to [-1, 1]
    try:
        face_tensor = preprocessor.process_image(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # Convert tensor back to numpy BGR [0, 255] for InsightFace
    # Face tensor is [-1, 1], RGB
    face_np = face_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    face_np = ((face_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

    # 3. Extract Embedding (InsightFace)
    # get_embedding expects BGR numpy array
    embedding = embedder.get_embedding(face_bgr)

    # 4. GAN Obfuscation
    obfuscated_embedding = gan_model.obfuscate(embedding)

    # 5. Differential Privacy
    dp_module = DifferentialPrivacyModule()
    secure_embedding = dp_module.add_noise(obfuscated_embedding, epsilon=epsilon)
    
    return secure_embedding

@app.post("/verify")
async def verify_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    epsilon: float = Form(1.0)
):
    # Read files
    img1_bytes = await file1.read()
    img2_bytes = await file2.read()
    
    # Process both images to get secure embeddings
    emb1 = process_image_to_embedding(img1_bytes, epsilon)
    emb2 = process_image_to_embedding(img2_bytes, epsilon)
    
    # Verify
    result = verify_module.verify(emb1, emb2)
    
    return {
        "match": result["match"],
        "similarity_score": round(result["score"], 4),
        "epsilon_used": epsilon
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
