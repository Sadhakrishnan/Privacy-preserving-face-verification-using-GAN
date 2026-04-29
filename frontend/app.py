import streamlit as st
import requests
from PIL import Image

# Configure Streamlit page
st.set_page_config(
    page_title="Privacy-Preserving Face Verification",
    page_icon="🛡️",
    layout="centered"
)

# Backend API URL
API_URL = "http://localhost:8000/verify"

st.title("🛡️ Privacy-Preserving Face Verification")
st.markdown("""
This system uses a Hybrid GAN + Differential Privacy framework to verify identity 
while preventing face reconstruction and attribute inference attacks.
""")

st.sidebar.header("Settings")
epsilon = st.sidebar.slider(
    "Privacy Budget ($\epsilon$)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Lower epsilon = More noise (Higher Privacy, Lower Accuracy). Higher epsilon = Less noise (Lower Privacy, Higher Accuracy)."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image 1")
    file1 = st.file_uploader("Upload first face image", type=["jpg", "jpeg", "png"], key="file1")
    if file1:
        img1 = Image.open(file1)
        st.image(img1, use_column_width=True)

with col2:
    st.subheader("Image 2")
    file2 = st.file_uploader("Upload second face image", type=["jpg", "jpeg", "png"], key="file2")
    if file2:
        img2 = Image.open(file2)
        st.image(img2, use_column_width=True)

if st.button("Verify Identity", use_container_width=True):
    if not file1 or not file2:
        st.error("Please upload both images to proceed.")
    else:
        with st.spinner("Processing with Privacy-Preserving Framework..."):
            try:
                # Prepare files for multipart/form-data
                files = {
                    'file1': (file1.name, file1.getvalue(), file1.type),
                    'file2': (file2.name, file2.getvalue(), file2.type)
                }
                data = {
                    'epsilon': epsilon
                }
                
                # Make POST request to FastAPI backend
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()
                    st.subheader("Verification Result")
                    
                    # Display metrics nicely
                    met1, met2, met3 = st.columns(3)
                    
                    if result["match"]:
                        met1.success("MATCH ✅")
                    else:
                        met1.error("NO MATCH ❌")
                        
                    met2.metric("Similarity Score", f"{result['similarity_score']:.4f}")
                    met3.metric("Privacy Level ($\epsilon$)", f"{result['epsilon_used']}")
                    
                    st.info(f"Threshold for matching is ~0.35. The current score is {result['similarity_score']:.4f}.")
                else:
                    st.error(f"Error from server: {response.json().get('detail', 'Unknown error')}")
            
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend server. Please ensure the FastAPI backend is running on http://localhost:8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
