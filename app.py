import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torchvision.models as models

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your model
@st.cache_resource  # Cache the model for efficiency
def load_model():
    # Define the model architecture
    model = models.resnet34(pretrained=False)  # Replace with your model architecture (e.g., resnet34)
    
    # Load the saved state dict
    model.load_state_dict(torch.load("plantDisease-resnet34.pth", map_location=device))  # Replace with your model path
    model = model.to(device)  # Move model to GPU if available
    model.eval()  # Set model to evaluation mode
    return model

model = load_model()

# Define your input transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model's input size
    transforms.ToTensor(),         # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Streamlit app title
st.title("PyTorch Model Deployment with Streamlit")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform the image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)  # Move input to GPU if available

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)  # Forward pass
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Apply softmax

    # Display predictions
    st.subheader("Predictions")
    for i, prob in enumerate(probabilities):
        st.write(f"Class {i}: {prob:.4f}")
