import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os

# Example model class (replace with your actual model class)
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model layers here
        self.layer1 = torch.nn.Linear(224 * 224 * 3, 128)  # Example layer
        self.layer2 = torch.nn.Linear(128, 10)  # Example layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Function to load the model
def load_model(model_path):
    model = MyModel()  # Initialize your model
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  # Load the state dictionary
    model.eval()  # Set the model to evaluation mode
    return model

# Function to preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

# Main function to run the Streamlit app
def main():
    # Set the page configuration
    st.set_page_config(
        page_title="Plant Disease Classification",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load custom CSS
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Add a title and description
    st.title("🌿 Plant Disease Classification")
    st.markdown("Upload an image of a plant leaf to classify its disease.")

    # Add a sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("This app uses a deep learning model to classify plant diseases from images.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Load the model
        model_path = 'disease_classifier.pth'  # Replace with your model file name
        if os.path.exists(model_path):
            model = load_model(model_path)

            # Make prediction
            with torch.no_grad():
                prediction = model(processed_image)
                predicted_class = prediction.argmax(dim=1).item()

            # Display the prediction
            st.success(f"Prediction: {predicted_class}")  # Adjust based on your model's output
        else:
            st.error("Model file not found. Please ensure the model file is in the same directory.")

if __name__ == "__main__":
    main()