import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set the page configuration
st.set_page_config(
    page_title="Image Captioning App",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Image Captioning App 📸")
st.write(
    """
    Upload an image and get a caption describing the content of the image.
    """
)

# Sidebar for image upload
st.sidebar.header("Upload Image")
uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Initialize the image captioning model
@st.cache_resource(show_spinner=False)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

def generate_captions(image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, num_beams=5, num_return_sequences=1, max_length=16, min_length=5)
    captions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

# Main content
if uploaded_image is not None:
    # Display uploaded image with reduced size
    image = Image.open(uploaded_image)
    max_image_size = (600, 600)  # Maximum dimensions for the displayed image
    image.thumbnail(max_image_size)  # Reduce image size while maintaining aspect ratio

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate captions
    with st.spinner("Generating caption..."):
        captions = generate_captions(image)

    if captions:
        caption = captions[0]
        st.write("## Caption:")
        # Capitalize the first word of the caption
        capitalized_caption = caption.capitalize()
        st.write(f"{capitalized_caption}")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        text-align: center;
        padding: 10px;
        color: #333;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        <p>Created by Pushpa</p>
    </div>
    """,
    unsafe_allow_html=True,
)