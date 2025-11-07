# =========================================================
# 🧠 Streamlit App — Pix2Pix Mask → Defect Image Generator
# =========================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

# =========================================================
# ⚙️ APP CONFIG
# =========================================================
st.set_page_config(page_title="Pix2Pix Leather Defect Generator", layout="wide")
st.title("🧠 Pix2Pix — Mask to Defective Leather Image Generator")
st.markdown("Upload a **mask image** and choose which trained model to use.")

# =========================================================
# 📁 MODEL PATHS (update paths as needed)
# =========================================================
MODEL_PATHS = {
    "Poke Defect": "pix2pix_mask2defect_poke.h5",
    "Glue Defect": "pix2pix_mask2defect_glue.h5",
    "Color Defect": "pix2pix_mask2defect_color.h5"
}

# =========================================================
# 📦 LOAD MODEL (cached for performance)
# =========================================================
@st.cache_resource
def load_generator(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# =========================================================
# 🧩 IMAGE PREPROCESSING FUNCTION
# =========================================================
def preprocess_mask(uploaded_image):
    image = Image.open(uploaded_image).convert("L")  # Convert to grayscale
    image = image.resize((256, 256))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=(0, -1))  # (1, 256, 256, 1)
    return image

# =========================================================
# 🎨 POSTPROCESS OUTPUT IMAGE
# =========================================================
def postprocess_image(predicted_image):
    predicted_image = np.clip(predicted_image, -1, 1)
    predicted_image = ((predicted_image + 1) / 2 * 255).astype(np.uint8)
    return predicted_image

# =========================================================
# 🧠 MAIN APP
# =========================================================
st.sidebar.header("⚙️ Choose Model")
model_choice = st.sidebar.radio("Select Pix2Pix Model:", list(MODEL_PATHS.keys()))

generator = load_generator(MODEL_PATHS[model_choice])
if generator is None:
    st.stop()

# =========================================================
# 📤 IMAGE UPLOAD
# =========================================================
st.subheader(f"📸 Generate using {model_choice}")
uploaded_file = st.file_uploader("Upload a Mask Image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded mask
    st.image(uploaded_file, caption="Uploaded Mask Image", width=300)

    if st.button("🔮 Generate Defective Image"):
        with st.spinner("Generating image... Please wait ⏳"):
            input_image = preprocess_mask(uploaded_file)
            predicted = generator.predict(input_image)[0]
            predicted_image = postprocess_image(predicted)

            # ✅ FIX: No color conversion — model outputs RGB already
            st.success("✅ Image generation complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="🩶 Input Mask", width=300)
            with col2:
                st.image(predicted_image, caption=f"🎨 Generated {model_choice} Image", width=300)

            # ✅ FIX: Save directly using PIL (no cv2 needed)
            save_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            Image.fromarray(predicted_image).save(save_path.name)
            with open(save_path.name, "rb") as f:
                st.download_button("💾 Download Generated Image", f, file_name="generated_defect.png")

else:
    st.info("👆 Upload a mask image to get started.")

# =========================================================
# ✅ END
# =========================================================
