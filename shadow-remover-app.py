import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image

# -------------------------------
# Shadow detection
# -------------------------------
def detect_shadows(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    _, shadow_mask = cv2.threshold(enhanced_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shadow_mask = cv2.bitwise_not(shadow_mask)

    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    shadow_mask = cv2.GaussianBlur(shadow_mask, (11, 11), 0)

    return shadow_mask


# -------------------------------
# Shadow removal
# -------------------------------
def remove_shadows(image, shadow_mask):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    norm_mask = shadow_mask.astype(float) / 255.0

    mean_shadow = np.mean(l[norm_mask > 0.5]) if np.any(norm_mask > 0.5) else 0
    mean_non_shadow = np.mean(l[norm_mask < 0.5]) if np.any(norm_mask < 0.5) else 0

    if mean_shadow == 0 or mean_non_shadow == 0:
        return image

    illum_ratio = (mean_non_shadow + 1e-6) / (mean_shadow + 1e-6)
    illum_map = l.astype(float) * (1 + norm_mask * (illum_ratio - 1))

    illum_map = np.clip(illum_map, 0, 255).astype(np.uint8)

    result = cv2.merge([illum_map, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    # Optional guided filter (needs contrib package)
    # result = cv2.ximgproc.guidedFilter(image, result, 10, 1e-6)

    return result


# -------------------------------
# Convert image (OpenCV ↔ PIL)
# -------------------------------
def cv2_to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)


def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Shadow Remover", layout="wide")
    st.title("🖼️ Shadow Remover App")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load uploaded image
        pil_image = Image.open(uploaded_file).convert("RGB")
        image = pil_to_cv2(pil_image)

        # Process
        shadow_mask = detect_shadows(image)
        result = remove_shadows(image, shadow_mask)

        # Convert for display
        mask_pil = Image.fromarray(shadow_mask)
        result_pil = cv2_to_pil(result)

        # Show images side by side
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original")
            st.image(pil_image, use_container_width=True)
        with col2:
            st.subheader("Shadow Mask")
            st.image(mask_pil, use_container_width=True)
        with col3:
            st.subheader("Shadow Removed")
            st.image(result_pil, use_container_width=True)

        # Download button
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="📥 Download Result",
            data=byte_im,
            file_name="shadow_removed.png",
            mime="image/png"
        )


if __name__ == "__main__":
    main()

