# import streamlit as st
# import SimpleITK as sitk
# import numpy as np
# import cv2
# from PIL import Image
# import io
# import base64
# import os
# import tempfile

# # --- Session State Initialization ---
# if 'points' not in st.session_state:
#     st.session_state.points = [] # List of (x, y, is_positive)
# if 'current_image_to_display' not in st.session_state:
#     st.session_state.current_image_to_display = None # This will be RGB uint8 for display
# if 'original_nifti_slice' not in st.session_state: # For NIfTI, the raw slice before drawing
#     st.session_state.original_nifti_slice = None
# if 'original_2d_image' not in st.session_state: # For 2D, the raw image before drawing
#     st.session_state.original_2d_image = None
# if 'last_click_data' not in st.session_state: # To prevent processing same click
#     st.session_state.last_click_data = None
# if 'current_file_key' not in st.session_state: # To change component key on file change
#     st.session_state.current_file_key = "initial"
# if 'nifti_slice_index' not in st.session_state:
#     st.session_state.nifti_slice_index = 0

# # --- Helper Functions ---

# def ndarray_to_base64_data_url(img_array_rgb_uint8):
#     """Converts a NumPy array (RGB, uint8) to a base64 data URL for PNG."""
#     img_pil = Image.fromarray(img_array_rgb_uint8)
#     buffered = io.BytesIO()
#     img_pil.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     return f"data:image/png;base64,{img_str}"

# def draw_points_on_image(image_array_rgb_uint8, points_list):
#     """Draws points on a copy of the image. Returns a new image array."""
#     img_with_points = image_array_rgb_uint8.copy()
#     for x, y, is_positive in points_list:
#         color = (0, 255, 0) if is_positive else (255, 0, 0)  # Green for positive, Red for negative
#         # Radius of the circle
#         radius = 5
#         # Thickness of the circle outline (negative for filled circle)
#         thickness = -1
#         cv2.circle(img_with_points, (int(x), int(y)), radius, color, thickness)
#     return img_with_points

# # --- Custom JavaScript for Clickable Image ---
# CLICKABLE_IMAGE_HTML = """
# <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
#     <img id="clickable_image_id" src="{image_data_url}" style="cursor: crosshair; max-width: 100%; max-height: 600px; object-fit: contain;">
# </div>
# <script>
#     const img = document.getElementById('clickable_image_id');
#     img.onload = function() {{
#         img.onclick = function(event) {{
#             const rect = event.target.getBoundingClientRect();
#             const x = event.clientX - rect.left;
#             const y = event.clientY - rect.top;
#             const clickData = {{
#                 normX: x / event.target.width,
#                 normY: y / event.target.height,
#                 timestamp: new Date().getTime()
#             }};
#             window.parent.streamlitApi.setComponentValue(clickData);
#         }};
#     }};
#     if (img.complete) {{
#         img.onload();
#     }}
# </script>
# """

# def main():
#     st.set_page_config(layout="wide")
#     st.title("Interactive Medical Image Viewer & Point Annotation")

#     # --- Sidebar Controls ---
#     st.sidebar.header("Controls")
#     uploaded_file = st.sidebar.file_uploader("Upload Image (.nii.gz, .png, .jpg)", type=["nii.gz", "png", "jpg"])
    
#     if uploaded_file is not None:
#         # Generate a new key for the file to reset components if file changes
#         new_file_key = uploaded_file.name + str(uploaded_file.size)
#         if st.session_state.current_file_key != new_file_key:
#             st.session_state.current_file_key = new_file_key
#             st.session_state.points = [] # Reset points for new image
#             st.session_state.last_click_data = None
#             st.session_state.current_image_to_display = None
#             st.session_state.original_nifti_slice = None
#             st.session_state.original_2d_image = None
#             st.session_state.nifti_slice_index = 0 # Reset slice index

#     point_type_str = st.sidebar.radio("Point Type:", ("Positive (FG)", "Negative (BG)"), key="point_type_selector_" + st.session_state.current_file_key)
#     is_positive_click = True if point_type_str == "Positive (FG)" else False

#     # --- Image Loading and Processing ---
#     display_image_prepared = None # This will be the RGB uint8 image passed to JS
#     current_slice_for_nifti_raw = None # Store the raw NIfTI slice if applicable

#     if uploaded_file is not None:
#         file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
#         with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             tmp_file_path = tmp_file.name

#         try:
#             if file_extension == ".nii.gz":
#                 itk_img = sitk.ReadImage(tmp_file_path)
#                 img_array_3d = sitk.GetArrayFromImage(itk_img) # z, y, x

#                 if 'nifti_slice_index' not in st.session_state or st.session_state.nifti_slice_index >= img_array_3d.shape[0]:
#                     st.session_state.nifti_slice_index = img_array_3d.shape[0] // 2
                
#                 st.session_state.nifti_slice_index = st.sidebar.slider(
#                     "Slice", 
#                     0, 
#                     img_array_3d.shape[0] - 1, 
#                     st.session_state.nifti_slice_index, 
#                     key="slice_slider_" + st.session_state.current_file_key
#                 )
#                 current_slice_for_nifti_raw = img_array_3d[st.session_state.nifti_slice_index, :, :]
                
#                 # Normalize and convert to RGB for display
#                 slice_norm = ((current_slice_for_nifti_raw - current_slice_for_nifti_raw.min()) / 
#                               (current_slice_for_nifti_raw.max() - current_slice_for_nifti_raw.min() + 1e-6) * 255).astype(np.uint8)
#                 display_image_prepared = cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2RGB)
#                 st.session_state.original_nifti_slice = current_slice_for_nifti_raw # Store raw for potential model use

#             elif file_extension in [".png", ".jpg", ".jpeg"]:
#                 img_pil = Image.open(tmp_file_path)
#                 img_np = np.array(img_pil)
#                 if len(img_np.shape) == 2: # Grayscale
#                     display_image_prepared = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
#                 elif len(img_np.shape) == 3 and img_np.shape[2] == 4: # RGBA
#                      display_image_prepared = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
#                 elif len(img_np.shape) == 3 and img_np.shape[2] == 3: # RGB
#                      display_image_prepared = img_np
#                 else:
#                     st.error(f"Unsupported image format or channel count: {img_np.shape}")
#                     display_image_prepared = None
#                 st.session_state.original_2d_image = display_image_prepared # Store for potential model use (already RGB)
            
#             if display_image_prepared is not None:
#                  st.session_state.current_image_to_display = display_image_prepared

#         except Exception as e:
#             st.error(f"Error loading or processing image: {e}")
#             st.session_state.current_image_to_display = None
#         finally:
#             os.remove(tmp_file_path) # Clean up temp file

#     # --- Display Image and Handle Clicks ---
#     if st.session_state.current_image_to_display is not None:
#         base_image_for_html = st.session_state.current_image_to_display
        
#         # Draw current points onto a copy of the base image for display
#         image_with_points_display = draw_points_on_image(base_image_for_html, st.session_state.points)
        
#         # Convert image to base64
#         b64_img_with_points = ndarray_to_base64_data_url(image_with_points_display)
        
#         # Display the image using custom HTML and JavaScript
#         click_data = st.components.v1.html(
#             CLICKABLE_IMAGE_HTML.format(image_data_url=b64_img_with_points),
#             height=600,
#             key="clickable_image"
#         )

#         # Process click data
#         if click_data:
#             norm_x = click_data.get('normX')
#             norm_y = click_data.get('normY')
#             if norm_x is not None and norm_y is not None:
#                 img_height, img_width = base_image_for_html.shape[:2]
#                 actual_x = norm_x * img_width
#                 actual_y = norm_y * img_height
#                 is_positive = st.sidebar.radio("Point Type", ("Positive", "Negative"), key=f"point_type_{st.session_state.nifti_slice_index}") == "Positive"
#                 st.session_state.click_processor.add_point(actual_x, actual_y, is_positive)
#                 st.rerun()

#         # Display the original slice and the image with points side by side
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(base_image_for_html, caption="Original Slice", use_column_width=True)
#         with col2:
#             st.image(image_with_points_display, caption="Image with Points", use_column_width=True)

#         # Handle clicks using Streamlit's button or click event
#         st.sidebar.subheader("Annotation Controls")
#         if st.sidebar.button("Clear All Points", key="clear_pts_" + st.session_state.current_file_key):
#             st.session_state.points = []
#             st.session_state.last_click_data = None
#             st.rerun()
        
#         if st.session_state.points:
#             st.sidebar.subheader("Current Points:")
#             for i, (x, y, is_pos) in enumerate(st.session_state.points):
#                 pt_type = "Positive" if is_pos else "Negative"
#                 st.sidebar.markdown(f"{i+1}. ({int(x)}, {int(y)}) - <span style='color:green;'>{pt_type}</span>" if is_pos 
#                                     else f"{i+1}. ({int(x)}, {int(y)}) - <span style='color:red;'>{pt_type}</span>", 
#                                     unsafe_allow_html=True)
#         else:
#             st.sidebar.info("No points added yet. Click on the image above.")
        
#     elif uploaded_file is None and not st.session_state.current_image_to_display:
#         st.info("Please upload an image file using the sidebar to begin.")

#     # --- Save Functionality ---
#     st.sidebar.subheader("Save Options")
#     if st.sidebar.button("Save Mask", key="save_mask_" + st.session_state.current_file_key):
#         if st.session_state.current_image_to_display is not None:
#             mask_path = os.path.join(tempfile.gettempdir(), "saved_mask.png")
#             cv2.imwrite(mask_path, image_with_points_display)
#             st.sidebar.success(f"Mask saved to {mask_path}")
#         else:
#             st.sidebar.error("No image to save.")

# if __name__ == "__main__":
#     main() 



import streamlit as st
import SimpleITK as sitk
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
import os
import sys
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
import hydra
from streamlit_drawable_canvas import st_canvas
import time

from kprism.inference.inference import InferenceCore

class ClickProcessor:
    def __init__(self):
        self.points = []
        
    def add_point(self, x, y, is_positive=True):
        self.points.append((x, y, is_positive))
        
    def clear_points(self):
        self.points = []
        
    def get_points(self):
        return self.points

# Function to load and visualize a NIfTI file
def load_and_visualize_nii(file_path):
    try:
        itk_img = sitk.ReadImage(file_path)
        img_array = sitk.GetArrayFromImage(itk_img)
        return img_array
    except Exception as e:
        st.error(f"Error loading NIfTI file: {e}")
        return None

# Function to prepare image for model input
def prepare_image_for_model(image):
    # Ensure the image is float and has the right shape
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = np.stack([image, image, image], axis=2)
    
    # Normalize image if needed
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
        
    # Convert to tensor with shape [C, H, W]
    if len(image.shape) == 3:
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    else:
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        
    return image_tensor

# Function for model inference
def perform_inference(model, image_tensor, points, unique_label=1):
    with torch.no_grad():
        # Prepare batched input
        batch_data = [{
            "image": image_tensor.unsqueeze(0),  # Add batch dimension
            "points": points,
            "unique_label": unique_label
        }]
        
        # Perform inference
        processed_results_dict, point_dict = model(batch_data)
        
        # Get the result from the last iteration
        last_iter = max(processed_results_dict.keys())
        result = processed_results_dict[last_iter][0]  # Get result for first batch item
        
        return result, point_dict[last_iter][0]

# Load config using Hydra
@hydra.main(config_path="config", config_name="eval_config.yaml", version_base="1.3")
def load_config(cfg: DictConfig):
    return cfg

# Function to convert image to 8-bit format
def convert_to_8bit(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
    return image

# Main UI application
def main():
    st.title("Interactive Medical Image Segmentation")
    
    # Initialize the click processor
    if 'click_processor' not in st.session_state:
        st.session_state.click_processor = ClickProcessor()
    
    # Sidebar for uploading and configuration
    st.sidebar.title("Configuration")
    
    # Model configuration
    config_path = st.sidebar.text_input("Config Path", "kprism/config/eval_config.yaml")
    model_checkpoint = st.sidebar.text_input("Model Checkpoint", "/path/to/your/model/checkpoint.pth")
    
    # File upload
    file_type = st.sidebar.selectbox("Select File Type", ("NIfTI (.nii.gz)", "2D Image (.png, .jpg)"))
    uploaded_file = st.sidebar.file_uploader(f"Upload {file_type}", type=["nii.gz"] if file_type == "NIfTI (.nii.gz)" else ["png", "jpg"])
    
    # Initialize model when config is set
    try:
        if os.path.exists(config_path):
            # Load config using Hydra
            sys.argv = [sys.argv[0]]  # Reset sys.argv to avoid Hydra conflicts
            cfg = OmegaConf.load(config_path)
            
            # Override the resume path to use the provided checkpoint
            if os.path.exists(model_checkpoint):
                cfg.testing.resume = model_checkpoint
                
            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = InferenceCore(cfg).to(device)
            model.eval()
            
            if os.path.exists(cfg.testing.resume):
                checkpoint = torch.load(cfg.testing.resume, map_location=device)
                # Remove 'module.' prefix if the model was saved with DistributedDataParallel
                if isinstance(checkpoint['model'], dict):
                    checkpoint_model = {}
                    for k, v in checkpoint['model'].items():
                        if k.startswith('module.'):
                            k = k[7:]  # Remove 'module.' prefix
                        checkpoint_model[k] = v
                    model.load_state_dict(checkpoint_model, strict=False)
                else:
                    model.load_state_dict(checkpoint['model'], strict=False)
                st.sidebar.success(f"Loaded checkpoint from {cfg.testing.resume}")
            else:
                st.sidebar.error(f"Checkpoint not found at {cfg.testing.resume}")
    except Exception as e:
        st.sidebar.error(f"Error initializing model: {e}")
        model = None
    
    # Use a more unique key for st.radio widgets
    session_id = st.session_state.get('session_id', str(time.time()))
    st.session_state.session_id = session_id

    # Pre-select point type before clicking
    point_type_str = st.sidebar.radio("Select Point Type for Next Click:", ("Positive", "Negative"), key="pre_select_point_type")
    is_positive_click = point_type_str == "Positive"

    # --- User Interaction for Adding Points with Mouse ---
    if file_type == "NIfTI (.nii.gz)" and uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.nii.gz", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the NIfTI file
        nii_data = load_and_visualize_nii("temp.nii.gz")
        if nii_data is not None:
            num_slices = nii_data.shape[0]
            slice_idx = st.sidebar.slider("Slice", 0, num_slices-1, num_slices//2)
            current_slice = nii_data[slice_idx]
            
            # Convert current_slice to 8-bit format
            current_slice_8bit = convert_to_8bit(current_slice)
            
            # Display the slice using streamlit-drawable-canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=5,
                stroke_color="#FF0000",
                background_color="#eee",
                background_image=Image.fromarray(current_slice_8bit),
                update_streamlit=True,
                height=current_slice_8bit.shape[0],
                width=current_slice_8bit.shape[1],
                drawing_mode="point",
                key="canvas"
            )
            
            # Process the points drawn on the canvas
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                for obj in objects:
                    x, y = obj["left"], obj["top"]
                    # Use the pre-selected point type
                    st.session_state.click_processor.add_point(x, y, is_positive_click)
                    # Allow user to select point type for next click
                    point_type_str = st.sidebar.radio("Select Point Type for Next Click:", ("Positive", "Negative"), key=f"pre_select_point_type_{x}_{y}_{time.time()}")
                    is_positive_click = point_type_str == "Positive"

            # Display points information below the image
            st.subheader("Points Information:")
            for i, (x, y, is_positive) in enumerate(st.session_state.click_processor.get_points()):
                point_type = "Positive" if is_positive else "Negative"
                st.write(f"Point {i+1}: Location ({x:.2f}, {y:.2f}), Type: {point_type}")

            # Display points
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(current_slice_8bit, cmap='gray')
            for point in st.session_state.click_processor.points:
                x, y, is_positive = point
                color = 'green' if is_positive else 'red'
                ax.add_patch(Circle((x, y), radius=5, color=color))
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            st.image(buf, caption=f"Slice {slice_idx}", use_column_width=True)

            if st.button("Clear Points"):
                st.session_state.click_processor.clear_points()
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

    # For 2D images
    elif file_type == "2D Image (.png, .jpg)" and uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_image", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the image
        image = cv2.imread("temp_image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert image to 8-bit format
        image_8bit = convert_to_8bit(image)
        
        # Display the image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_8bit)
        
        # Display points
        for point in st.session_state.click_processor.points:
            x, y, is_positive = point
            color = 'green' if is_positive else 'red'
            ax.add_patch(Circle((x, y), radius=5, color=color))
        
        # Convert to image and display
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        st.image(buf, caption="Image", use_column_width=True)
        
        # Similar click handling and segmentation as for NIfTI files
        # ...

if __name__ == "__main__":
    main()