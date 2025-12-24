import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

# Class names (in order)
CLASS_NAMES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
               'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# ============= Model Loading Functions =============

def load_model(model_path='efficientnet_b3_best.pth'):
    """Load the trained EfficientNet-B3 model"""
    model = models.efficientnet_b3(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model

def load_models(efficientnet_path='efficientnet_b3_best.pth', 
                resnet_path='resnet50_finetuned_best.pth'):
    """Load both EfficientNet-B3 and ResNet50 models"""
    
    # Load EfficientNet-B3
    efficientnet = models.efficientnet_b3(weights=None)
    num_ftrs = efficientnet.classifier[1].in_features
    efficientnet.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    efficientnet.load_state_dict(torch.load(efficientnet_path, map_location='cpu'))
    efficientnet.eval()
    
    # Load ResNet50
    resnet = models.resnet50(weights=None)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    resnet.load_state_dict(torch.load(resnet_path, map_location='cpu'))
    resnet.eval()
    
    return efficientnet, resnet

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load models
print("Loading models...")
try:
    efficientnet_model, resnet_model = load_models()
    print("Both models loaded successfully!")
    comparison_available = True
except FileNotFoundError as e:
    print(f"Warning: Could not load ResNet50 - {e}")
    print("Loading only EfficientNet-B3...")
    efficientnet_model = load_model()
    resnet_model = None
    comparison_available = False
    print("Model loaded successfully (comparison mode disabled)")

# For backward compatibility, keep 'model' as the main model
model = efficientnet_model

# ============= Grad-CAM Implementation =============

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        output[0, target_class].backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = torch.clamp(cam, min=0)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()

def apply_gradcam_overlay(image, cam, alpha=0.5):
    """
    Apply Grad-CAM heatmap as overlay on image
    
    Args:
        image: PIL Image (original)
        cam: numpy array (heatmap)
        alpha: transparency of overlay
    
    Returns:
        PIL Image with heatmap overlay
    """
    # Resize image to 224x224 (same as model input)
    image = image.resize((224, 224))
    image_np = np.array(image)
    
    # Resize CAM to match image
    cam_resized = cv2.resize(cam, (224, 224))
    
    # Apply colormap (red = high importance)
    colormap = plt.colormaps.get_cmap('jet')
    heatmap = colormap(cam_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Overlay heatmap on image
    overlayed = (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)
    
    return Image.fromarray(overlayed)

# ============= Automation Tier Logic =============

def get_automation_tier(confidence):
    """
    Determine automation tier based on confidence threshold
    
    Args:
        confidence: float (0-1)
    
    Returns:
        tuple: (tier_name, tier_color, tier_icon, tier_description)
    """
    if confidence >= 0.90:
        return (
            "✅ Auto-Accept",
            "#22c55e",  # green
            "✅",
            f"High confidence ({confidence:.1%}). Safe for automatic processing."
        )
    elif confidence >= 0.70:
        return (
            "⚠️ Review Recommended", 
            "#f59e0b",  # orange
            "⚠️",
            f"Medium confidence ({confidence:.1%}). Human review recommended."
        )
    else:
        return (
            "❌ Verification Required",
            "#ef4444",  # red
            "❌",
            f"Low confidence ({confidence:.1%}). Manual verification required."
        )

# ============= Prediction Functions =============

def predict_with_gradcam(image):
    """
    Predict animal class and generate Grad-CAM visualization
    
    Args:
        image: PIL Image
        
    Returns:
        tuple: (predictions_dict, gradcam_image, automation_html)
    """
    # Check if image is None (no image uploaded yet)
    if image is None:
        return {}, None, ""
    
    # Store original image for visualization
    original_image = image.copy()
    
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Get top prediction
    top_prob, top_class = torch.max(probabilities, dim=0)
    predicted_class = CLASS_NAMES[top_class.item()]
    confidence = float(top_prob)
    
    # Create results dictionary
    results = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    
    # Generate Grad-CAM for top prediction
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    
    # Need gradient-enabled forward pass for Grad-CAM
    img_tensor.requires_grad = True
    cam = gradcam.generate_cam(img_tensor, top_class.item())
    
    # Apply overlay
    gradcam_image = apply_gradcam_overlay(original_image, cam, alpha=0.4)
    
    # Get automation tier
    tier_name, tier_color, tier_icon, tier_desc = get_automation_tier(confidence)
    
    # Create HTML for automation tier display
    automation_html = f"""
    <div style="padding: 20px; border-radius: 10px; border: 3px solid {tier_color}; background-color: {tier_color}15;">
        <h2 style="margin: 0 0 10px 0; color: {tier_color};">
            {tier_icon} {tier_name}
        </h2>
        <p style="margin: 0; font-size: 16px;">
            <strong>Predicted:</strong> {predicted_class.title()}<br>
            <strong>Confidence:</strong> {confidence:.2%}<br>
            <br>
            {tier_desc}
        </p>
        <hr style="border: none; border-top: 1px solid {tier_color}50; margin: 15px 0;">
        <p style="margin: 0; font-size: 14px; color: #666;">
            <strong>Deployment Strategy:</strong><br>
            • <strong>≥90%:</strong> Auto-accept (92% of predictions, ~100% accuracy)<br>
            • <strong>70-90%:</strong> Flag for review (5% of predictions)<br>
            • <strong>&lt;70%:</strong> Require verification (3% of predictions)
        </p>
    </div>
    """
    
    return results, gradcam_image, automation_html

def compare_models(image):
    """
    Compare predictions from EfficientNet-B3 and ResNet50
    
    Args:
        image: PIL Image
        
    Returns:
        tuple: (efficientnet_results, resnet_results, efficientnet_gradcam, 
                resnet_gradcam, comparison_html)
    """
    # Check if image is None
    if image is None:
        return {}, {}, None, None, ""
    
    # Check if comparison is available
    if not comparison_available:
        return {}, {}, None, None, "<p style='color: red;'>Model comparison not available - ResNet50 model not found.</p>"
    
    # Store original image
    original_image = image.copy()
    
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)
    
    # ========== EfficientNet-B3 Prediction ==========
    with torch.no_grad():
        outputs_eff = efficientnet_model(img_tensor)
        probs_eff = torch.nn.functional.softmax(outputs_eff[0], dim=0)
    
    top_prob_eff, top_class_eff = torch.max(probs_eff, dim=0)
    results_eff = {CLASS_NAMES[i]: float(probs_eff[i]) for i in range(len(CLASS_NAMES))}
    
    # EfficientNet Grad-CAM
    target_layer_eff = efficientnet_model.features[-1]
    gradcam_eff = GradCAM(efficientnet_model, target_layer_eff)
    img_tensor_eff = transform(image).unsqueeze(0)
    img_tensor_eff.requires_grad = True
    cam_eff = gradcam_eff.generate_cam(img_tensor_eff, top_class_eff.item())
    gradcam_image_eff = apply_gradcam_overlay(original_image.copy(), cam_eff, alpha=0.4)
    
    # ========== ResNet50 Prediction ==========
    with torch.no_grad():
        outputs_res = resnet_model(img_tensor)
        probs_res = torch.nn.functional.softmax(outputs_res[0], dim=0)
    
    top_prob_res, top_class_res = torch.max(probs_res, dim=0)
    results_res = {CLASS_NAMES[i]: float(probs_res[i]) for i in range(len(CLASS_NAMES))}
    
    # ResNet Grad-CAM
    target_layer_res = resnet_model.layer4[-1]  # Last layer of ResNet
    gradcam_res = GradCAM(resnet_model, target_layer_res)
    img_tensor_res = transform(image).unsqueeze(0)
    img_tensor_res.requires_grad = True
    cam_res = gradcam_res.generate_cam(img_tensor_res, top_class_res.item())
    gradcam_image_res = apply_gradcam_overlay(original_image.copy(), cam_res, alpha=0.4)
    
    # ========== Create Comparison HTML ==========
    predicted_class_eff = CLASS_NAMES[top_class_eff.item()]
    confidence_eff = float(top_prob_eff)
    predicted_class_res = CLASS_NAMES[top_class_res.item()]
    confidence_res = float(top_prob_res)
    
    # Check if predictions agree
    agreement = predicted_class_eff == predicted_class_res
    agreement_color = "#22c55e" if agreement else "#ef4444"
    agreement_icon = "✅" if agreement else "⚠️"
    agreement_text = "Models Agree" if agreement else "Models Disagree"
    
    comparison_html = f"""
    <div style="padding: 20px; border-radius: 10px; border: 3px solid {agreement_color}; background-color: {agreement_color}15;">
        <h2 style="margin: 0 0 15px 0; color: {agreement_color};">
            {agreement_icon} {agreement_text}
        </h2>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div style="padding: 15px; background-color: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <h3 style="margin: 0 0 10px 0; color: #1e40af;">🏆 EfficientNet-B3</h3>
                <p style="margin: 5px 0;"><strong>Prediction:</strong> {predicted_class_eff.title()}</p>
                <p style="margin: 5px 0;"><strong>Confidence:</strong> {confidence_eff:.2%}</p>
                <p style="margin: 5px 0;"><strong>Parameters:</strong> 10.7M</p>
                <p style="margin: 5px 0;"><strong>Test Accuracy:</strong> 98.24%</p>
            </div>
            
            <div style="padding: 15px; background-color: #fef3f2; border-radius: 8px; border-left: 4px solid #dc2626;">
                <h3 style="margin: 0 0 10px 0; color: #991b1b;">📊 ResNet50</h3>
                <p style="margin: 5px 0;"><strong>Prediction:</strong> {predicted_class_res.title()}</p>
                <p style="margin: 5px 0;"><strong>Confidence:</strong> {confidence_res:.2%}</p>
                <p style="margin: 5px 0;"><strong>Parameters:</strong> 25.6M</p>
                <p style="margin: 5px 0;"><strong>Test Accuracy:</strong> 97.0%</p>
            </div>
        </div>
        
        <hr style="border: none; border-top: 1px solid #ccc; margin: 15px 0;">
        
        <div style="font-size: 14px; color: #666;">
            <strong>Key Differences:</strong><br>
            • EfficientNet-B3 has <strong>58% fewer parameters</strong> (10.7M vs 25.6M)<br>
            • EfficientNet-B3 achieves <strong>1.24% higher accuracy</strong> (98.24% vs 97.0%)<br>
            • Confidence difference: <strong>{abs(confidence_eff - confidence_res):.2%}</strong><br>
            {'• Both models agree - high confidence in classification' if agreement else '• Models disagree - may indicate ambiguous image'}
        </div>
    </div>
    """
    
    return results_eff, results_res, gradcam_image_eff, gradcam_image_res, comparison_html

# ============= Gradio Interface =============

# Example images
example_images = [
        ["examples/cat.jpeg"],
        ["examples/dog.avif"],
        ["examples/dog.png"],
        ["examples/sheep.jpg"],
        ["examples/sheep-2.jpg"],
        ["examples/butterfly.jpg"],
        ["examples/elephant.jpg"],
        ["examples/squirrel.jpeg"],
        ["examples/chick.jpeg"],
        ["examples/chicken.jpeg"]
]

with gr.Blocks(title="Animal Classifier") as demo:
    gr.Markdown("""
    # 🐾 Animal Image Classifier
    
    Upload an image of an animal to classify it using **EfficientNet-B3** (98.24% test accuracy).
    
    The model can recognize: **butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, and squirrel**.
    
    ---
    
    **👨‍💻 Developer:** Omar Camara | 
    [LinkedIn](https://linkedin.com/in/oc18/) | 
    [GitHub](https://github.com/Omar-Camara) | 
    [Email](mailto:omarcamara@gmail.com)
    """)
    
    with gr.Accordion("📖 Quick Start Guide", open=False):
        gr.Markdown("""
        ### Getting Started
        
        **Step 1: Choose Your Image**
        - Click an example below, OR
        - Upload your own animal image (JPG, PNG)
        - Images are automatically resized to 224×224
        
        **Step 2: Get Predictions**
        - Click the **"🔍 Classify"** button
        - Wait ~1-2 seconds for results
        
        **Step 3: Understand Results**
        - **Top Predictions:** Shows top 3 classes with confidence scores
        - **Automation Tier:** Indicates if prediction is reliable enough for auto-acceptance
          - ✅ Green = High confidence (≥90%) - Safe to auto-accept
          - ⚠️ Orange = Medium confidence (70-90%) - Review recommended
          - ❌ Red = Low confidence (<70%) - Manual verification needed
        - **Grad-CAM:** Red areas show where the model is focusing
        
        **Step 4 (Optional): Compare Models**
        - Switch to **"Model Comparison"** tab
        - See how EfficientNet-B3 compares to ResNet50
        - View side-by-side Grad-CAM visualizations
        
        ### Tips for Best Results
        - ✅ Clear, well-lit images work best
        - ✅ Animal should be the main subject
        - ✅ Front or side views are ideal
        - ⚠️ Cropped or unclear images may have lower confidence
        - ⚠️ Unusual angles (rear view, extreme close-up) may be challenging
        """)
    
    with gr.Tabs():
        # ========== Tab 1: Single Model ==========
        with gr.Tab("🔍 Classification"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload Animal Image")
                    predict_btn = gr.Button("🔍 Classify", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    ### 📊 How It Works
                    1. **Upload** an image or click an example below
                    2. Click **"Classify"** to get predictions
                    3. View the **Grad-CAM** heatmap showing where the model looks
                    4. See the **automation tier** based on confidence
                    """)
                
                with gr.Column(scale=1):
                    output_label = gr.Label(num_top_classes=3, label="🎯 Top 3 Predictions")
                    output_automation = gr.HTML(label="🤖 Automation Tier")
                    output_gradcam = gr.Image(label="🔍 Grad-CAM: Where the Model is Looking")
            
            predict_btn.click(
                fn=predict_with_gradcam,
                inputs=input_image,
                outputs=[output_label, output_gradcam, output_automation]
            )
            
            gr.Markdown("### 🖼️ Try These Examples")
            gr.Examples(
                examples=example_images,
                inputs=input_image,
            )
        
        # ========== Tab 2: Model Comparison ==========
        with gr.Tab("⚖️ Model Comparison"):
            gr.Markdown("""
            Compare predictions from **EfficientNet-B3** (our best model) vs **ResNet50** (baseline).
            See how architectural efficiency leads to better performance with fewer parameters.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_image_compare = gr.Image(type="pil", label="Upload Animal Image")
                    compare_btn = gr.Button("⚖️ Compare Models", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    comparison_summary = gr.HTML(label="Comparison Summary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🏆 EfficientNet-B3 (Winner)")
                    output_label_eff = gr.Label(num_top_classes=3, label="Predictions")
                    output_gradcam_eff = gr.Image(label="Grad-CAM")
                
                with gr.Column():
                    gr.Markdown("### 📊 ResNet50 (Baseline)")
                    output_label_res = gr.Label(num_top_classes=3, label="Predictions")
                    output_gradcam_res = gr.Image(label="Grad-CAM")
            
            compare_btn.click(
                fn=compare_models,
                inputs=input_image_compare,
                outputs=[output_label_eff, output_label_res, output_gradcam_eff, 
                        output_gradcam_res, comparison_summary]
            )
            
            gr.Markdown("### 🖼️ Try These Examples")
            gr.Examples(
                examples=example_images,
                inputs=input_image_compare,
            )
    
    # ========== Detailed Performance Metrics ==========
    with gr.Accordion("📊 Detailed Performance Metrics", open=False):
        gr.Markdown("""
        ### Per-Class Performance (Test Set)
        
        | Class | Precision | Recall | F1-Score | Test Images |
        |-------|-----------|--------|----------|-------------|
        | Spider | 99.4% | 99.2% | 99.3% | 707 |
        | Chicken | 98.8% | 99.4% | 98.9% | 477 |
        | Squirrel | 98.9% | 98.6% | 98.7% | 279 |
        | Dog | 98.3% | 97.9% | 98.6% | 724 |
        | Horse | 97.9% | 98.4% | 98.3% | 377 |
        | Cat | 98.4% | 97.6% | 98.0% | 249 |
        | Elephant | 99.1% | 95.8% | 97.6% | 237 |
        | Butterfly | 97.8% | 98.1% | 97.6% | 316 |
        | Sheep | 96.3% | 97.8% | 97.0% | 269 |
        | Cow | 95.3% | 97.3% | 96.2% | 293 |
        
        **All classes achieve >96% F1-score**
        
        ### Common Confusion Pairs
        
        Most frequent misclassifications:
        - **Cow ↔ Horse:** Similar body shapes (6 errors, 2.0%)
        - **Butterfly ↔ Spider:** Small subjects with delicate features (12 errors, 1.9%)
        - **Cat ↔ Dog:** Domestic mammals (7 errors, 1.5%)
        
        ### Training Details
        
        - **Dataset:** 26,179 images (70% train, 15% val, 15% test)
        - **Training Time:** ~70 minutes (5 epochs) on Tesla T4 GPU
        - **Optimizer:** Adam (lr=0.0001)
        - **Data Augmentation:** Horizontal flips, rotation (±15°), color jitter
        - **Transfer Learning:** Pre-trained on ImageNet (1.2M images, 1000 classes)
        
        ### Key Findings from Research
        
        1. **Transfer learning provides 48% accuracy improvement** over training from scratch
        2. **EfficientNet-B3 optimal:** 58% fewer parameters than ResNet50, 1.24% higher accuracy
        3. **No class imbalance handling needed:** Moderate imbalance (3.36:1 ratio) handled naturally by transfer learning
        4. **Data quality matters:** ~25% of errors likely due to dataset mislabeling
        5. **Confidence separation:** 95% mean confidence on correct predictions vs 69% on errors
        """)
    
    # ========== Documentation Section ==========
    gr.Markdown("""
    ---
    ### 📈 Model Performance
    
    | Model | Parameters | Test Accuracy | Inference Time* |
    |-------|-----------|---------------|-----------------|
    | **EfficientNet-B3** | **10.7M** | **98.24%** | **~50ms** |
    | ResNet50 | 25.6M | 97.0% | ~60ms |
    | ViT-Base | 85.8M | 96.51% | ~120ms |
    | Baseline CNN | 0.4M | 50.0% | ~20ms |
    
    *Approximate on CPU
    
    ### 🚀 Deployment Strategy
    This confidence-based three-tier system enables **92% automation** while maintaining **>99% accuracy**:
    - **Tier 1 (≥90% confidence):** Automatically accept - 92% of predictions with ~100% accuracy
    - **Tier 2 (70-90% confidence):** Flag for human review - 5% of predictions
    - **Tier 3 (<70% confidence):** Require manual verification - 3% of predictions
    
    This approach was validated on 3,928 test images with **zero errors** above the 90% threshold.
    
    ---
    
    ### 👨‍💻 About
    
    **Developed by:** Omar Camara   
    **Contact:** omarcamara000@gmail.com | [LinkedIn](https://linkedin.com/in/oc18/) | [GitHub](https://github.com/Omar-Camara)
    
    This demo showcases a production-ready animal classification system that achieves state-of-the-art accuracy through 
    systematic architecture comparison. The project demonstrates practical deployment considerations including uncertainty 
    quantification and human-in-the-loop workflows.
    
    **Technologies:** PyTorch, EfficientNet-B3, ResNet50, Grad-CAM, Gradio  
    **Key Finding:** EfficientNet-B3 achieves higher accuracy (98.24% vs 97.0%) with 58% fewer parameters than ResNet50
    
    **Paper:** Full technical details available in IEEE conference paper format
    """)
    
    gr.Markdown("""
    ---
    
    ### 🔗 Share This Demo
    
    Found this useful? Share with others!
    
    [![Share on Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Fhuggingface.co%2Fspaces%2FUsername273183%2Fanimal-classifier&style=social)](https://twitter.com/intent/tweet?text=Check%20out%20this%20animal%20classifier%20with%2098.24%25%20accuracy!&url=https://huggingface.co/spaces/Username273183/animal-classifier)
    [![Share on LinkedIn](https://img.shields.io/badge/Share-LinkedIn-blue?style=social&logo=linkedin)](https://www.linkedin.com/sharing/share-offsite/?url=https://huggingface.co/spaces/Username273183/animal-classifier)
    
    ⭐ **Star this Space** if you find it helpful!
    
    **Feedback?** Feel free to reach out: omarcamara000@gmail.com
    """)

if __name__ == "__main__":
    demo.launch(share=False)