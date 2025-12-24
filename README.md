# 🐾 Animal Image Classifier

[![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/Username273183/animal-classifier)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Multi-class animal image classification system achieving **98.24% test accuracy** using EfficientNet-B3 with transfer learning.

🚀 **[Try the Live Demo](https://huggingface.co/spaces/Username273183/animal-classifier)**

## 🎯 Overview

The system classifies images across 10 animal classes with state-of-the-art accuracy.

### Key Features

-  **98.24% test accuracy** on 3,928 test images
-  **Grad-CAM visualization** for model interpretability
-  **Confidence-based automation tiers** for deployment
-  **Side-by-side model comparison** (EfficientNet vs ResNet50)
-  **Interactive web demo** built with Gradio

## 📊 Results

| Model | Parameters | Test Accuracy | Key Insight |
|-------|-----------|---------------|-------------|
| **EfficientNet-B3** ✅ | **10.7M** | **98.24%** | Winner: Best accuracy with 58% fewer params |
| ResNet50 | 25.6M | 97.0% | Strong baseline, less efficient |
| ViT-Base | 85.8M | 96.51% | Underperforms on medium datasets |
| Baseline CNN | 0.4M | 50.0% | Training from scratch fails |

**Key Finding:** Transfer learning provides **48% accuracy improvement** over training from scratch.

## 🛠️ Technologies

- **Framework:** PyTorch
- **Architecture:** EfficientNet-B3 (pre-trained on ImageNet)
- **Interpretability:** Grad-CAM
- **Interface:** Gradio
- **Deployment:** Hugging Face Spaces

## 📁 Project Structure
```
animal-image-classifier/
├── app.py                          # Gradio web application
├── requirements.txt                # Python dependencies
├── examples/                       # Example images for demo
└── README.md                       # This file
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Omar-Camara/animal-image-classifier.git
cd animal-image-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Model Files

Model files are too large for GitHub. Download them from the [Hugging Face Space](https://huggingface.co/spaces/Username273183/animal-classifier/tree/main):

1. Download `efficientnet_b3_best.pth` (~43 MB)
2. Download `resnet50_finetuned_best.pth` (~103 MB)
3. Place both files in the project root directory

Or download directly:
```bash
# Download EfficientNet model
curl -L -o efficientnet_b3_best.pth "https://huggingface.co/spaces/Username273183/animal-classifier/resolve/main/efficientnet_b3_best.pth"

# Download ResNet model
curl -L -o resnet50_finetuned_best.pth "https://huggingface.co/spaces/Username273183/animal-classifier/resolve/main/resnet50_finetuned_best.pth"
```

### Run the Demo
```bash
python app.py
```

Then open http://localhost:7860 in your browser.

## 📖 How It Works

### 1. Transfer Learning
- Pre-trained EfficientNet-B3 on ImageNet (1.2M images, 1000 classes)
- Fine-tuned on animal dataset (26,179 images, 10 classes)
- Achieves 48% improvement over training from scratch

### 2. Grad-CAM Visualization
- Visualizes which parts of the image the model focuses on
- Confirms model learns anatomically relevant features
- No background bias or spurious correlations

### 3. Confidence-Based Automation
- **≥90% confidence:** Auto-accept (92% of predictions, ~100% accuracy)
- **70-90% confidence:** Flag for review (5% of predictions)
- **<70% confidence:** Require verification (3% of predictions)

## 🔬 Methodology

### Dataset
- **Total Images:** 26,179
- **Classes:** butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel
- **Split:** 70% train / 15% validation / 15% test
- **Class Imbalance:** 3.36:1 ratio (handled naturally by transfer learning)

### Training Configuration
- **Epochs:** 5 (transfer learning converges quickly)
- **Optimizer:** Adam (lr=0.0001)
- **Batch Size:** 64
- **Data Augmentation:** Horizontal flips, rotation (±15°), color jitter
- **Hardware:** Tesla T4 GPU
- **Training Time:** ~70 minutes

### Evaluation Metrics
- **Test Accuracy:** 98.24% (3,859/3,928 correct)
- **Macro F1-Score:** 98.01%
- **All classes:** >96% F1-score
- **Zero errors** above 90% confidence threshold

## 📈 Key Research Findings

1. **Transfer learning is essential**
   - 48% accuracy gain vs training from scratch (50% → 98%)
   - Converges in 2 epochs vs 20+ for baseline

2. **Architecture efficiency matters**
   - EfficientNet (10.7M params) beats ViT (85.8M params)
   - Systematic scaling > brute force parameter increase

3. **Class imbalance handling not always needed**
   - Moderate imbalance (3.36:1) handled naturally
   - Weighted/focal loss decreased performance

4. **Data quality limits performance**
   - ~25% of errors traced to dataset mislabeling
   - Model confidence serves as quality detector

5. **Uncertainty quantification enables deployment**
   - Clear confidence separation (95% vs 69%)
   - Enables 92% automation with >99% accuracy

## 📄 Citation

If you use this work, please cite:
```bibtex
@misc{camara2024animal,
  title={Multi-Class Animal Image Classification Using Transfer Learning: 
         A Comparative Study of Deep Learning Architectures},
  author={Camara, Omar},
  year={2025},
  institution={Syracuse University},
}
```

## 👨‍💻 Author

**Omar Camara**  
Graduate Student, Computer Science  
Syracuse University

- 📧 Email: omarcamara000@gmail.com
- 💼 LinkedIn: [linkedin.com/in/oc18](https://linkedin.com/in/oc18/)
- 🐙 GitHub: [github.com/Omar-Camara](https://github.com/Omar-Camara)

## 🎓 Project Context

This project was completed as part of coursework at Syracuse University (Fall 2025). The work demonstrates:

- Systematic experimental methodology
- Transfer learning effectiveness
- Production deployment considerations
- Model interpretability and uncertainty quantification

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- EfficientNet and ResNet architectures from torchvision
- Grad-CAM implementation inspired by original paper
- Gradio team for the excellent web framework

## 🔗 Links

- 🚀 [Live Demo](https://huggingface.co/spaces/Username273183/animal-classifier)
- 📊 [Hugging Face Space](https://huggingface.co/spaces/Username273183/animal-classifier)

---

⭐ **Star this repo if you find it helpful!**