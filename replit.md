# Retinoblastoma AI Diagnostic System

## Overview

This is a comprehensive AI-powered medical diagnostic system for retinoblastoma detection and staging. The application uses deep learning models built on VGG16 architecture to analyze retinal images and provide clinical insights. The system performs multiple tasks including tumor detection, staging classification according to International Classification guidelines, and image segmentation for tumor localization. Built with Streamlit for the web interface, it integrates advanced image preprocessing, medical visualization, and comprehensive clinical information to support healthcare professionals in retinoblastoma diagnosis.

## User Preferences

Preferred communication style: Simple, everyday language.
UI/UX preferences: Modern, visually appealing medical interface with enhanced styling and professional design.
Authentication: Secure sign-in system with demo credentials (doctor/medical123).
Visual enhancements: Gradient backgrounds, enhanced cards, modern color schemes, professional medical theming.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with wide layout configuration
- **Caching Strategy**: Uses `@st.cache_resource` decorators for model loading and component initialization to optimize performance
- **Page Structure**: Single-page application with sidebar navigation and responsive design
- **Icon Integration**: Medical-themed UI with eye icon (👁️) for branding

### Machine Learning Architecture
- **Base Model**: VGG16 pre-trained on ImageNet as feature extractor
- **Multi-task Learning**: Custom architecture with multiple output heads for different clinical tasks
- **Model Components**:
  - Detection branch for binary tumor classification
  - Staging branch for 5-class staging (Groups A-E)
  - Segmentation capabilities for tumor localization
- **Training Strategy**: Transfer learning with frozen early layers and fine-tuned later layers
- **Input Specifications**: 224x224x3 RGB images with ImageNet normalization

### Image Processing Pipeline
- **Preprocessing Chain**: Multi-stage pipeline optimized for medical retinal imaging
- **Enhancement Techniques**: Medical image enhancement including CLAHE, noise reduction, and contrast adjustment
- **Normalization**: ImageNet statistics for transfer learning compatibility
- **Format Handling**: Supports multiple image formats with automatic RGB conversion
- **Region Extraction**: Automated cropping to focus on retinal regions

### Clinical Decision Support
- **Staging System**: International Classification for Retinoblastoma implementation
- **Risk Stratification**: Automated risk assessment based on tumor characteristics
- **Treatment Guidelines**: Evidence-based treatment recommendations per staging group
- **Medical Knowledge Base**: Comprehensive disease information and clinical criteria

### Visualization System
- **Libraries Used**: Matplotlib and Plotly for different visualization needs
- **Segmentation Overlay**: Color-coded tumor region highlighting on original images
- **Interactive Charts**: Plotly-based interactive visualizations for results presentation
- **Medical Annotations**: Anatomical structure identification and labeling

### Data Flow Architecture
- **Input Processing**: Image upload → preprocessing → normalization → model inference
- **Multi-output Processing**: Parallel processing for detection, staging, and segmentation tasks
- **Results Integration**: Consolidation of multiple model outputs into unified clinical report
- **Visualization Pipeline**: Results → overlay generation → interactive chart creation

## External Dependencies

### Machine Learning Frameworks
- **TensorFlow/Keras**: Primary deep learning framework for model development and inference
- **OpenCV**: Computer vision operations and image manipulation
- **NumPy**: Numerical computing and array operations
- **scikit-image**: Advanced image processing algorithms including exposure and filtering

### Web Framework and UI
- **Streamlit**: Primary web application framework for the user interface
- **Plotly**: Interactive visualization library for charts and graphs
- **Matplotlib**: Static plotting and medical image visualization

### Image Processing
- **Pillow (PIL)**: Image loading, manipulation, and format conversion
- **SciPy**: Scientific computing including ndimage operations for image processing

### Data Handling
- **Pandas**: Data manipulation and analysis for structured clinical data
- **Base64**: Encoding utilities for image data handling in web interface

### Medical Imaging Specific
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Part of OpenCV for medical image enhancement
- **ImageNet Pre-trained Weights**: For transfer learning initialization of VGG16 base model