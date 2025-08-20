import numpy as np
import cv2
from PIL import Image, ImageEnhance
import tensorflow as tf
from skimage import exposure, filters
from scipy import ndimage

class ImagePreprocessor:
    """
    Advanced image preprocessing pipeline optimized for medical retinal imaging
    """
    
    def __init__(self):
        self.target_size = (224, 224)
        self.mean_rgb = [0.485, 0.456, 0.406]  # ImageNet means
        self.std_rgb = [0.229, 0.224, 0.225]   # ImageNet stds
    
    def preprocess_for_inference(self, image):
        """
        Complete preprocessing pipeline for model inference
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image array ready for model input
        """
        # Convert to numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Ensure RGB format
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # Remove alpha channel if present
            image_array = image_array[:, :, :3]
        elif len(image_array.shape) == 2:
            # Convert grayscale to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 3 and image_array.dtype == np.uint8:
            # Ensure RGB (not BGR)
            pass
        else:
            raise ValueError(f"Unsupported image format: {image_array.shape}")
        
        # Apply medical image enhancements
        enhanced_image = self._enhance_medical_image(image_array)
        
        # Crop to retinal region (remove black borders common in fundus images)
        cropped_image = self._crop_retinal_region(enhanced_image)
        
        # Resize to target size
        resized_image = cv2.resize(cropped_image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize pixel values
        normalized_image = self._normalize_image(resized_image)
        
        # Add batch dimension
        batch_image = np.expand_dims(normalized_image, axis=0)
        
        return batch_image
    
    def _enhance_medical_image(self, image):
        """
        Apply medical-specific image enhancements
        """
        # Convert to float for processing
        image_float = image.astype(np.float32) / 255.0
        
        # Contrast Limited Adaptive Histogram Equalization (CLAHE)
        lab = cv2.cvtColor((image_float * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        # Gamma correction for better contrast
        gamma_corrected = exposure.adjust_gamma(enhanced, gamma=1.2)
        
        # Reduce noise while preserving edges
        denoised = cv2.bilateralFilter(
            (gamma_corrected * 255).astype(np.uint8), 
            d=9, sigmaColor=75, sigmaSpace=75
        ).astype(np.float32) / 255.0
        
        # Enhance blood vessel contrast (green channel typically best for vessels)
        green_channel = denoised[:, :, 1]
        enhanced_green = exposure.equalize_adapthist(green_channel, clip_limit=0.03)
        denoised[:, :, 1] = enhanced_green
        
        return (denoised * 255).astype(np.uint8)
    
    def _crop_retinal_region(self, image):
        """
        Automatically detect and crop the circular retinal region
        """
        # Convert to grayscale for region detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create mask to find non-black regions
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (should be the retinal region)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Expand slightly to ensure we don't cut off important regions
            margin = min(w, h) // 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            
            # Crop the image
            cropped = image[y:y+h, x:x+w]
            
            # Make it square by padding if necessary
            if w != h:
                size = max(w, h)
                square_image = np.zeros((size, size, 3), dtype=np.uint8)
                start_x = (size - w) // 2
                start_y = (size - h) // 2
                square_image[start_y:start_y+h, start_x:start_x+w] = cropped
                return square_image
            
            return cropped
        
        # If no contours found, return original image
        return image
    
    def _normalize_image(self, image):
        """
        Normalize image using ImageNet statistics
        """
        # Convert to float and scale to [0, 1]
        image_float = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        normalized = np.zeros_like(image_float)
        for i in range(3):
            normalized[:, :, i] = (image_float[:, :, i] - self.mean_rgb[i]) / self.std_rgb[i]
        
        return normalized
    
    def preprocess_for_training(self, image, augment=True):
        """
        Preprocessing pipeline for training with data augmentation
        
        Args:
            image: PIL Image object
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed and optionally augmented image
        """
        # Basic preprocessing
        processed = self.preprocess_for_inference(image)
        
        if augment:
            processed = self._apply_augmentations(processed[0])
            processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def _apply_augmentations(self, image):
        """
        Apply medical-appropriate data augmentations
        """
        # Random rotation (small angles to preserve anatomical structure)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, mode='nearest')
        
        # Random horizontal flip (anatomically appropriate)
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, -3, 3)  # Clamp to reasonable range
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast_factor + mean, -3, 3)
        
        # Random zoom (crop and resize)
        if np.random.random() > 0.5:
            zoom_factor = np.random.uniform(0.9, 1.1)
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            
            if zoom_factor > 1:
                # Crop center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                temp_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                image = temp_image[start_h:start_h+h, start_w:start_w+w]
            else:
                # Pad and crop
                temp_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = np.pad(temp_image, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='edge')
        
        return image
    
    def denormalize_image(self, normalized_image):
        """
        Convert normalized image back to displayable format
        """
        denormalized = np.zeros_like(normalized_image)
        for i in range(3):
            denormalized[:, :, i] = normalized_image[:, :, i] * self.std_rgb[i] + self.mean_rgb[i]
        
        # Clip and convert to uint8
        denormalized = np.clip(denormalized * 255, 0, 255).astype(np.uint8)
        return denormalized
