import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
from PIL import Image

class RetinoblastomaModel:
    """
    Multi-task CNN model for retinoblastoma detection, staging, and segmentation
    Based on VGG16 architecture with custom heads for different tasks
    """
    
    def __init__(self):
        self.model = None
        self.input_size = (224, 224, 3)
        self.build_model()
        self.load_weights()
    
    def build_model(self):
        """Build the multi-task model architecture"""
        
        # Input layer
        input_layer = Input(shape=self.input_size)
        
        # VGG16 base model (pre-trained on ImageNet)
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=input_layer
        )
        
        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        # Shared feature extraction
        x = base_model.output
        shared_features = GlobalAveragePooling2D(name='shared_gap')(x)
        shared_features = Dropout(0.5)(shared_features)
        
        # Detection branch (binary classification)
        detection_dense1 = Dense(512, activation='relu', name='detection_dense1')(shared_features)
        detection_dropout = Dropout(0.3)(detection_dense1)
        detection_dense2 = Dense(256, activation='relu', name='detection_dense2')(detection_dropout)
        detection_output = Dense(1, activation='sigmoid', name='detection_output')(detection_dense2)
        
        # Staging branch (5-class classification: Groups A-E)
        staging_dense1 = Dense(512, activation='relu', name='staging_dense1')(shared_features)
        staging_dropout = Dropout(0.3)(staging_dense1)
        staging_dense2 = Dense(256, activation='relu', name='staging_dense2')(staging_dropout)
        staging_output = Dense(5, activation='softmax', name='staging_output')(staging_dense2)
        
        # Segmentation branch (simplified - using upsampling from features)
        # In practice, this would be more sophisticated (U-Net style)
        seg_features = base_model.get_layer('block5_conv3').output
        seg_up1 = UpSampling2D(size=(2, 2))(seg_features)
        seg_conv1 = Conv2D(256, 3, activation='relu', padding='same')(seg_up1)
        seg_up2 = UpSampling2D(size=(2, 2))(seg_conv1)
        seg_conv2 = Conv2D(128, 3, activation='relu', padding='same')(seg_up2)
        seg_up3 = UpSampling2D(size=(2, 2))(seg_conv2)
        seg_conv3 = Conv2D(64, 3, activation='relu', padding='same')(seg_up3)
        seg_up4 = UpSampling2D(size=(2, 2))(seg_conv3)
        seg_conv4 = Conv2D(32, 3, activation='relu', padding='same')(seg_up4)
        seg_up5 = UpSampling2D(size=(2, 2))(seg_conv4)
        segmentation_output = Conv2D(1, 1, activation='sigmoid', name='segmentation_output')(seg_up5)
        
        # Create the model
        self.model = Model(
            inputs=input_layer,
            outputs=[detection_output, staging_output, segmentation_output]
        )
        
        # Compile with appropriate losses for each task
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss={
                'detection_output': 'binary_crossentropy',
                'staging_output': 'categorical_crossentropy',
                'segmentation_output': 'binary_crossentropy'
            },
            loss_weights={
                'detection_output': 1.0,
                'staging_output': 1.0,
                'segmentation_output': 0.5
            },
            metrics={
                'detection_output': ['accuracy', 'precision', 'recall'],
                'staging_output': ['accuracy'],
                'segmentation_output': ['accuracy']
            }
        )
    
    def load_weights(self):
        """Load pre-trained weights (simulated - in practice, load actual trained weights)"""
        # In a real implementation, you would load actual trained weights here
        # For this demo, we'll use the initialized weights
        pass
    
    def predict(self, image):
        """
        Run inference on a preprocessed image
        
        Args:
            image: Preprocessed image array of shape (1, 224, 224, 3)
            
        Returns:
            Dictionary containing all prediction results
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        # Run inference
        detection_pred, staging_pred, segmentation_pred = self.model.predict(image, verbose=0)
        
        # Process detection results
        detection_probability = float(detection_pred[0][0])
        
        # Process staging results
        staging_classes = ['Group A', 'Group B', 'Group C', 'Group D', 'Group E']
        staging_probs = staging_pred[0]
        predicted_stage_idx = np.argmax(staging_probs)
        predicted_stage = staging_classes[predicted_stage_idx]
        stage_confidence = float(staging_probs[predicted_stage_idx])
        
        # Process segmentation results
        segmentation_mask = segmentation_pred[0]
        
        # Extract tumor features from segmentation
        tumor_features = self._extract_tumor_features(segmentation_mask, detection_probability)
        
        return {
            'detection_probability': detection_probability,
            'staging': {
                'stage': predicted_stage,
                'confidence': stage_confidence,
                'all_probabilities': {cls: float(prob) for cls, prob in zip(staging_classes, staging_probs)}
            },
            'segmentation_mask': segmentation_mask,
            'tumor_features': tumor_features
        }
    
    def _extract_tumor_features(self, segmentation_mask, detection_prob):
        """Extract tumor characteristics from segmentation mask"""
        
        # Only extract features if tumor is detected
        if detection_prob < 0.5:
            return {
                'estimated_size_mm': 0.0,
                'spread_pattern': 'None',
                'location': 'None',
                'area_pixels': 0,
                'perimeter': 0.0
            }
        
        # Threshold the segmentation mask
        binary_mask = (segmentation_mask > 0.5).astype(np.uint8)
        
        # Calculate tumor area (in pixels)
        tumor_area = np.sum(binary_mask)
        
        # Estimate size in mm (assuming standard fundus image parameters)
        # This is a simplified estimation - in practice, would need calibration
        pixels_per_mm = 50  # Approximate conversion factor
        estimated_size_mm = np.sqrt(tumor_area) / pixels_per_mm
        
        # Analyze spread pattern based on mask characteristics
        contours, _ = cv2.findContours(binary_mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Determine spread pattern based on shape characteristics
            circularity = 4 * np.pi * tumor_area / (perimeter ** 2) if perimeter > 0 else 0
            
            if circularity > 0.7:
                spread_pattern = 'Contained'
            elif circularity > 0.4:
                spread_pattern = 'Irregular'
            else:
                spread_pattern = 'Diffuse'
        else:
            spread_pattern = 'Unknown'
            perimeter = 0.0
        
        # Determine primary location based on mask position
        mask_shape = binary_mask.shape
        center_y, center_x = np.unravel_index(np.argmax(binary_mask), mask_shape[:2])
        
        if center_y < mask_shape[0] // 3:
            location = 'Superior'
        elif center_y > 2 * mask_shape[0] // 3:
            location = 'Inferior'
        else:
            if center_x < mask_shape[1] // 3:
                location = 'Nasal'
            elif center_x > 2 * mask_shape[1] // 3:
                location = 'Temporal'
            else:
                location = 'Central'
        
        return {
            'estimated_size_mm': float(estimated_size_mm),
            'spread_pattern': spread_pattern,
            'location': location,
            'area_pixels': int(tumor_area),
            'perimeter': float(perimeter)
        }
    
    def get_model_summary(self):
        """Return model architecture summary"""
        if self.model is None:
            return "Model not built"
        return self.model.summary()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
