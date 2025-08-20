import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

class ResultsVisualizer:
    """
    Comprehensive visualization utilities for retinoblastoma analysis results
    """
    
    def __init__(self):
        self.colors = {
            'tumor': (255, 0, 0),      # Red for tumor regions
            'normal': (0, 255, 0),     # Green for normal tissue
            'vessels': (0, 0, 255),    # Blue for blood vessels
            'optic_disc': (255, 255, 0) # Yellow for optic disc
        }
    
    def overlay_segmentation(self, original_image, segmentation_mask, alpha=0.4):
        """
        Overlay segmentation mask on original image
        
        Args:
            original_image: PIL Image or numpy array
            segmentation_mask: Binary mask (H, W) or (H, W, 1)
            alpha: Transparency of overlay
            
        Returns:
            PIL Image with segmentation overlay
        """
        # Convert inputs to numpy arrays
        if isinstance(original_image, Image.Image):
            image_array = np.array(original_image)
        else:
            image_array = original_image.copy()
        
        # Ensure image is RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        
        # Process segmentation mask
        if len(segmentation_mask.shape) == 3:
            mask = segmentation_mask.squeeze()
        else:
            mask = segmentation_mask
        
        # Resize mask to match image if necessary
        if mask.shape != image_array.shape[:2]:
            mask = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]))
        
        # Create colored overlay
        overlay = np.zeros_like(image_array)
        
        # Threshold mask
        binary_mask = (mask > 0.5).astype(bool)
        
        # Apply tumor color to mask regions
        overlay[binary_mask] = self.colors['tumor']
        
        # Blend with original image
        result = cv2.addWeighted(image_array, 1-alpha, overlay, alpha, 0)
        
        # Add contour outline for better visibility
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, self.colors['tumor'], 2)
        
        return Image.fromarray(result)
    
    def create_confidence_visualization(self, detection_prob, staging_probs=None):
        """
        Create confidence score visualizations
        """
        # Detection confidence gauge
        fig_detection = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=detection_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Detection Confidence (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if detection_prob > 0.5 else "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        visualizations = {'detection': fig_detection}
        
        # Staging confidence if provided
        if staging_probs is not None:
            stages = list(staging_probs.keys())
            probs = list(staging_probs.values())
            
            fig_staging = go.Figure(data=[
                go.Bar(x=stages, y=[p*100 for p in probs], 
                       marker_color=['red' if p == max(probs) else 'lightblue' for p in probs])
            ])
            fig_staging.update_layout(
                title="Staging Classification Confidence",
                xaxis_title="Stage",
                yaxis_title="Confidence (%)",
                yaxis_range=[0, 100]
            )
            visualizations['staging'] = fig_staging
        
        return visualizations
    
    def create_tumor_analysis_plot(self, tumor_features):
        """
        Create comprehensive tumor analysis visualization
        """
        # Radar chart for tumor characteristics
        categories = ['Size', 'Spread Pattern', 'Location Risk', 'Border Regularity']
        
        # Normalize features to 0-100 scale for visualization
        size_score = min(tumor_features.get('estimated_size_mm', 0) * 10, 100)
        
        spread_scores = {
            'Contained': 20,
            'Irregular': 60,
            'Diffuse': 100,
            'None': 0,
            'Unknown': 50
        }
        spread_score = spread_scores.get(tumor_features.get('spread_pattern', 'Unknown'), 50)
        
        location_scores = {
            'Central': 80,
            'Superior': 40,
            'Inferior': 40,
            'Nasal': 30,
            'Temporal': 30,
            'None': 0
        }
        location_score = location_scores.get(tumor_features.get('location', 'None'), 0)
        
        # Calculate border regularity from perimeter and area
        area = tumor_features.get('area_pixels', 0)
        perimeter = tumor_features.get('perimeter', 0)
        if area > 0 and perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            border_score = (1 - circularity) * 100  # Higher score = more irregular
        else:
            border_score = 0
        
        values = [size_score, spread_score, location_score, border_score]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Tumor Characteristics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Tumor Risk Assessment Profile"
        )
        
        return fig
    
    def create_staging_comparison_chart(self, predicted_stage, confidence_scores):
        """
        Create staging comparison visualization
        """
        stages = ['Group A', 'Group B', 'Group C', 'Group D', 'Group E']
        prognosis_rates = [95, 90, 80, 60, 40]  # Approximate survival rates
        
        fig = go.Figure()
        
        # Bar chart of confidence scores
        colors = ['red' if stage == predicted_stage else 'lightblue' for stage in stages]
        fig.add_trace(go.Bar(
            x=stages,
            y=[confidence_scores.get(stage, 0) * 100 for stage in stages],
            name='Confidence',
            marker_color=colors,
            yaxis='y'
        ))
        
        # Line chart of prognosis rates
        fig.add_trace(go.Scatter(
            x=stages,
            y=prognosis_rates,
            mode='lines+markers',
            name='Typical Prognosis (%)',
            line=dict(color='green', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Staging Analysis - Predicted: {predicted_stage}',
            xaxis_title='Staging Groups',
            yaxis=dict(title='Confidence (%)', side='left'),
            yaxis2=dict(title='Prognosis Rate (%)', side='right', overlaying='y'),
            legend=dict(x=0.7, y=1)
        )
        
        return fig
    
    def create_anatomical_overlay(self, image, tumor_location, tumor_size):
        """
        Create anatomical reference overlay on retinal image
        """
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image.copy()
        
        # Create overlay for anatomical regions
        overlay = image_array.copy()
        h, w = overlay.shape[:2]
        
        # Draw anatomical reference grid
        # Optic disc region (typical location)
        optic_center = (int(w * 0.6), int(h * 0.5))
        cv2.circle(overlay, optic_center, int(min(w, h) * 0.1), (255, 255, 0), 2)
        cv2.putText(overlay, 'Optic Disc', (optic_center[0] + 20, optic_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Macula region
        macula_center = (int(w * 0.4), int(h * 0.5))
        cv2.circle(overlay, macula_center, int(min(w, h) * 0.05), (0, 255, 255), 2)
        cv2.putText(overlay, 'Macula', (macula_center[0] - 30, macula_center[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Highlight tumor location if detected
        if tumor_location != 'None' and tumor_size > 0:
            location_coords = self._get_location_coordinates(tumor_location, w, h)
            tumor_radius = int(tumor_size * 5)  # Scale for visualization
            cv2.circle(overlay, location_coords, tumor_radius, (255, 0, 0), 3)
            cv2.putText(overlay, f'Tumor ({tumor_location})', 
                       (location_coords[0] + 20, location_coords[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return Image.fromarray(overlay)
    
    def _get_location_coordinates(self, location, width, height):
        """
        Map anatomical location names to image coordinates
        """
        location_map = {
            'Central': (width // 2, height // 2),
            'Superior': (width // 2, height // 4),
            'Inferior': (width // 2, 3 * height // 4),
            'Nasal': (width // 4, height // 2),
            'Temporal': (3 * width // 4, height // 2)
        }
        return location_map.get(location, (width // 2, height // 2))
    
    def generate_report_summary(self, results):
        """
        Generate a visual summary report of all results
        """
        detection_prob = results.get('detection_probability', 0)
        staging_info = results.get('staging', {})
        tumor_features = results.get('tumor_features', {})
        
        # Create summary figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Detection result
        ax1 = plt.subplot(2, 3, 1)
        colors = ['green' if detection_prob < 0.5 else 'red']
        plt.bar(['Detection'], [detection_prob], color=colors)
        plt.ylim(0, 1)
        plt.title('Retinoblastoma Detection')
        plt.ylabel('Probability')
        
        # Staging results
        if staging_info:
            ax2 = plt.subplot(2, 3, 2)
            stages = list(staging_info.get('all_probabilities', {}).keys())
            probs = list(staging_info.get('all_probabilities', {}).values())
            bars = plt.bar(stages, probs)
            predicted_stage = staging_info.get('stage', '')
            for i, (stage, bar) in enumerate(zip(stages, bars)):
                if stage == predicted_stage:
                    bar.set_color('red')
            plt.title('Staging Classification')
            plt.ylabel('Probability')
            plt.xticks(rotation=45)
        
        # Tumor characteristics
        ax3 = plt.subplot(2, 3, 3)
        characteristics = ['Size (mm)', 'Area (pixels)']
        values = [tumor_features.get('estimated_size_mm', 0), 
                 tumor_features.get('area_pixels', 0) / 1000]  # Scale area
        plt.bar(characteristics, values)
        plt.title('Tumor Measurements')
        
        # Risk assessment
        ax4 = plt.subplot(2, 3, 4)
        risk_factors = ['Detection', 'Stage Risk', 'Size Risk']
        stage_risk_map = {'Group A': 0.2, 'Group B': 0.4, 'Group C': 0.6, 'Group D': 0.8, 'Group E': 1.0}
        stage_risk = stage_risk_map.get(staging_info.get('stage', ''), 0)
        size_risk = min(tumor_features.get('estimated_size_mm', 0) / 20, 1.0)
        
        risk_values = [detection_prob, stage_risk, size_risk]
        colors = ['red' if r > 0.5 else 'orange' if r > 0.3 else 'green' for r in risk_values]
        plt.bar(risk_factors, risk_values, color=colors)
        plt.ylim(0, 1)
        plt.title('Risk Assessment')
        plt.ylabel('Risk Level')
        
        plt.tight_layout()
        
        # Convert to base64 for display
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        plot_url = base64.b64encode(plot_data).decode()
        
        return f"data:image/png;base64,{plot_url}"
