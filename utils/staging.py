import numpy as np
from typing import Dict, List, Tuple

class StagingClassifier:
    """
    Clinical staging classifier for retinoblastoma using International Classification
    """
    
    def __init__(self):
        self.staging_criteria = self._initialize_staging_criteria()
        self.risk_stratification = self._initialize_risk_stratification()
        self.treatment_guidelines = self._initialize_treatment_guidelines()
    
    def _initialize_staging_criteria(self):
        """Initialize International Classification criteria"""
        return {
            "Group A": {
                "tumor_size_max_mm": 3.0,
                "distance_from_foveola_min_mm": 3.0,
                "distance_from_optic_disc_min_mm": 1.5,
                "subretinal_seeding": False,
                "vitreous_seeding": False,
                "description": "Small intraretinal tumors away from foveola and optic disc"
            },
            "Group B": {
                "tumor_size_max_mm": float('inf'),
                "distance_from_foveola_min_mm": 0.0,  # Can be close to foveola
                "distance_from_optic_disc_min_mm": 0.0,  # Can be close to optic disc
                "subretinal_seeding": False,
                "vitreous_seeding": False,
                "subretinal_fluid": True,  # Clear subretinal fluid allowed
                "description": "All remaining intraretinal tumors not in Group A"
            },
            "Group C": {
                "subretinal_seeds_distance_max_mm": 3.0,
                "vitreous_seeds_distance_max_mm": 3.0,
                "discrete_seeding": True,
                "description": "Discrete local disease with minimal subretinal/vitreous seeding"
            },
            "Group D": {
                "subretinal_seeds_distance_max_mm": float('inf'),
                "vitreous_seeds_distance_max_mm": float('inf'),
                "diffuse_seeding": True,
                "description": "Diffuse disease with significant subretinal/vitreous seeding"
            },
            "Group E": {
                "extensive_disease": True,
                "poor_visual_potential": True,
                "description": "Extensive disease with poor visual potential"
            }
        }
    
    def _initialize_risk_stratification(self):
        """Initialize risk levels for each stage"""
        return {
            "Group A": {
                "risk_level": "Very Low",
                "eye_salvage_rate": 0.98,
                "vision_preservation_rate": 0.95,
                "survival_rate": 0.99
            },
            "Group B": {
                "risk_level": "Low", 
                "eye_salvage_rate": 0.93,
                "vision_preservation_rate": 0.85,
                "survival_rate": 0.99
            },
            "Group C": {
                "risk_level": "Moderate",
                "eye_salvage_rate": 0.85,
                "vision_preservation_rate": 0.65,
                "survival_rate": 0.97
            },
            "Group D": {
                "risk_level": "High",
                "eye_salvage_rate": 0.50,
                "vision_preservation_rate": 0.30,
                "survival_rate": 0.95
            },
            "Group E": {
                "risk_level": "Very High",
                "eye_salvage_rate": 0.05,
                "vision_preservation_rate": 0.02,
                "survival_rate": 0.93
            }
        }
    
    def _initialize_treatment_guidelines(self):
        """Initialize treatment recommendations by stage"""
        return {
            "Group A": {
                "primary_treatment": "Focal therapy",
                "options": [
                    "Laser photocoagulation",
                    "Cryotherapy", 
                    "Close observation for very small tumors"
                ],
                "chemotherapy": False,
                "monitoring_frequency": "Every 3-4 months"
            },
            "Group B": {
                "primary_treatment": "Focal therapy ± chemotherapy",
                "options": [
                    "Laser photocoagulation",
                    "Cryotherapy",
                    "Systemic chemotherapy if multiple tumors",
                    "Intra-arterial chemotherapy"
                ],
                "chemotherapy": True,
                "monitoring_frequency": "Every 3-4 months"
            },
            "Group C": {
                "primary_treatment": "Systemic chemotherapy + focal therapy",
                "options": [
                    "CEV protocol (Carboplatin, Etoposide, Vincristine)",
                    "Intra-arterial chemotherapy",
                    "Focal consolidation therapy",
                    "Intravitreal chemotherapy for vitreous seeds"
                ],
                "chemotherapy": True,
                "monitoring_frequency": "Every 3-4 weeks during treatment"
            },
            "Group D": {
                "primary_treatment": "Intensive multimodal therapy",
                "options": [
                    "High-dose systemic chemotherapy",
                    "Intra-arterial chemotherapy",
                    "Intravitreal chemotherapy",
                    "Consider enucleation if poor response"
                ],
                "chemotherapy": True,
                "enucleation_consideration": True,
                "monitoring_frequency": "Every 2-3 weeks during treatment"
            },
            "Group E": {
                "primary_treatment": "Primary enucleation (usually recommended)",
                "options": [
                    "Enucleation with orbital implant",
                    "Rarely: Intensive chemotherapy if bilateral disease",
                    "Palliative therapy in select cases"
                ],
                "chemotherapy": False,
                "enucleation_recommendation": True,
                "monitoring_frequency": "Post-surgical follow-up"
            }
        }
    
    def classify_stage(self, tumor_features: Dict) -> Tuple[str, float]:
        """
        Classify tumor stage based on features
        
        Args:
            tumor_features: Dictionary containing tumor characteristics
            
        Returns:
            Tuple of (predicted_stage, confidence_score)
        """
        # Extract key features
        tumor_size = tumor_features.get('estimated_size_mm', 0)
        spread_pattern = tumor_features.get('spread_pattern', 'None')
        location = tumor_features.get('location', 'Unknown')
        area_pixels = tumor_features.get('area_pixels', 0)
        
        # Initialize stage scores
        stage_scores = {}
        
        # Group A criteria
        if tumor_size <= 3.0 and spread_pattern in ['Contained', 'None']:
            stage_scores['Group A'] = 0.8
        else:
            stage_scores['Group A'] = 0.1
        
        # Group B criteria  
        if tumor_size > 3.0 and spread_pattern in ['Contained', 'None']:
            stage_scores['Group B'] = 0.8
        elif tumor_size <= 3.0 and location in ['Central']:
            stage_scores['Group B'] = 0.7
        else:
            stage_scores['Group B'] = 0.2
        
        # Group C criteria
        if spread_pattern == 'Irregular' and tumor_size < 10.0:
            stage_scores['Group C'] = 0.7
        elif tumor_size > 5.0 and spread_pattern == 'Contained':
            stage_scores['Group C'] = 0.6
        else:
            stage_scores['Group C'] = 0.2
        
        # Group D criteria
        if spread_pattern == 'Diffuse':
            stage_scores['Group D'] = 0.8
        elif tumor_size > 10.0:
            stage_scores['Group D'] = 0.7
        else:
            stage_scores['Group D'] = 0.2
        
        # Group E criteria
        if tumor_size > 15.0 or area_pixels > 10000:
            stage_scores['Group E'] = 0.8
        elif spread_pattern == 'Diffuse' and tumor_size > 8.0:
            stage_scores['Group E'] = 0.6
        else:
            stage_scores['Group E'] = 0.1
        
        # Apply softmax to normalize scores
        stage_scores = self._softmax_normalize(stage_scores)
        
        # Return highest scoring stage
        predicted_stage = max(stage_scores.keys(), key=lambda k: stage_scores[k])
        confidence = stage_scores[predicted_stage]
        
        return predicted_stage, confidence
    
    def _softmax_normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply softmax normalization to stage scores"""
        values = np.array(list(scores.values()))
        exp_values = np.exp(values - np.max(values))  # Subtract max for numerical stability
        softmax_values = exp_values / np.sum(exp_values)
        
        return {stage: float(score) for stage, score in zip(scores.keys(), softmax_values)}
    
    def get_risk_level(self, stage: str) -> str:
        """Get risk level for a given stage"""
        return self.risk_stratification.get(stage, {}).get('risk_level', 'Unknown')
    
    def get_prognosis_rates(self, stage: str) -> Dict[str, float]:
        """Get prognosis rates for a given stage"""
        return self.risk_stratification.get(stage, {})
    
    def get_treatment_considerations(self, stage: str) -> List[str]:
        """Get treatment considerations for a given stage"""
        guidelines = self.treatment_guidelines.get(stage, {})
        
        considerations = [
            f"Primary treatment approach: {guidelines.get('primary_treatment', 'Unknown')}",
            f"Monitoring frequency: {guidelines.get('monitoring_frequency', 'Unknown')}"
        ]
        
        if guidelines.get('chemotherapy', False):
            considerations.append("Systemic chemotherapy indicated")
        
        if guidelines.get('enucleation_consideration', False):
            considerations.append("Consider enucleation if poor treatment response")
        
        if guidelines.get('enucleation_recommendation', False):
            considerations.append("Primary enucleation typically recommended")
        
        # Add specific treatment options
        options = guidelines.get('options', [])
        if options:
            considerations.append("Treatment options include:")
            considerations.extend([f"  • {option}" for option in options])
        
        return considerations
    
    def get_staging_details(self, stage: str) -> str:
        """Get detailed staging information"""
        criteria = self.staging_criteria.get(stage, {})
        risk_info = self.risk_stratification.get(stage, {})
        
        details = f"""
        **{stage} - {risk_info.get('risk_level', 'Unknown')} Risk**
        
        **Description:** {criteria.get('description', 'No description available')}
        
        **Prognosis Rates:**
        • Eye salvage: {risk_info.get('eye_salvage_rate', 0)*100:.0f}%
        • Vision preservation: {risk_info.get('vision_preservation_rate', 0)*100:.0f}%
        • Overall survival: {risk_info.get('survival_rate', 0)*100:.0f}%
        
        **Clinical Characteristics:**
        """
        
        # Add specific criteria based on stage
        if stage == "Group A":
            details += f"""
            • Tumor size ≤ {criteria.get('tumor_size_max_mm', 3)} mm
            • Distance from foveola ≥ {criteria.get('distance_from_foveola_min_mm', 3)} mm  
            • Distance from optic disc ≥ {criteria.get('distance_from_optic_disc_min_mm', 1.5)} mm
            • No subretinal or vitreous seeding
            """
        elif stage == "Group B":
            details += """
            • Larger intraretinal tumors or closer to critical structures
            • May have clear subretinal fluid
            • No seeding present
            """
        elif stage == "Group C":
            details += """
            • Discrete subretinal seeds ≤ 3 mm from tumor
            • Discrete vitreous seeds ≤ 3 mm from tumor
            • Minimal local disease spread
            """
        elif stage == "Group D":
            details += """
            • Diffuse subretinal seeds > 3 mm from tumor
            • Diffuse vitreous seeds > 3 mm from tumor
            • Significant disease spread
            """
        elif stage == "Group E":
            details += """
            • Extensive retinal detachment
            • Tumor in anterior segment
            • Secondary glaucoma
            • Poor visual potential
            """
        
        return details
    
    def calculate_overall_risk_score(self, stage: str, tumor_features: Dict) -> float:
        """Calculate overall risk score combining staging and tumor features"""
        base_risk = {
            'Group A': 0.1,
            'Group B': 0.2, 
            'Group C': 0.4,
            'Group D': 0.7,
            'Group E': 0.9
        }.get(stage, 0.5)
        
        # Adjust based on tumor features
        size_modifier = min(tumor_features.get('estimated_size_mm', 0) / 20, 0.3)
        
        spread_modifiers = {
            'Contained': 0.0,
            'Irregular': 0.1,
            'Diffuse': 0.2,
            'None': 0.0
        }
        spread_modifier = spread_modifiers.get(tumor_features.get('spread_pattern', 'None'), 0.1)
        
        # Location risk modifier
        location_modifiers = {
            'Central': 0.15,
            'Superior': 0.05,
            'Inferior': 0.05,
            'Nasal': 0.03,
            'Temporal': 0.03,
            'None': 0.0
        }
        location_modifier = location_modifiers.get(tumor_features.get('location', 'None'), 0.0)
        
        # Calculate final risk score
        final_risk = min(base_risk + size_modifier + spread_modifier + location_modifier, 1.0)
        
        return final_risk
