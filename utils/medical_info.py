class MedicalInfo:
    """
    Comprehensive medical information about retinoblastoma
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_disease_overview():
        """Return comprehensive disease overview"""
        return """
        ## Retinoblastoma Overview
        
        **Retinoblastoma** is the most common primary intraocular malignancy in children, arising from the neural retina. 
        It represents approximately 3% of all childhood cancers and affects approximately 1 in 15,000-20,000 live births globally.
        
        ### Key Characteristics:
        - **Peak incidence**: 0-5 years of age (90% diagnosed before age 5)
        - **Bilateral cases**: Often diagnosed earlier (median age 12-15 months)
        - **Unilateral cases**: Typically diagnosed later (median age 24-30 months)
        - **Genetic basis**: RB1 gene mutations (chromosome 13q14)
        
        ### Clinical Presentation:
        The most common presenting sign is **leukocoria** (white pupillary reflex), occurring in 50-60% of cases.
        Other presentations include strabismus, decreased vision, orbital inflammation, and rarely, proptosis.
        
        ### Prognosis:
        With early detection and appropriate treatment, the overall survival rate exceeds 95% in developed countries.
        However, prognosis varies significantly with staging and extent of disease.
        """
    
    @staticmethod
    def get_staging_guidelines():
        """Return International Classification staging information"""
        return {
            "Group A": {
                "name": "Very Low Risk",
                "description": "Small intraretinal tumors away from foveola and optic disc",
                "characteristics": "≤3 mm in greatest dimension, ≥3 mm from foveola, ≥1.5 mm from optic disc",
                "prognosis": "Excellent (>98% eye salvage)",
                "treatment": "Focal therapy (laser photocoagulation, cryotherapy)"
            },
            "Group B": {
                "name": "Low Risk", 
                "description": "All remaining intraretinal tumors not in Group A",
                "characteristics": ">3 mm, <3 mm from foveola, or <1.5 mm from optic disc, clear subretinal fluid",
                "prognosis": "Very good (93-100% eye salvage)",
                "treatment": "Focal therapy, possible chemotherapy"
            },
            "Group C": {
                "name": "Moderate Risk",
                "description": "Discrete local disease with minimal subretinal/vitreous seeding",
                "characteristics": "Subretinal seeds ≤3 mm from tumor, vitreous seeds ≤3 mm from tumor",
                "prognosis": "Good (85-90% eye salvage)",
                "treatment": "Chemotherapy + focal therapy"
            },
            "Group D": {
                "name": "High Risk",
                "description": "Diffuse disease with significant subretinal/vitreous seeding",
                "characteristics": "Subretinal seeds >3 mm from tumor, vitreous seeds >3 mm from tumor",
                "prognosis": "Guarded (40-60% eye salvage)",
                "treatment": "Intensive chemotherapy, possible enucleation"
            },
            "Group E": {
                "name": "Very High Risk",
                "description": "Extensive disease with poor visual potential",
                "characteristics": "Extensive retinal detachment, tumor in anterior segment, secondary glaucoma",
                "prognosis": "Poor (<5% eye salvage)",
                "treatment": "Primary enucleation usually recommended"
            }
        }
    
    @staticmethod
    def get_symptoms():
        """Return list of clinical symptoms"""
        return [
            "**Leukocoria (white pupillary reflex)** - Most common presenting sign (50-60% of cases)",
            "**Strabismus (eye misalignment)** - Second most common sign (20-25% of cases)",
            "**Decreased vision or visual field defects** - May be subtle in young children",
            "**Red, painful eye** - May indicate secondary glaucoma or inflammation",
            "**Proptosis (eye protrusion)** - Rare, indicates advanced extraocular extension",
            "**Heterochromia (different colored eyes)** - Uncommon presentation",
            "**Nystagmus (involuntary eye movements)** - May occur with bilateral disease",
            "**Poor visual tracking or fixation** - Especially in bilateral cases"
        ]
    
    @staticmethod
    def get_risk_factors():
        """Return list of risk factors"""
        return [
            "**Hereditary RB1 gene mutation** - 40% of cases, all bilateral cases",
            "**Family history of retinoblastoma** - 6-15% of patients have affected relatives", 
            "**Advanced paternal age** - Increased risk of new germline mutations",
            "**Previous radiation exposure** - Rare contributing factor",
            "**Bilateral disease** - Higher risk of secondary cancers",
            "**13q deletion syndrome** - Associated with intellectual disability and dysmorphic features"
        ]
    
    @staticmethod
    def get_treatment_options():
        """Return comprehensive treatment information"""
        return [
            "**Focal Therapy** - Laser photocoagulation, cryotherapy for small tumors",
            "**Systemic Chemotherapy** - Carboplatin, etoposide, vincristine (CEV protocol)",
            "**Intra-arterial Chemotherapy** - Melphalan delivery via ophthalmic artery",
            "**Intravitreal Chemotherapy** - Direct injection for vitreous seeding",
            "**External Beam Radiation** - Limited use due to secondary cancer risk",
            "**Plaque Radiotherapy** - Episcleral radioactive plaques for focal treatment",
            "**Enucleation** - Surgical eye removal for advanced cases",
            "**Orbital Exenteration** - Rare, for extensive extraocular disease"
        ]
    
    @staticmethod
    def get_prognosis_factors():
        """Return factors affecting prognosis"""
        return {
            "Favorable": [
                "Early stage disease (Groups A-C)",
                "Unilateral presentation", 
                "No family history",
                "Age >12 months at diagnosis",
                "Good response to initial therapy"
            ],
            "Unfavorable": [
                "Advanced stage (Groups D-E)",
                "Bilateral disease",
                "Germline RB1 mutations",
                "Extraocular extension",
                "Optic nerve invasion",
                "Choroidal invasion >3mm",
                "Poor response to therapy"
            ]
        }
    
    @staticmethod
    def get_follow_up_guidelines():
        """Return follow-up recommendations"""
        return {
            "Active Treatment Phase": [
                "Examination under anesthesia every 3-4 weeks",
                "Imaging (MRI/ultrasound) as clinically indicated",
                "Monitor for treatment response and complications"
            ],
            "Surveillance Phase": [
                "Bilateral patients: Lifelong every 6 months until age 5, then annually",
                "Unilateral patients: Every 6 months for 2 years, then annually until age 5",
                "Genetic counseling for all patients and families",
                "Secondary cancer screening for germline mutation carriers"
            ],
            "Long-term Care": [
                "Annual ophthalmologic examination lifelong",
                "Genetic counseling for reproductive planning",
                "Secondary cancer surveillance",
                "Vision rehabilitation services as needed",
                "Psychosocial support for patient and family"
            ]
        }
    
    @staticmethod
    def get_differential_diagnosis():
        """Return differential diagnosis considerations"""
        return [
            "**Coat's disease** - Unilateral retinal telangiectasia with exudation",
            "**Persistent fetal vasculature** - Developmental vascular anomaly",
            "**Toxocariasis** - Parasitic infection causing retinal granulomas",
            "**Astrocytic hamartoma** - Benign retinal tumor, often in tuberous sclerosis",
            "**Combined hamartoma of retina and RPE** - Benign developmental lesion",
            "**Medulloepithelioma** - Rare intraocular tumor",
            "**Retinal detachment** - May mimic advanced retinoblastoma",
            "**Vitreous hemorrhage** - Can cause leukocoria in infants"
        ]
