# Sample Medical Data

This directory contains sample medical images for testing and demonstration purposes.

## Directory Structure

```
sample_data/
├── xray/
│   ├── chest_normal_001.jpg
│   ├── chest_pneumonia_001.jpg
│   └── chest_covid_001.jpg
├── dermoscopy/
│   ├── melanoma_001.jpg
│   ├── nevus_001.jpg
│   └── basal_cell_carcinoma_001.jpg
├── ct/
│   ├── brain_normal_001.dcm
│   ├── lung_nodule_001.dcm
│   └── abdomen_001.dcm
├── mri/
│   ├── brain_t1_001.dcm
│   ├── brain_t2_001.dcm
│   └── spine_001.dcm
└── labels/
    ├── xray_labels.json
    ├── dermoscopy_labels.json
    ├── ct_labels.json
    └── mri_labels.json
```

## Data Sources

### X-Ray Images
- **Chest X-rays**: Normal, pneumonia, COVID-19 cases
- **Source**: Kaggle COVID-19 Radiography Dataset
- **Format**: JPEG/PNG
- **Resolution**: 512x512 to 1024x1024

### Dermoscopy Images
- **Skin Lesions**: Melanoma, nevus, basal cell carcinoma
- **Source**: ISIC Archive (International Skin Imaging Collaboration)
- **Format**: JPEG
- **Resolution**: 512x512 to 1024x1024

### CT Scans
- **Brain CT**: Normal brain, tumors, hemorrhage
- **Lung CT**: Normal lungs, nodules, pneumonia
- **Format**: DICOM (.dcm)
- **Slices**: Individual axial slices

### MRI Scans
- **Brain MRI**: T1, T2, FLAIR sequences
- **Spine MRI**: Sagittal and axial views
- **Format**: DICOM (.dcm)
- **Sequences**: T1-weighted, T2-weighted

## Label Format

Each modality has corresponding labels in JSON format:

```json
{
    "images": [
        {
            "filename": "chest_normal_001.jpg",
            "diagnosis": "Normal",
            "confidence": 1.0,
            "modality": "chest_xray",
            "view": "PA",
            "age": 45,
            "sex": "M"
        }
    ]
}
```

## Usage Guidelines

1. **Testing**: Use for model validation and benchmarking
2. **Demo**: Perfect for Gradio interface demonstrations
3. **Calibration**: Subset can be used for INT8 quantization calibration
4. **Evaluation**: Ground truth labels for accuracy assessment

## Ethical Considerations

- All sample data is anonymized
- Images sourced from public datasets with proper attribution
- No patient identifiable information included
- Intended for research and educational purposes only

## Data Privacy

- No real patient data stored in repository
- Synthetic or publicly available datasets only
- HIPAA compliance maintained through anonymization