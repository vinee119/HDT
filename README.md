[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20131830.svg)](https://doi.org/10.5281/zenodo.20131830)

Archived and published via Zenodo.

# Human Digital Twin for Behavior Simulation Using Artificial Intelligence

## Overview

Human behavior is influenced by emotional states, contextual conditions, and environmental factors. Traditional artificial intelligence systems often focus on isolated tasks such as emotion recognition or decision prediction, limiting their ability to simulate realistic human responses.

This project presents a Human Digital Twin (HDT) framework designed to simulate behavior by integrating:

- Emotion Detection
- Contextual Modeling
- Behavioral Prediction
- Explainable Behavioral Interpretation

The system processes textual input, identifies emotional states using Natural Language Processing (NLP) techniques, integrates contextual information, and predicts behavioral tendencies through machine learning models.

---

## Key Features

- Emotion classification using TF-IDF and Logistic Regression
- Context-aware behavioral prediction
- Synthetic behavioral dataset generation
- Real-time simulation interface using Streamlit
- Explainable behavioral interpretation layer
- Human-centric AI framework for behavior simulation

---

## Technologies Used

| Category | Technologies / Methods Used |
|---|---|
| Programming Language | Python |
| Machine Learning Algorithms | Logistic Regression, HistGradientBoostingClassifier |
| Classification Strategy | OneVsRestClassifier |
| Feature Extraction | TF-IDF Vectorization |
| Encoding Technique | OneHotEncoder |
| Natural Language Processing | Text Preprocessing, Tokenization |
| Dataset | GoEmotions Dataset, Synthetic Behavioral Dataset |
| Framework / Interface | Streamlit |
| Data Processing | Pandas, NumPy |
| Model Training | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Development Environment | Jupyter Notebook, VS Code |

---

## System Workflow

User Input → Emotion Detection → Context Integration → Behavioral Prediction → Explanation → Behavioral Interpretation

---

## Project Architecture

The HDT framework consists of the following modules:

1. Text Preprocessing
2. Feature Extraction using TF-IDF
3. Emotion Classification using Logistic Regression
4. Contextual Feature Integration
5. Behavioral Prediction
6. Explanation Module
7. Behavioral Interpretation Layer

---

## Datasets

### GoEmotions Dataset

The GoEmotions dataset is used for emotion classification and contains fine-grained emotional categories for textual analysis.

### Synthetic Behavioral Dataset

A custom synthetic dataset was developed to model relationships between:

- Emotional States
- Contextual Conditions
- Behavioral Outcomes

This dataset enables controlled behavioral prediction and simulation.

---

## Experimental Results

| Dataset | Accuracy |
|---|---|
| Synthetic Behavioral Dataset | 93.09% |
| GoEmotions Dataset | 3.31% |

The synthetic dataset achieved strong behavioral consistency in controlled conditions, while the GoEmotions dataset highlighted the complexity of real-world emotional understanding.

---

## User Interface

The system provides a real-time interactive Streamlit interface capable of:

- Detecting emotions
- Predicting behavioral tendencies
- Generating behavioral explanations
- Displaying contextual interpretations

---

## Applications

- Human-Computer Interaction
- Intelligent Simulation Systems
- Decision Support Systems
- Behavioral Analytics
- AI-based Human Modeling
- Educational and Research Applications

---

## Limitations

- Reliance on synthetic behavioral data
- Limited real-world behavioral mapping
- Static contextual categories
- Complexity in fine-grained emotion classification

---

## Future Enhancements

- Transformer-based emotion modeling
- Adaptive Human Digital Twins
- Multi-turn contextual reasoning
- Real-world behavioral dataset integration
- Advanced explainable AI techniques

---

## Repository Structure

```text
HDT/
│
├── app/
│   └── streamlit_app.py
│
├── core/
│   ├── __init__.py
│   └── simulation_engine.py
│
├── data/
│   ├── goemotions.csv
│   ├── goemotions_labels.npy
│   ├── goemotions_processed.csv
│   ├── goemotions_tfidf_features.pkl
│   ├── synthetic.csv
│   ├── synthetic_features.npy
│   ├── synthetic_features_scaled.npy
│   ├── synthetic_labels.npy
│   └── synthetic_labels_final.npy
│
├── docs/
│   ├── HDT_documentation_final.pdf
│   ├── HDT_documentation_final.zip
│   ├── HDT_final_paper.pdf
│   └── HDT_final_paper.zip
│
├── models/
│   ├── difficulty_map.pkl
│   ├── goemotions_label_names.pkl
│   ├── goemotions_model.pkl
│   ├── goemotions_vectorizer.pkl
│   ├── scenarios.pkl
│   ├── synthetic_encoder.pkl
│   ├── synthetic_model.pkl
│   └── synthetic_scaler.pkl
│
├── notebooks/
│   ├── goemotions_data_processing.ipynb
│   ├── goemotions_feature_engineering.ipynb
│   ├── goemotions_model_training.ipynb
│   ├── model_evaluation.ipynb
│   ├── preprocessing_feature_selection.ipynb
│   ├── simulation_pipeline_demo.ipynb
│   ├── synthetic_data_generation.ipynb
│   ├── synthetic_feature_engineering.ipynb
│   └── synthetic_model_training.ipynb
│
├── outputs/
│   ├── ui_input_interface.png
│   ├── behavioral_prediction_result.png
│   └── integrated_behavioral_simulation.png
│
├── requirements.txt
├── README.md
└── LICENSE

---

## Installation

```bash
git clone <repository-link>
cd HDT
pip install -r requirements.txt

## Running the Project

streamlit run app/streamlit_app.py

## Research Contribution

This project proposes an interpretable Human Digital Twin framework capable of integrating emotional intelligence and contextual reasoning for behavioral simulation using artificial intelligence.

## Author

Yenni Vineeth Kumar
Department of Computer Science and Engineering
Krishna University College of Engineering and Technology
Machilipatnam, Andhra Pradesh, India

