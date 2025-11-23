# Exploring Asteroid Orbits: Neural Network Classification

A machine learning research project that classifies asteroid orbits using neural networks and other ML models, achieving 99.18% validation accuracy through custom Keras architecture and SMOTE data augmentation.

## Overview

This project tackles the challenge of classifying asteroids into 11 different orbital categories using NASA JPL orbital data. By employing advanced machine learning techniques including neural networks, SMOTE for handling imbalanced data, and hyperparameter optimization with Weights & Biases, the models successfully distinguish between complex orbital patterns that are crucial for understanding solar system evolution and planetary defense.

## Features

- **Multi-Model Comparison**: Implements and evaluates 5 different classification approaches
  - Logistic Regression
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP) Neural Networks
  - Custom Keras Neural Network with hyperparameter tuning

- **Data Preprocessing Pipeline**: Comprehensive data cleaning and preparation
  - Removal of 38 unnecessary features from original 45-column dataset
  - Null value imputation using median values
  - Train-test split (80-20)

- **SMOTE Data Augmentation**: Addresses class imbalance by generating synthetic samples for minority classes

- **Hyperparameter Optimization**: Automated tuning using Weights & Biases sweeps across 6,750 parameter combinations

- **Comprehensive Evaluation**: Detailed model assessment using accuracy scores, confusion matrices, and overfitting analysis

## Orbital Classes

The project classifies asteroids into the following categories:

- **Apollo (APO)** - Earth-orbit intersecting
- **Amor (AMO)** - Earth-approaching 
- **Atira (IEO)** - Interior to Earth's orbit
- **Mars-Crossing (MCA)** - Mars orbit intersecting
- **Inner Main-belt (IMB)** - Inner asteroid belt
- **Outer Main-belt (OMB)** - Outer asteroid belt
- **Jupiter Trojan (TJN)** - Jupiter's Lagrange points
- **Centaur (CEN)** - Between Jupiter and Neptune
- **Trans-Neptunian Objects (TNO)** - Beyond Neptune
- **Hyperbolic (HYA)** - Hyperbolic trajectories

*Note: Main-belt asteroids (MBA) were removed to reduce computational load, as they comprised 89% of the 950,000+ samples.*

## Results

### Best Performing Model: Custom Keras Neural Network
- **Validation Accuracy**: 99.18%
- **Architecture**: 5 hidden layers with 16 nodes each
- **Activation Function**: ReLU
- **Epochs**: 149
- **Batch Size**: 1000
- **Performance**: Nearly diagonal confusion matrix indicating excellent classification across all orbital types

### Other Model Performance
- **MLP Neural Network**: 96.93% accuracy
- **K-Nearest Neighbors**: 96.07% accuracy
- **Random Forest**: 94.87% accuracy
- **Logistic Regression**: 56.23% accuracy

## Installation

### Prerequisites
```bash
Python 3.7+
Google Colab (recommended) or local environment with GPU support
```

### Required Libraries
```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn keras tensorflow wandb
```

## Usage

### 1. Data Preprocessing

Run `data_preprocessing.py` to clean and prepare the dataset:

```python
# Update the file path to your dataset location
dataSetPath = "/path/to/your/dataset.csv"

# The script will:
# - Remove unnecessary features
# - Handle null values
# - Split into X (features) and y (labels)
# - Export processed data to CSV files
```

### 2. Model Training and Classification

Run `class_prediction.py` to train and evaluate models:

```python
# Update paths to your processed data
xPath = "/path/to/dataSetX.csv"
yPath = "/path/to/dataSetY.csv"

# The script will:
# - Apply SMOTE for data augmentation
# - Train multiple classification models
# - Generate confusion matrices
# - Display accuracy scores
# - Perform hyperparameter tuning with W&B
```

### 3. Hyperparameter Tuning

For hyperparameter optimization with Weights & Biases:

```python
# Login to W&B (required on first run)
wandb.login()

# Configure sweep parameters in the sweep_config dictionary
# Run the sweep to find optimal hyperparameters
wandb.agent(sweep_id, train, count=100)
```

## Project Structure

```
asteroid-orbit-classification/
│
├── data_preprocessing.py      # Data cleaning and preparation
├── class_prediction.py         # Model training and evaluation
├── Research_Paper.pdf          # Detailed research findings
├── dataset.csv                 # Raw NASA JPL data (not included)
├── dataSetX.csv               # Processed features (generated)
├── dataSetY.csv               # Processed labels (generated)
└── model.keras                # Saved trained model (generated)
```

## Dataset

**Source**: NASA Jet Propulsion Laboratory (JPL)

**Original Size**: 958,524 entries × 45 features

**Features Used** (12 total):
- Absolute magnitude (H)
- Diameter
- Geometric albedo
- Median anomaly (ma)
- Orbital period (per)
- Eccentricity (e)
- Semi-major axis length (a)
- Perihelion distance (q)
- Inclination (i)
- Mean motion (n)
- Argument of perihelion (w)
- Orbital class (target variable)

## Methodology

1. **Data Collection**: Sourced from NASA JPL Small-Body Database
2. **Feature Selection**: Reduced from 45 to 12 most relevant orbital parameters
3. **Data Cleaning**: Median imputation for null values
4. **Class Balancing**: SMOTE with k-neighbors=3 (after MBA removal)
5. **Model Training**: Tested 5 different classification approaches
6. **Optimization**: Hyperparameter tuning using W&B sweeps
7. **Evaluation**: Accuracy metrics and confusion matrix analysis

## Key Findings

- **Number of nodes** showed the highest correlation with validation accuracy
- Neural networks significantly outperformed traditional ML models
- SMOTE effectively addressed class imbalance issues
- The custom Keras model successfully distinguished between complex orbital patterns
- MLP and KNN models also demonstrated strong performance (96%+)

## Limitations

- Main-belt asteroids (MBAs) were excluded to reduce computational load
- Dataset may not fully represent all orbital dynamics in the solar system
- Model performance on unseen data including MBAs requires further validation
- Computational resources limited the scope of hyperparameter exploration

## Future Work

- Reintroduce main-belt asteroids with optimized computational resources
- Incorporate additional orbital features for enhanced classification
- Extend model to classify comets, satellites, and free-floating objects
- Explore ensemble methods combining multiple high-performing models
- Deploy model as a web application for real-time orbit classification

## References

For detailed methodology, results, and references, please see the accompanying research paper `Research_Paper.pdf`.

## Contact

**Author**: Aarav Sonthalia  
**Location**: Short Hills, NJ, USA

---

*This project demonstrates the application of machine learning to astronomical classification problems and contributes to our understanding of orbital dynamics in the solar system.*
