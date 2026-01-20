# Exploring Asteroid Orbits: Neural Network Classification

A machine learning research project that classifies asteroid orbits using neural networks and other ML models, achieving 99.18% validation accuracy through custom Keras architecture and SMOTE data augmentation.

## Overview

This project tackles the challenge of classifying asteroids into 11 different orbital categories using NASA JPL orbital data. By employing advanced machine learning techniques including neural networks, SMOTE for handling imbalanced data, and hyperparameter optimization with Weights & Biases, the models successfully distinguish between complex orbital patterns that are crucial for understanding solar system evolution and planetary defense.

## Features

This project includes a full multi-model pipeline that compares five different classification approaches for predicting asteroid orbital classes: Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), an MLP neural network, and a custom Keras neural network enhanced with automated hyperparameter tuning. The workflow begins with a comprehensive preprocessing stage that reduces the original 45-column dataset by removing 38 unnecessary features, imputes missing values using median statistics, and performs an 80–20 train-test split to standardize evaluation. 

To address significant class imbalance across orbital types, the project applies SMOTE data augmentation to generate synthetic samples for underrepresented classes, helping models learn robust decision boundaries rather than overfitting to the majority class. Finally, the best-performing neural network is optimized through Weights & Biases sweeps, exploring thousands of hyperparameter combinations and pairing the results with thorough evaluation tooling, including accuracy scoring, confusion matrices, and overfitting checks.

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

Across all tested models, performance clearly improved as the classifiers became better at capturing non-linear boundaries and higher-dimensional interactions between orbital elements. Simpler linear decision-making struggled with the complexity of the class structure, while neighborhood-based and neural approaches were able to separate orbital regimes far more effectively.

The strongest performer was the custom Keras neural network, which achieved 99.18% validation accuracy. Its confusion matrix was nearly perfectly diagonal, indicating that the model generalized cleanly across the orbital categories rather than succeeding only on a few dominant classes. This result reflects both the network’s representational capacity and the project’s focus on feature selection, preprocessing, and handling imbalance.

The best model used a consistent, compact architecture with five hidden layers and 16 nodes per layer, trained with ReLU activations for 149 epochs and a batch size of 1000. In practice, this setup was strong enough to learn the subtle transitions between orbit families (for example, near-Earth vs. main-belt boundaries) without requiring an excessively large network.

Other models performed well but trailed the tuned Keras approach. The MLP neural network reached 96.93% accuracy, followed closely by KNN at 96.07%, showing that non-linear decision surfaces (either learned or instance-based) are well-suited to the orbital feature space. The Random Forest achieved 94.87%, which is still strong but suggests that the problem benefits from smoother, higher-capacity representations than tree ensembles alone provided in this setup.

In contrast, Logistic Regression reached only 56.23% accuracy, reinforcing that orbital classes are not cleanly separable using a linear model in the chosen feature space. Overall, the results show that combining a carefully cleaned feature set with imbalance correction and neural network optimization can yield extremely high classification performance on large-scale orbital data.

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

## Dataset

The dataset used in this project comes from NASA’s Jet Propulsion Laboratory (JPL) and contains orbital and physical parameters for a large population of asteroids, enabling supervised classification into distinct orbital families relevant to solar system dynamics and planetary defense. The original dataset contains 958,524 entries with 45 total features, combining both orbital elements and observational/physical properties. 

To reduce computational load and improve learning efficiency, the dataset was streamlined to a smaller set of high-signal features, and the dominant Main-belt asteroid (MBA) class was removed, since it made up roughly 89% of the total samples and would otherwise overwhelm training. The final training data uses a focused feature set designed to preserve the orbital geometry and motion information most directly tied to class definitions, while keeping the target label as the asteroid’s orbital class.

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

## References

For detailed methodology, results, and references, please see the accompanying research paper `Research_Paper.pdf`.

## Contact

**Author**: Aarav Sonthalia  
**Location**: Short Hills, NJ, USA

---

*This project demonstrates the application of machine learning to astronomical classification problems and contributes to our understanding of orbital dynamics in the solar system.*
