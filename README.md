# Numerical Data Discretization using KBinsDiscretizer

## Project Description
This project demonstrates how to transform continuous numerical features into discrete categorical bins (Feature Quantization/Binning) to simplify complex numerical representations and handle outliers gracefully. Using the Titanic dataset, we systematically evaluate three unsupervised binning strategies to observe their impact on a Decision Tree Classifier's performance.

## Core Features
* **Equal Width (Uniform) Binning:** Divides the range of continuous features into intervals of equal size.
* **Equal Frequency (Quantile) Binning:** Divides data such that each bin contains roughly the same number of sample observations.
* **KMeans Binning:** Utilizes a 1D KMeans clustering algorithm to group values into distinct clusters.
* **Pipeline Comparison:** Side-by-side accuracy and visualization comparison before and after discretization.

## Dependencies
* Python 3.x
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib / Seaborn

## Code Preview
python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer

## Automated Feature Binarization Pipeline

### Description
This project focuses on transforming numerical variables into binary outcome based on a given threshold via Scikit-Learn's `Binarizer`. Engineered around a specific domain-knowledge scenario from the Titanic dataset, it combines `SibSp` (Siblings/Spouses) and `Parch` (Parents/Children) columns into a single `Family` metric. It then applies a structural binarization boundary threshold to generate a new definitive, binary feature: `Travelled_Alone`. 

# Automated Feature Binarization for Machine Learning

## Project Description
This repository implements automated Feature Binarization using Scikit-Learn's `Binarizer`. The project engineers a raw dataset by setting custom structural threshold boundaries to turn multi-value numerical columns into single binary flags. It uses the Titanic dataset to map raw family metrics to a boolean category representing whether a passenger traveled solo or with companions.

## Core Features
* **Feature Engineering:** Combines overlapping components (`SibSp` + `Parch`) to derive a unified `Family` size metric.
* **Threshold-Based Mapping:** Employs `Binarizer(threshold=0.0)` to distinguish between solo travelers (0) and accompanied individuals (1).
* **In-place Transformations:** Employs parameter tuning (`copy=False`) to manipulate vectors directly inside the active DataFrame.

## Dependencies
* Python 3.x
* Pandas
* Scikit-Learn

## Code Preview
python
from sklearn.preprocessing import Binarizer
from sklearn.compose import ColumnTransformer

# Values matching threshold or below become 0, above become 1
binarizer = Binarizer(threshold=0.0, copy=False)

trf = ColumnTransformer([
    ('binarize_family', binarizer, 'family')
], remainder='passthrough')

X_train_binarized = trf.fit_transform(X_train)

# Initialize the binning strategy
kbins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

# Apply transformer to specific numerical columns
trf = ColumnTransformer([
    ('bin_age', kbins, [0]),
    ('bin_fare', kbins, [1])
], remainder='passthrough')

X_train_trans = trf.fit_transform(X_train)

## Structural Outcome
Transforms continuous relationship hierarchies into highly actionable inputs, allowing structural classifiers to isolate key passenger survival behaviors based on individual status.
