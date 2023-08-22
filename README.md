# Pokémon Type Classification using Logistic Regression

This project harnesses the power of logistic regression to classify Pokémon by type. Through this effort, we dive deep into the intricacies of the Pokémon dataset and leverage the scikit-learn library to build and evaluate the model.

## Overview

Pokémon, those captivating creatures with various abilities, have intrigued many. This project's core intention is to predict a Pokémon's primary type based on its attributes. While the task might sound simple, it's intriguingly challenging, primarily because Pokémon types are not strictly determined by numerical attributes.

## Dataset Description

The dataset used, `pokemon_alopez247.csv`, consists of various attributes of Pokémon, including their names, types, and several numeric features. Each row in the dataset corresponds to a unique Pokémon, and the columns provide detailed attributes about them.

Key columns:
- **Name:** The name of the Pokémon.
- **Type_1:** Primary type of the Pokémon.
- **Type_2:** Secondary type, if available.
- **Various numeric columns:** These represent different attributes, strengths, and weaknesses of the Pokémon.

## Methodology

### Data Preprocessing
1. **Removing Sparse Categories:** Due to the insufficiency of data points, Pokémon with a 'Flying' type as their primary type are excluded from the analysis.
2. **Categorical Transformation:** The function `transform_to_numeric` plays a pivotal role here. It translates categorical variables into a format suitable for machine learning. NaN values are filled, and dummy variables are created for categorical columns.

### Model Building and Evaluation
1. **Data Splitting:** The dataset is divided into training and test sets, with 80% of the data used for training and the rest for evaluation.
2. **Pipeline Creation:** A pipeline is utilized to streamline the process. First, data standardization is performed using `StandardScaler`, ensuring that all features have a mean of 0 and a standard deviation of 1. This is crucial for algorithms like logistic regression which are sensitive to feature scales. Following this, a logistic regression model, equipped to handle multiclass classification, is applied.
3. **Model Training:** Using the training data, the model is trained to understand patterns and relationships between Pokémon attributes and their primary types.
4. **Evaluation:** Post training, the model's predictive prowess is gauged on the test set. The results are articulated using a confusion matrix, which provides a visual representation of the model's performance, highlighting where it was accurate and where misclassifications occurred.

## Results

The classification report provides detailed metrics, like precision, recall, and F1-score, for each Pokémon type. Furthermore, a confusion matrix visually captures the true vs. predicted classifications, offering insights into the areas where the model shines and where it falters.

