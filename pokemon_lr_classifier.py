# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# Function to transform categorical columns to numeric
def transform_to_numeric(df):
    """
    Converts specified categorical columns to numeric, fills NaN values, and creates dummy variables.
    
    Parameters:
    - df: DataFrame to be transformed

    Returns:
    - DataFrame with transformed columns
    """
    temp_df = df.drop(columns=['Name', 'Type_1', 'Type_2']).fillna(0)
    return pd.get_dummies(temp_df).replace({True: 1, False: 0})

# Load the dataset
data = pd.read_csv('pokemon_alopez247.csv', index_col='Number')

# Drop 'Flying' type Pokémon due to insufficiency in data
data = data.drop(index=data[data['Type_1'] == 'Flying'].index)

# Transform data and split features and target
X = transform_to_numeric(data)
y = data['Type_1']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier using a pipeline for scaling and logistic regression
clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=2500))
clf.fit(X_train, y_train)

# Predict the classes for the test set
pred = clf.predict(X_test)

# Print the classification report for performance metrics
print(classification_report(y_test, pred))

# Calculate and display the confusion matrix for a visual representation of the model's performance
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.xticks(rotation='vertical')
plt.title("Confusion Matrix for Pokémon Type Classification")
plt.show()
