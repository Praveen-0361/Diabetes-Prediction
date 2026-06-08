# Diabetes Prediction using Support Vector Machine (SVM)

## Project Overview

This project focuses on predicting whether a person has diabetes using the **PIMA Indians Diabetes Dataset** and a **Support Vector Machine (SVM)** machine learning algorithm. The model analyzes various health-related parameters and predicts the likelihood of diabetes.

The objective of this project is to build a reliable classification model that can assist in the early detection of diabetes based on medical diagnostic measurements.

---

## Dataset Information

The project uses the **PIMA Indians Diabetes Dataset**, a widely used dataset for machine learning classification problems.

### Dataset Features

| Feature                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| Pregnancies              | Number of times pregnant                         |
| Glucose                  | Plasma glucose concentration                     |
| BloodPressure            | Diastolic blood pressure (mm Hg)                 |
| SkinThickness            | Triceps skin fold thickness (mm)                 |
| Insulin                  | 2-Hour serum insulin (mu U/ml)                   |
| BMI                      | Body Mass Index                                  |
| DiabetesPedigreeFunction | Diabetes pedigree function                       |
| Age                      | Age of the patient                               |
| Outcome                  | Target Variable (0 = Non-Diabetic, 1 = Diabetic) |

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* Support Vector Machine (SVM)
* Jupyter Notebook

---

## Project Workflow

### 1. Importing Dependencies

The required Python libraries are imported for data processing, visualization, model building, and evaluation.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
```

---

### 2. Data Collection and Analysis

* Load the PIMA Diabetes dataset.
* Explore dataset dimensions.
* Check for missing values.
* Analyze statistical information.
* Understand feature distributions.

```python
diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.head()
```

---

### 3. Data Standardization

Since the dataset contains features with different scales, standardization is applied to improve model performance.

```python
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
```

Benefits:

* Improves SVM performance.
* Ensures all features contribute equally.
* Speeds up model convergence.

---

### 4. Train-Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
```

Typical Split:

* Training Data: 80%
* Testing Data: 20%

---

### 5. Training the Model

A Support Vector Machine classifier with a linear kernel is used.

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

Why SVM?

* Effective for binary classification.
* Works well with high-dimensional data.
* Provides robust decision boundaries.

---

### 6. Model Evaluation

The model performance is evaluated using accuracy score.

```python
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```

Evaluation Metrics:

* Training Accuracy
* Testing Accuracy
* Classification Performance

---

### 7. Making a Predictive System

The trained model can predict diabetes status for new patient data.

```python
input_data = (5,166,72,19,175,25.8,0.587,51)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
```

---

## Project Structure

```text
DiabetesPrediction/
│
├── diabetes.csv
├── Diabetes_Prediction.ipynb
├── diabetes_prediction.py
├── README.md
└── requirements.txt
```

---

## Applications

* Healthcare Decision Support Systems
* Early Diabetes Detection
* Clinical Research
* Medical Data Analysis
* Predictive Healthcare Solutions

---

## Future Enhancements

* Hyperparameter tuning using GridSearchCV
* Comparison with other algorithms (Random Forest, XGBoost, Logistic Regression)
* Deployment using Flask or Streamlit
* Real-time prediction web application
* Integration with healthcare management systems

---

## Results

The SVM-based diabetes prediction model successfully classifies patients as diabetic or non-diabetic based on medical parameters. With proper data preprocessing and standardization, the model achieves reliable prediction accuracy and demonstrates the practical application of machine learning in healthcare.

---

## Author

**Anduri Praveen Kumar Reddy**                                                               
Bachelor of Technology in Computer Science & Engineering (Information Security)      
Machine Learning & Artificial Intelligence Enthusiast                
Python | Data Science | Machine Learning | Healthcare Analytics
