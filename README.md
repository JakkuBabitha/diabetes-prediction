# Health and Lifestyle Survey: Diabetes Prediction Using Decision Tree

## Overview
A machine learning mini project submitted to **Alliance University, School of Advanced Computing**
for the course **Principles of Data Science (Z5AMC25600)**.

The project predicts the risk of **Type-2 Diabetes** based on a person's health
and lifestyle data using a **Decision Tree Classifier**.

**Submitted by:** Jakku Babitha (Roll No: 03)
**Submitted to:** Prof. Vivek Mishra
**Semester:** 1 | 2025–2027

---

##  Tools & Libraries Used
- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn (DecisionTreeClassifier)
- Google Colab

---

##  Files
- `pds_project.ipynb` — Main notebook with full code and analysis
- `health_activity_data.csv` — Kaggle dataset (1000 records, 16 features)
- `pds_report.pdf` — Full mini project report

---

##  Dataset
- **Source:** Kaggle — Health & Lifestyle Dataset by Mahdi Mashayekhi
- **Size:** 1000 records × 16 features
- **Key Features:**
  - Age, Gender, Height, Weight, BMI
  - Daily Steps, Exercise Hours per Week
  - Calories Intake, Hours of Sleep
  - Heart Rate, Blood Pressure
  - Smoker, Alcohol Consumption per Week
  - Diabetic *(Target variable)*
  - Heart Disease

---

##  Key Steps

### 1. Data Preprocessing
- Checked for missing values using `df.isnull().sum()` — no nulls found
- Removed duplicates using `df.drop_duplicates()`
- Encoded target column: `Yes → 1`, `No → 0`
- Selected key features: Age, BMI, Daily Steps, Exercise Hours, Calories Intake

### 2. Exploratory Data Analysis (EDA)
- Box plots for all numeric features to detect outliers
- Histograms for feature distributions
- Correlation heatmap to identify relationships between variables

### 3. Model Development
- Split data: 80% training, 20% testing using `train_test_split()`
- Trained a **Decision Tree Classifier** (`random_state=42`)

### 4. Model Evaluation
- **Accuracy: 76%**
- Confusion Matrix:
  ```
  [[146  25]
   [ 23   6]]
  ```
- Visualized the full decision tree
- Plotted **Feature Importance** chart

---

##  Key Findings
- **Daily Steps** was the most important feature for prediction
- **Calories Intake** and **Exercise Hours per Week** were the next most important
- Higher BMI + lower physical activity → higher diabetes risk
- The Decision Tree clearly shows how each factor influences the prediction

---

##  How to Run
1. Download `health_activity_data.csv` from [Kaggle](https://www.kaggle.com/datasets/mahdimashayekhi/health-and-lifestyle-dataset)
2. Upload to Google Colab
3. Run all cells in the notebook

---

##  References
- American Diabetes Association. Standards of Medical Care in Diabetes, 2022
- Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning, 2009
- Kaggle Dataset: https://www.kaggle.com/datasets/mahdimashayekhi/health-and-lifestyle-dataset
- https://www.scikit-learn.org
