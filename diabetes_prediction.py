import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
df = pd.read_csv('health_activity_data.csv')
df.head()

# ── 2. Dataset Info ───────────────────────────────────────────────────────────
print(df.info())

# ── 3. Check Missing Values ───────────────────────────────────────────────────
print("Null values per column:\n", df.isnull().sum())

# ── 4. Remove Duplicates ──────────────────────────────────────────────────────
df = df.drop_duplicates()

# ── 5. Handle Missing Values ──────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include='number').columns
df = df.fillna(df[numeric_cols].median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

# ── 6. Box Plot of Numeric Features ──────────────────────────────────────────
plt.figure(figsize=(15, 8))
df[numeric_cols].plot(kind='box')
plt.title("Box Plot of Numeric Features")
plt.xticks(rotation=45)
plt.show()

# ── 7. Histograms ─────────────────────────────────────────────────────────────
df[numeric_cols].hist(figsize=(15, 10), bins=15)
plt.show()

# ── 8. Correlation Heatmap ────────────────────────────────────────────────────
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numeric Features")
plt.show()

# ── 9. Encode Target Variable ─────────────────────────────────────────────────
df['Diabetic'] = df['Diabetic'].map({'Yes': 1, 'No': 0})

# ── 10. Select Features ───────────────────────────────────────────────────────
features = ['Age', 'BMI', 'Daily_Steps', 'Exercise_Hours_per_Week', 'Calories_Intake']
X = df[features]
y = df['Diabetic']

# ── 11. Train-Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 12. Train Decision Tree ───────────────────────────────────────────────────
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# ── 13. Predict & Evaluate ────────────────────────────────────────────────────
y_pred = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ── 14. Visualize Decision Tree ───────────────────────────────────────────────
plt.figure(figsize=(15, 8))
plot_tree(dt, feature_names=features, class_names=['No', 'Yes'], filled=True)
plt.show()

# ── 15. Feature Importance ────────────────────────────────────────────────────
pd.Series(dt.feature_importances_, index=features).sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()
