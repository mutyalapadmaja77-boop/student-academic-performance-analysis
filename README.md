# student-academic-performance-analysis
Machine Learning project for predicting student academic performance using Logistic Regression and Random Forest. Includes data preprocessing, exploratory data analysis (EDA), feature engineering, model evaluation, and feature importance analysis.
Author - Mutyala Padmaja
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("student-mat.csv", sep=';')

print(df.head())
print(df.columns)

# Basic Analysis
df.head()
df.info()
df.describe()
print(df.isnull().sum())

# Create Target Variable
def classify_grade(g):
    if g < 10:
        return "Fail"
    elif g < 14:
        return "Average"
    else:
        return "Excellent"

df['Performance'] = df['G3'].apply(classify_grade)

print(df['Performance'].value_counts())

# Features and Target
X = df.drop(['G3', 'Performance'], axis=1)
y = df['Performance']

print("\nFEATURE MATRIX SHAPE:", X.shape)
print("TARGET VECTOR SHAPE:", y.shape)
print(X.columns)

# Visualizations
sns.histplot(df['G3'], bins=20)
plt.title("Distribution of Final Grades")
plt.show()

sns.boxplot(x='studytime', y='G3', data=df)
plt.show()

sns.scatterplot(x='absences', y='G3', data=df)
plt.show()

sns.boxplot(x='failures', y='G3', data=df)
plt.show()

sns.boxplot(x='Medu', y='G3', data=df)
plt.title("Mother Education vs Final Grade")
plt.show()

numeric_df = df.select_dtypes(include=['int64'])

plt.figure(figsize=(14,10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.boxplot(x='sex', y='G3', data=df)
plt.title("Gender vs Final Grade")
plt.show()

# Label Encoding
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print(df.head())

# Feature Selection
selected_features = [
    'school',
    'sex',
    'age',
    'address',
    'famsize',
    'Pstatus',
    'Medu',
    'Fedu',
    'Mjob',
    'Fjob',
    'guardian',
    'studytime',
    'failures',
    'schoolsup',
    'famsup',
    'paid',
    'higher',
    'internet',
    'absences',
    'G1',
    'G2'
]

X = df[selected_features]
y = df['Performance']

print("\nFEATURES:\n")
print(X.columns)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("LOGISTIC REGRESSION RESULTS")
print("\nAccuracy :", lr_accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, lr_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, lr_pred))

# Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    max_depth=10,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("RANDOM FOREST RESULTS")
print("\nAccuracy :", rf_accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, rf_pred))

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("FEATURE IMPORTANCE")
print(importance)
df = pd.read_csv("student-por.csv", sep=';')

# Dataset Overview
print(df.head())
print(df.columns)

df.info()
print(df.describe())
print(df.isnull().sum())

# Create Performance Categories
def classify_grade(g):
    if g < 10:
        return "Fail"
    elif g < 14:
        return "Average"
    else:
        return "Excellent"

df['Performance'] = df['G3'].apply(classify_grade)

print(df['Performance'].value_counts())

# Feature Matrix and Target Vector
X = df.drop(['G3', 'Performance'], axis=1)
y = df['Performance']

print("\nFEATURE MATRIX SHAPE:", X.shape)
print("TARGET VECTOR SHAPE:", y.shape)

# Visualizations
sns.histplot(df['G3'], bins=20)
plt.title("Distribution of Final Grades")
plt.show()

sns.boxplot(x='studytime', y='G3', data=df)
plt.title("Study Time vs Final Grade")
plt.show()

sns.scatterplot(x='absences', y='G3', data=df)
plt.title("Absences vs Final Grade")
plt.show()

sns.boxplot(x='failures', y='G3', data=df)
plt.title("Failures vs Final Grade")
plt.show()

sns.boxplot(x='Medu', y='G3', data=df)
plt.title("Mother Education vs Final Grade")
plt.show()

numeric_df = df.select_dtypes(include=['int64'])

plt.figure(figsize=(14, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.boxplot(x='sex', y='G3', data=df)
plt.title("Gender vs Final Grade")
plt.show()

# Label Encoding
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print(df.head())

# Selected Features
selected_features = [
    'school',
    'sex',
    'age',
    'address',
    'famsize',
    'Pstatus',
    'Medu',
    'Fedu',
    'Mjob',
    'Fjob',
    'guardian',
    'studytime',
    'failures',
    'schoolsup',
    'famsup',
    'paid',
    'higher',
    'internet',
    'absences',
    'G1',
    'G2'
]

X = df[selected_features]
y = df['Performance']

print("\nFEATURES:")
print(X.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardization
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, lr_pred))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance)