# Student Academic Performance Analysis

Machine Learning project for predicting student academic performance using demographic, social, and academic factors. The project applies Exploratory Data Analysis (EDA), feature engineering, classification models, and feature importance analysis to identify key factors influencing student success.

## Author
**Mutyala Padmaja**

---

## Project Overview

This project analyzes two student datasets:

- Mathematics Student Dataset (`student-mat.csv`)
- Portuguese Student Dataset (`student-por.csv`)

The objective is to classify students into performance categories based on their final grades:

| Grade Range | Category |
|------------|----------|
| < 10 | Fail |
| 10 - 13 | Average |
| ≥ 14 | Excellent |

The project compares the performance of:
- Logistic Regression
- Random Forest Classifier

---

## Dataset Information

The datasets contain student-related information such as:

- Age
- Gender
- Family Size
- Parent Education
- Parent Occupation
- Study Time
- School Support
- Internet Access
- Previous Academic Performance
- Number of Failures
- Absences

Total Features: **33**

---

## Exploratory Data Analysis (EDA)

The following analyses were performed:

### Grade Distribution
Visualized the distribution of final grades (G3).

### Study Time vs Performance
Analyzed the impact of study time on academic performance.

### Absences vs Performance
Studied the relationship between attendance and final grades.

### Failures vs Performance
Examined how previous failures affect student outcomes.

### Parent Education Analysis
Investigated the influence of parental education on grades.

### Gender Analysis
Compared academic performance across genders.

### Correlation Heatmap
Identified relationships among numerical features.

---

## Machine Learning Workflow

### 1. Data Preprocessing
- Missing value analysis
- Data inspection
- Target variable creation

### 2. Feature Engineering
Created a new target variable:

```python
Fail
Average
Excellent
```

### 3. Encoding
Applied Label Encoding to categorical features.

### 4. Feature Selection

Selected important attributes including:

- Age
- Gender
- Study Time
- Failures
- Parent Education
- School Support
- Internet Access
- Absences
- G1
- G2

### 5. Model Training

Implemented:

- Logistic Regression
- Random Forest Classifier

### 6. Model Evaluation

Used:

- Accuracy Score
- Classification Report
- Confusion Matrix

---

## Results

### Mathematics Dataset (student-mat.csv)

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 84.81% |
| Random Forest | 84.81% |

### Portuguese Dataset (student-por.csv)

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 83.08% |
| Random Forest | 85.38% |

### Best Performing Model

**Random Forest Classifier**
achieved the highest accuracy of **85.38%** on the Portuguese Student Dataset.

---

## Feature Importance Analysis

The most influential features identified by Random Forest were:

1. G2 (Second Period Grade)
2. G1 (First Period Grade)
3. Absences
4. Failures
5. Study Time
6. Parent Education

These factors showed the strongest impact on final student performance.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Project Structure

```text
student-academic-performance-analysis/
│
├── student-mat.csv
├── student-por.csv
├── student_performance.py
├── README.md
├── requirements.txt
│
├── reports/
│   ├── ML_hackathon_endsem.pdf
│   └── Project_Presentation.pdf
│
└── screenshots/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/student-academic-performance-analysis.git
```

Move into the project folder:

```bash
cd student-academic-performance-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python student_performance.py
```

---

## Key Insights

- Previous academic performance (G1 and G2) is the strongest predictor of final grades.
- Students with fewer absences tend to achieve higher scores.
- Higher study time generally correlates with better academic performance.
- Previous failures negatively impact future performance.
- Parent education levels show a positive relationship with student achievement.

---

## Future Enhancements

- Hyperparameter Tuning
- Cross Validation
- XGBoost Implementation
- Streamlit Web Application Deployment
- Advanced Feature Engineering
- Real-Time Student Performance Prediction

---

## Repository Topics

```text
machine-learning
data-science
python
classification
random-forest
logistic-regression
student-performance
education-analytics
eda
```

---

## License

This project is developed for academic and learning purposes.