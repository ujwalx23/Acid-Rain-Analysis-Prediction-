# 🌧️ Acid Rain Analysis & Prediction Across India

This project analyzes and predicts acid rain behavior (specifically **pH levels**) across different regions of India using air quality and chemical ion concentration data. The solution includes data cleaning, feature engineering, exploratory analysis, machine learning modeling, and clustering to identify pollution hotspots and their correlation with acidic precipitation.

---

## 📂 Dataset Overview

- **Name:** `acid_rain_pan_india_cleaned.csv`
- **Size:** ~1000+ records
- **Features Include:**
  - SO2, NOx, PM2.5, RSPM, SPM
  - pH levels (target variable)
  - Sulfate, Nitrate, Ammonium, Calcium, Chloride
  - TDS, Conductivity
  - Location, State, Season

---

## 🎯 Objectives

- Predict the **pH level of rainfall** using pollutant concentrations.
- Identify which cities are **most affected** by acid rain precursors (SO2, NOx).
- Use **clustering** to detect geographical zones at high risk.
- Build **ML models** to quantify environmental damage potential.

---

## 🔧 Technologies & Libraries

- Python 3.x
- pandas, numpy, seaborn, matplotlib
- scikit-learn (Linear Regression, Random Forest, KMeans)

---

## 🔍 Exploratory Data Analysis

- Heatmaps for feature correlation
- Seasonal boxplots of pH levels
- Top 10 cities by SO₂ pollution
- Scatter plots with clustering on SO₂ vs pH

---

## 🧠 Machine Learning Models

### 1. **Linear Regression**
- Used for baseline prediction of pH
- **R² Score:** ~0.70

### 2. **Random Forest Regressor**
- Captures nonlinear interactions between pollutants and pH
- **R² Score:** ~0.88 (Best Performing)

### 3. **KMeans Clustering**
- Clustered cities based on pollution and pH risk
- 4 risk clusters visually separated on SO₂ vs pH plots

---

## 📈 Evaluation Metrics

- **Mean Absolute Error (MAE):** ~0.18
- **Mean Squared Error (MSE):** ~0.06
- **R² Score (Accuracy):** 
  - Linear Regression: 70%
  - Random Forest: **88%**

---

## 📌 File Structure

```bash
├── Project.ipynb               
├── acid_rain_pan_india_cleaned.csv   
├── project.py               
├── README.md
