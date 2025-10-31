# TP2-Supp-exam-2025

# Forecasting Employment Trends in South Africa (Task 2)

## Overview

This repository contains the supplementary exam submission for **Technical Programming 2** (Student: Samkelo Mbhele, Student Number: 22339019). The focus is on forecasting unemployment trends in South Africa over the next 10 years (2023–2033), with scenario analysis to consider AI adoption impacts on employment.

We use **time series forecasting** with Facebook Prophet, exploratory data analysis, and scenario modeling.

---

## Contents

| File | Description |
|------|-------------|
| `YourNotebook.ipynb` | Jupyter Notebook containing the full analysis, EDA, and forecasting steps. |
| `app.py` | Streamlit dashboard for interactive visualization of unemployment forecasts and scenarios. |
| `world_bank_dataset.csv` | Dataset containing South Africa's unemployment data from the World Bank. |

---

## Dataset

The dataset used is from the **World Bank Development Indicators**:

- **Indicator:** Unemployment, total (% of total labor force)
- **Years:** 2010–2022
- **Link:** [World Bank Dataset on Kaggle](https://www.kaggle.com/datasets/bhadramohit/world-bank-dataset)  

> ⚠️ The CSV file `world_bank_dataset.csv` should be placed in the same folder as the notebook and `app.py` to run the code successfully.

---

## Notebook

The Jupyter Notebook contains:

1. **Introduction and Task Justification**
2. **Dataset Loading and Preprocessing**
3. **Exploratory Data Analysis (EDA)**
4. **Time Series Forecasting using Prophet**
5. **Scenario Analysis (Optimistic, Pessimistic, Base Case)**
6. **Conclusions and Recommendations**
7. **Technical Documentation**

You can view the notebook here: [(https://colab.research.google.com/drive/1Elcew_T2hMKchs172-b0XE5kC6kCQj7R?usp=sharing)]

---

## Streamlit Dashboard

Run the interactive dashboard locally:

```bash
# Install dependencies (if not already installed)
pip install pandas numpy matplotlib seaborn prophet scikit-learn streamlit

# Launch dashboard
streamlit run app.py
