# Global Air Pollution AQI Analysis

## Project Overview
This project analyzes a global air pollution dataset and builds machine learning models to classify the overall AQI category based on pollutant measurements and location-related features. In addition to predictive modeling, the project also includes a statistical analysis component to examine pollutant relationships and country-level AQI differences.

## Dataset
Source: Global Air Pollution Dataset on Kaggle  
Dataset link: https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset

The dataset contains city-level air quality observations, including:
- Country
- City
- AQI category
- Pollutant-specific AQI values for CO, Ozone, NO2, and PM2.5

## Project Goal
The goal of this project is to classify the overall AQI category using pollutant AQI measurements and related categorical features, while also performing statistical analyses to better understand patterns in air pollution across locations.

## Methods

### Data preprocessing
- Removed duplicates
- Standardized text fields
- Converted AQI-related columns to numeric format
- Filtered valid AQI category labels

### Feature engineering
The current modeling version uses:
- Country
- City
- CO AQI Value
- Ozone AQI Value
- NO2 AQI Value
- PM2.5 AQI Value
- Dominant Pollutant

### Baseline model
- DummyClassifier with `most_frequent` strategy

### Machine learning models
- Logistic Regression
- Random Forest

### Statistical analysis component
- Spearman correlation analysis between overall AQI and pollutant AQI values
- Kruskal-Wallis test for AQI differences across top countries
- Bootstrap confidence interval for mean AQI

## Current Milestone 1 Results
The baseline model performed poorly, as expected, while Logistic Regression produced much stronger classification results, indicating that pollutant AQI measurements are highly informative for overall AQI category classification.

## Repository Structure
```text
data/
notebooks/
outputs/
README.md
requirements.txt

