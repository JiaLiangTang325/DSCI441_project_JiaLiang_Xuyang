# Global Air Pollution AQI Analysis and Forecasting

## Project Description
This project analyzes city-level global air pollution data and builds a machine learning model to classify the overall Air Quality Index (AQI) category. The project combines exploratory data analysis, statistical testing, and an interactive Streamlit WebApp for AQI category prediction.

The main project goal is to understand how pollutant-specific AQI values relate to the overall AQI category and to create a usable prediction tool based on pollutant measurements and country information.

## Data Source
Dataset: Global Air Pollution Dataset on Kaggle
Source link: <https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset>

The dataset contains city-level air quality observations, including:

- Country
- City
- Overall AQI value and AQI category
- CO AQI value and category
- Ozone AQI value and category
- NO2 AQI value and category
- PM2.5 AQI value and category

The expected local data path is:

```text
data/global_air_pollution_dataset.csv
```

See `data/readme_data.txt` for instructions on how to obtain the data and where to place it.

## Methods
The analysis includes:

- Removing duplicate records
- Standardizing text fields
- Converting AQI columns to numeric values
- Filtering valid AQI category labels
- Creating a dominant pollutant feature
- AQI category distribution analysis
- Country-level AQI comparisons
- Spearman correlation analysis between overall AQI and pollutant AQI values
- Bootstrap confidence interval estimation for mean AQI
- Random Forest classification for AQI category prediction

## Required Packages
Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Main packages:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- jupyter
- ipykernel

## How to Run the Code
Run the Streamlit WebApp from the project root:

```bash
streamlit run main.py
```

The app provides:

- Dataset overview and filters
- AQI category distribution
- Country-level AQI ranking
- Pollutant relationship charts
- Statistical summary tables
- Random Forest model evaluation
- Interactive AQI category prediction

## Streamlit Cloud Deployment
To create the public demo WebApp link required for submission:

1. Go to <https://share.streamlit.io/>.
2. Sign in with GitHub.
3. Create a new app from this repository.
4. Use branch `main`.
5. Use `main.py` as the main file path.
6. Deploy the app and submit the generated `streamlit.app` URL.

The original notebook analysis is available at:

```text
notebooks/global_air_pollution_aqi_analysis.ipynb
```

## Repository Structure
```text
data/
  global_air_pollution_dataset.csv
  readme_data.txt
notebooks/
  global_air_pollution_aqi_analysis.ipynb
outputs/
  figures/
main.py
aux_1.py
README.md
ReadMe.txt
requirements.txt
```

## GitHub Repository
Direct repository link:

<https://github.com/JiaLiangTang325/DSCI441_project_JiaLiang_Xuyang>

