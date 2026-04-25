Global Air Pollution AQI Analysis and Forecasting

Project description
This project analyzes global city-level air pollution data and builds a machine learning model to classify the overall AQI category. The analysis studies AQI distributions, country-level differences, pollutant relationships, and an interactive Random Forest AQI category prediction workflow.

Data source
The dataset is the Global Air Pollution Dataset from Kaggle:
https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset

The expected data file is:
data/global_air_pollution_dataset.csv

The data folder also contains readme_data.txt with instructions for obtaining and placing the dataset.

Required packages
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- jupyter
- ipykernel

Install packages
pip install -r requirements.txt

Run the Streamlit WebApp
streamlit run main.py

Deploy the Streamlit WebApp
1. Go to https://share.streamlit.io/
2. Sign in with GitHub.
3. Choose this repository.
4. Set branch to main.
5. Set main file path to main.py.
6. Deploy and use the generated streamlit.app link for the demo submission.

Repository structure
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
