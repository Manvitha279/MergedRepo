import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the dataset from the file
df = pd.read_csv("Unemployment in India.csv")  # Assuming the dataset is in a CSV file named "Unemployment in India.csv"

# Check if 'Year' column exists in the DataFrame
if 'Year' not in df.columns:
    raise KeyError("'Year' column not found in the dataset. Please check the column names.")

# Step 2: Identify the time period corresponding to COVID-19 outbreak (assuming 2019 as the COVID-19 period)
covid_period_start = 2019
covid_period_end = 2020

# Step 3: Analyze the unemployment rate during the COVID-19 period
covid_data = df[(df['Year'] >= covid_period_start) & (df['Year'] <= covid_period_end)]

sharp_increase = covid_data['UnemploymentRate'].max() - covid_data['UnemploymentRate'].min()
print(f"Sharp increase in unemployment rate during COVID-19 period: {sharp_increase}")

# Step 4: Visualize the unemployment rate data during the COVID-19 period
plt.figure(figsize=(10, 6))
plt.plot(covid_data['Year'], covid_data['UnemploymentRate'], marker='o')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate during COVID-19 Period')
plt.grid(True)
plt.show()