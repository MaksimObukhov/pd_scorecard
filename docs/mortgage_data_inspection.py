# %%
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load the data from a CSV file
data = pd.read_csv('mortgage_sample.csv')
# Check the shape of the dataset
print("Data shape:", data.shape)
print("Columns:", data.columns)

# %%
# Display the first 5 rows of the data
print(data.head())


# %%
# Check the data types of the variables
print("\nData types:\n", data.dtypes)

# %%
# Check for any missing values
print("\nMissing values:\n", data.isnull().sum())

# Count the number of rows with missing values
null_rows = data.isnull().any(axis=1)
print("\nNumber of rows with missing values:", null_rows.sum())

# %%
# Check basic descriptive statistics of the numerical variables

# First, select numerical variables only
num_vars = data.select_dtypes(include=['float64', 'int64'])

print("\nDescriptive statistics:", num_vars.describe())

# %%
# You can also compute each statistic individually

# Count
print("\nCounts:\n", num_vars.count())

# Mean
print("\nMeans:\n", num_vars.mean())

# Standard deviation
print("\nStandard deviations:\n", num_vars.std())

# Minimum
print("\nMinimum values:\n", num_vars.min())

# Maximum
print("\nMaximum values:\n", num_vars.max())

# Quantile
print("\nQuantiles (0.25):\n", num_vars.quantile(q=0.25))

# %%
# Similarly, you can compute individual statistics for a single variable

mean_uer_time = data['uer_time'].mean()
std_uer_time = data['uer_time'].std()
min_uer_time = data['uer_time'].min()

print(f"\nMean of uer_time: {mean_uer_time:.2f}")
print(f"Std deviation of uer_time: {std_uer_time:.2f}")
print(f"Minimum of uer_time: {min_uer_time:.2f}")

# %%
# Visualize variables to see the distribution of their values
# Histogram
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(data['FICO_orig_time'], bins=20)
ax.set_xlabel('FICO score')
ax.set_ylabel('Frequency')
plt.show()

# %%
# Scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['hpi_time'], data['gdp_time'], alpha=0.5)
ax.set_xlabel('House price index')
ax.set_ylabel('GDP growth')
plt.show()

# %%
# Correlation

# Calculate correlation matrix from all numerical variables
corr_matrix = num_vars.corr()

# Plot correlation matrix as a heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
ax.set_title('Correlation Matrix Heatmap')
plt.show()

# %%
# Aggregating data

# Group by home type and calculate the mean FICO score within each home type
home_type_fico_mean = data.groupby(['REtype_SF_orig_time'])['FICO_orig_time'].mean()
print(home_type_fico_mean)

# %%
# Filter by a condition

# Select only loans with a FICO score greater than 700
high_fico_loans = data[data['FICO_orig_time'] > 700]
print(high_fico_loans.head())

# %%
# Missing data

# Fill in missing data with the mean value of the variable
data_mean_imputed = num_vars.fillna(num_vars.mean())
print(data_mean_imputed.isnull().sum())

# %%
# Drop rows with missing data
data_dropped = data.dropna()
print(data_dropped.isnull().sum())

# %%
# Apply a function to a variable

# Define a function to classify loans based on FICO score
def fico_class(row):
    if row['FICO_orig_time'] >= 750:
        return 'excellent'
    elif row['FICO_orig_time'] >= 700:
        return 'good'
    else:
        return 'poor'

# Apply the function to the FICO score column
data['FICO_class'] = data.apply(fico_class, axis=1)
FICO_class_counts = data.groupby(['FICO_class'])['FICO_class'].count()
print(FICO_class_counts)

# %%



