# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 18:22:34 2025

@author: sulak
"""


# 1. Installing and importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#2. Downloaded the world happiness report dataset from Kaggle.com and loaded the dataset using pandas.
df = pd.read_csv('world-happiness-report.csv')

#3. Exploring the dataset- checking for the details and structure of the dataset

#Getting the basic information about the dataset
df.info()

#Displaying first few rows
df.head()

#Check column names
df.columns

#describing the dataset to see the min, max.
description = df.describe()


# Listing all attributes (columns) and their datatypes in the dataset
data_types = df.dtypes
print('The data types in the datasets are:', data_types)

# Categorising variables as ordinal, nominal, interval or ratio
value_types = {} 
for col in df.columns: 
    if df[col].dtype == 'object': 
        value_types[col] = 'Nominal' 
    elif df[col].dtype == 'int64' or df[col].dtype == 'float64': 
        value_types[col] = 'Ratio' if df[col].min() >= 0 else 'Interval' 
    else: 
        value_types[col] = 'Ordinal' 
        
# Convert dictionary to a DataFrame
table = pd.DataFrame(list(value_types.items()), columns=['Column Name', 'Data Type'])

# Display the table
print(table)

# 4. Data Cleaning 

missing_data = df.isnull().sum()
print("missing data in the file are: ", missing_data)

#Total number of records before cleaning
print("Before Cleaning:\n", df, "\n")

#Imputing records containing NaN(null values)
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

print(df.isnull().sum())  # Should show 0 missing values for numerical columns

#Total number of records after cleaning
print("After Cleaning:\n", df)

# Understand the happiness score distribution
#distribution of the happiness score (life ladder)
plt.figure(figsize=(10, 5))
sns.histplot(df["Life Ladder"], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Happiness Scores Across Countries")
plt.xlabel("Happiness Score")
plt.ylabel("Frequency")
plt.show()

#Analysing GDP vs happiness

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["Log GDP per capita"], y=df["Life Ladder"], alpha=0.7)
plt.title("GDP per Capita vs Happiness Score")
plt.xlabel("Log GDP per Capita")
plt.ylabel("Happiness Score")
plt.show()

#Correlation heatmap: to see how different factors impact happiness.
plt.figure(figsize=(10, 6))
corr = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Happiness Factors")
plt.show()

# comparing happiness scores by country

# Sort countries by Happiness Score (Life Ladder)
df_sorted = df[['Country name', 'Life Ladder']].groupby("Country name").mean().reset_index()
df_sorted = df_sorted.sort_values(by="Life Ladder", ascending=False)

# Select Top 10 & Bottom 10
top_10 = df_sorted.head(10)
bottom_10 = df_sorted.tail(10)

# Merge top & bottom
df_filtered = pd.concat([top_10, bottom_10])

# Barplot
plt.figure(figsize=(12, 6))
sns.barplot(x="Life Ladder", y="Country name", data=df_filtered, palette="coolwarm")
plt.title("Top 10 & Bottom 10 Happiest Countries")
plt.xlabel("Happiness Score")
plt.ylabel("Country")
plt.show()

#Tracking happiness trends over time
# Find the top 10 happiest and bottom 10 least happy countries (on average)
df_avg_happiness = df.groupby("Country name")["Life Ladder"].mean().reset_index()
top_10 = df_avg_happiness.nlargest(10, "Life Ladder")["Country name"]
bottom_10 = df_avg_happiness.nsmallest(10, "Life Ladder")["Country name"]

# Filter dataset for only these countries
df_filtered = df[df["Country name"].isin(top_10) | df["Country name"].isin(bottom_10)]

# Plot line chart
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_filtered, x="year", y="Life Ladder", hue="Country name")
plt.title("Happiness Trends for Top & Bottom 10 Countries")
plt.xlabel("Year")
plt.ylabel("Happiness Score")
plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#Feature Engineering
# 1. Economic & Social Ratios
df['Happiness_per_GDP'] = df['Life Ladder'] / df['Log GDP per capita']
df['SocialSupport_to_GDP'] = df['Social support'] / df['Log GDP per capita']
df['LifeExpectancy_per_GDP'] = df['Healthy life expectancy at birth'] / df['Log GDP per capita']

# 2. Psychological & Emotional Features
df['Generosity_minus_Corruption'] = df['Generosity'] - df['Perceptions of corruption']
df['Emotional_Balance'] = df['Positive affect'] - df['Negative affect']
df['Freedom_minus_Corruption'] = df['Freedom to make life choices'] - df['Perceptions of corruption']

# 3. Time-Based Features
df['Happiness_Change'] = df.groupby('Country name')['Life Ladder'].diff()
df['Decade'] = (df['year'] // 10) * 10

# 4. Ranking & Aggregation Features
df['Happiness_Rank'] = df['Life Ladder'].rank(ascending=False)
df['Region_Avg_Happiness'] = df.groupby('Country name')['Life Ladder'].transform('mean')

#Visualising the derived features
#1.Happiness vs. GDP Efficiency
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="Log GDP per capita", y="Happiness_per_GDP", hue="Country name", alpha=0.7)
plt.xlabel("Log GDP per capita")
plt.ylabel("Happiness per GDP Unit")
plt.title("Happiness Efficiency vs. GDP per Capita")
plt.show()
#Countries above the trend line achieve high happiness even with lower GDP
#Countries below the line have GDP but lower happiness efficiency


#2. Emotional balance:Positivity vs. Negatively
plt.figure(figsize=(12, 5))
sns.boxplot(x="Decade", y="Emotional_Balance", data=df)
plt.title("Emotional Balance Trends Over Time")
plt.xlabel("Decade")
plt.ylabel("Emotional Balance (Positive - Negative Affect)")
plt.show()
# A postive balance means happier emotional states over time.
#A decreasing trend may suggest increasing stress or dissatisfaction in modern times.

# Compute correlation matrix
corr_matrix = df[['Life Ladder', 'Social support', 'Log GDP per capita', 'Healthy life expectancy at birth', 'Freedom to make life choices', 'Perceptions of corruption']].corr()

# Heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Happiness Factors")
plt.show()

#Saving the cleaned data and visualisations
df.to_csv("cleaned_happiness_report.csv", index=False)
plt.savefig("happiness_trends.png")


































