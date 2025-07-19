# === Import Libraries ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load Dataset ===
df = pd.read_csv("titanic.csv")  # Ensure this file exists
print("Initial Dataset Preview:\n", df.head())

# === Dataset Overview ===
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# === Missing Value Analysis ===
print("\nMissing Values Count:")
print(df.isnull().sum())

# Heatmap of missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# === Handling Missing Values ===
# Fill missing age with median
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)

# Drop 'Cabin' safely if exists
if 'Cabin' in df.columns:
    df.drop(columns='Cabin', inplace=True)

# Drop rows with missing 'Embarked' if it exists
if 'Embarked' in df.columns:
    df.dropna(subset=['Embarked'], inplace=True)

# Confirm missing values are handled
print("\nMissing Values After Cleanup:")
print(df.isnull().sum())

# === Survival Distribution ===
if 'Survived' in df.columns:
    df['Survived'].value_counts().plot.pie(
        autopct='%1.1f%%', 
        labels=['Not Survived', 'Survived'], 
        colors=['lightcoral', 'lightgreen'], 
        startangle=90
    )
    plt.title("Survival Distribution")
    plt.ylabel('')
    plt.show()

    sns.countplot(x='Survived', data=df, palette='Set2')
    plt.title("Survival Count")
    plt.xticks([0, 1], ['Not Survived', 'Survived'])
    plt.show()

# === Survival by Gender ===
if 'Sex' in df.columns and 'Survived' in df.columns:
    sns.countplot(x='Sex', hue='Survived', data=df, palette='pastel')
    plt.title("Survival by Gender")
    plt.legend(labels=['Not Survived', 'Survived'])
    plt.show()

# === Survival by Passenger Class ===
if 'Pclass' in df.columns and 'Survived' in df.columns:
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='coolwarm')
    plt.title("Survival by Passenger Class")
    plt.legend(labels=['Not Survived', 'Survived'])
    plt.show()

# === Age Distribution & Survival ===
if 'Age' in df.columns and 'Survived' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', palette='Accent', bins=30)
    plt.title("Age vs Survival")
    plt.show()

    sns.boxplot(x='Survived', y='Age', data=df, palette='spring')
    plt.title("Age Distribution by Survival")
    plt.show()

# === Heatmap of Feature Correlations ===
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# === Crosstab Analysis ===
if 'Sex' in df.columns and 'Survived' in df.columns:
    print("\nCrosstab - Survival by Sex:")
    print(pd.crosstab(df['Sex'], df['Survived'], normalize='index'))

if 'Pclass' in df.columns and 'Survived' in df.columns:
    print("\nCrosstab - Survival by Pclass:")
    print(pd.crosstab(df['Pclass'], df['Survived'], normalize='index'))

# === Done ===
print("\nEDA Completed.")
