import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
# Make sure fetal_health.csv is in the same folder!
df = pd.read_csv('data/fetal_health.csv')

# 2. Check the first few rows
print("First 5 rows of data:")
print(df.head())

# 3. Check for missing values (Medical data must be complete)
print("\nMissing values:")
print(df.isnull().sum().sum())

# 4. Visualize the Target (Fetal Health)
# 1.0 = Normal, 2.0 = Suspect, 3.0 = Pathological
plt.figure(figsize=(8, 5))
sns.countplot(x='fetal_health', data=df, palette='viridis')
plt.title('Distribution of Fetal Health Classes')
plt.show()