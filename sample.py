import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('temp_data\weather.csv')
df = df.select_dtypes(exclude=['object'])  # Remove all object dtype columns

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()