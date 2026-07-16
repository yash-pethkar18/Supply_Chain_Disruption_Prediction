import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = {
    'Supplier_Rating': [5, 2, 4, 1, 5, 3, 2, 4, 1, 5, 4, 2, 3, 1, 5, 2, 4, 3, 1, 5],
    'Distance_km': [100, 800, 150, 1200, 50, 400, 900, 200, 1500, 80, 300, 600, 450, 1100, 120, 750, 250, 500, 1300, 90],
    'Weather_Factor': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    'Traffic_Density': [2, 8, 3, 9, 1, 6, 9, 2, 10, 2, 4, 7, 5, 9, 1, 8, 3, 4, 10, 2],
    'Disruption': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.tight_layout()


plt.savefig('static/images/correlation_graph.png')
print("Graph saved successfully!")