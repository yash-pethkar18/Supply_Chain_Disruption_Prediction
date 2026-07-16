import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

file_path = 'kaggle_supply_chain.csv'

if not os.path.exists(file_path):
    print(f"⚠️ ERROR: Could not find '{file_path}'. Please ensure the CSV is in the same folder.")
    exit()

print("Loading dataset...")
df = pd.read_csv(file_path)

print("Cleaning data and handling missing values...")

df['Supplier_Rating'] = df['Supplier_Rating'].fillna(df['Supplier_Rating'].median())
df['Distance_km'] = df['Distance_km'].fillna(df['Distance_km'].median())

df['Weather_Factor'] = df['Weather_Factor'].fillna(df['Weather_Factor'].mode()[0])
df['Vehicle_Type'] = df['Vehicle_Type'].fillna(df['Vehicle_Type'].mode()[0])

df = df[df['Distance_km'] < 10000]

weather_mapping = {
    'Sunny': 0, 
    'Clear': 0, 
    'Rain': 1, 
    'Storm': 1, 
    'Snow': 1
}

vehicle_mapping = {
    'Bike': 0, 
    'Car': 1, 
    'Van': 1, 
    'Truck': 2
}

df['Weather_Factor'] = df['Weather_Factor'].map(weather_mapping)
df['Vehicle_Type'] = df['Vehicle_Type'].map(vehicle_mapping)

print("Training the Random Forest Algorithm...")

X = df[['Supplier_Rating', 'Distance_km', 'Weather_Factor', 'Traffic_Density', 'Vehicle_Type']]
y = df['Disruption']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train the Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("-" * 40)
print(f"✅ Data processing complete! ({len(df)} rows used)")
print(f"📊 Model Accuracy on Test Data: {round(accuracy * 100, 2)}%")
print("✅ AI Brain successfully saved as 'model.pkl'")
print("-" * 40)