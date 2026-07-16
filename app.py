from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

gemini_model = genai.GenerativeModel('gemini-2.5-flash')
app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

if not os.path.exists('static/images'):
    os.makedirs('static/images')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get data
    rating = float(request.form['rating'])
    distance = float(request.form['distance'])
    weather = float(request.form['weather'])
    traffic = float(request.form['traffic'])
    vehicle = int(request.form['vehicle']) 

    # 2. Predict using Random Forest
    features = np.array([[rating, distance, weather, traffic, vehicle]])
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    # Map numbers back to text for the Gemini Prompt
    weather_text = "Adverse/Stormy" if weather == 1 else "Optimal/Clear"
    vehicle_dict = {0: "Bike", 1: "Car/Van", 2: "Heavy Truck"}
    vehicle_text = vehicle_dict.get(vehicle, "Unknown")

    # 3. Determine Risk Level
    if prediction[0] == 1:
        result_text = "High Risk of Disruption"
        css_class = "danger"
    else:
        result_text = "Low Risk - Supply Chain Stable"
        css_class = "success"

    prompt = f"""
    You are an expert Supply Chain and Logistics Consultant. 
    Analyze this current shipment:
    - Supplier Reliability Rating: {rating}/5.0
    - Distance: {distance} km
    - Weather Conditions: {weather_text}
    - Traffic Density: {traffic}/10
    - Vehicle Type: {vehicle_text}
    
    The Predictive AI Model has flagged this shipment as: '{result_text}'.
    
    Based on these exact parameters, provide a highly professional, 2-sentence actionable recommendation for the logistics manager. Do not use bold text, stars, or formatting. Just output the two sentences.
    """
    try:
        response = gemini_model.generate_content(prompt)
        suggestion = response.text.strip()
    except Exception as e:
        # Fallback if internet drops or API fails
      suggestion = f"System alert: Monitor this shipment closely. (AI connection error: {str(e)})"


    rating_risk = (5 - rating) * 2
    distance_risk = min((distance / 2000) * 10, 10)
    weather_risk = 9 if weather == 1 else 2
    traffic_risk = traffic

    plot_data = pd.DataFrame({
        'Factor': ['Supplier Risk', 'Distance', 'Weather', 'Traffic'],
        'Severity': [rating_risk, distance_risk, weather_risk, traffic_risk]
    })

    plt.figure(figsize=(8, 4))
    sns.set_theme(style="whitegrid")
    
    colors = ["#1e3a8a" if x < 6 else "#dc2626" for x in plot_data['Severity']]
    ax = sns.barplot(x='Factor', y='Severity', data=plot_data, palette=colors)
    
    plt.axhline(y=5, color='gray', linestyle='--', label='Risk Threshold')
    plt.title('Real-Time Risk Factor Analysis', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 10)
    plt.ylabel('Severity Score (0-10)')
    plt.xlabel('')
    
    timestamp = int(time.time())
    image_path = f'static/images/prediction_plot.png'
    plt.savefig(image_path, bbox_inches='tight', dpi=100)
    plt.close() 


    return render_template('index.html', 
                           prediction_text=result_text, 
                           prediction_class=css_class, 
                           prob=round(probability*100, 2),
                           suggestion_text=suggestion, # THIS IS NOW POWERED BY GEMINI!
                           plot_url=image_path,
                           timestamp=timestamp) 

@app.route('/dashboard')
def dashboard():
    importances = model.feature_importances_
    data_list = list(importances)
    return render_template('dashboard.html', data=data_list)

if __name__ == '__main__':
    app.run(debug=True)