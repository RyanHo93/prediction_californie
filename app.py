import streamlit as st
import joblib
import pandas as pd
import numpy as np 

# Charger le modèle
model_path = 'C:/Users/ryanh/OneDrive/Documents/SiliconPred/model_california.pkl'
model = joblib.load(model_path)

# Dictionnaire pour traduire les options de proximité à l'océan
ocean_proximity_translation = {
    'Proche de la baie': 'NEAR BAY',
    'Loin de l\'océan': 'INLAND',
    'Proche de l\'océan': 'NEAR OCEAN',
    'Sur une Île': 'ISLAND',
    'À moins d\'une heure de l\'océan': '<1H OCEAN'
}

# Fonction de prédiction
def predict_price(features):
    # Créer un DataFrame avec les features
    features_df = pd.DataFrame([features])
    
    # Faire la prédiction
    log_prediction = model.predict(features_df)
    
    # Appliquer l'inverse de la transformation logarithmique
    prediction = np.exp(log_prediction)
    
    return prediction[0]

# Interface utilisateur Streamlit
st.title("Prédiction Immobilière Californie")

# Collecter les entrées de l'utilisateur
unnamed_0 = st.number_input('N° d\' appartement', min_value=0)
longitude = st.number_input('Longitude')
latitude = st.number_input('Latitude')
median_income = st.number_input('Revenu médian ($)')

# Afficher les options traduites pour l'utilisateur
ocean_proximity_french = st.selectbox(
    'Proximité de l\'océan', 
    ['Proche de la baie', 'Loin de l\'océan', 'Proche de l\'océan', 'Sur une Île', 'À moins d\'une heure de l\'océan']
)

# Convertir la sélection en anglais pour le modèle
ocean_proximity = ocean_proximity_translation[ocean_proximity_french]

# Lorsque l'utilisateur appuie sur le bouton "Prédire"
if st.button('Prédire'):
    features = {
        'Unnamed: 0': unnamed_0,
        'longitude': longitude,
        'latitude': latitude,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }
    
    # Prédire le prix en utilisant le modèle
    predicted_price = predict_price(features)
    st.write(f"Le prix prédit est : {predicted_price:.2f} $")