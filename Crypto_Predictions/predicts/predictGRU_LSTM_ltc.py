from keras.models import load_model
import os
from datetime import datetime
from keras.losses import mean_absolute_error
import tensorflow as tf
from datetime import datetime
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def import_model():
    # Chemin du répertoire contenant les sauvegardes
    saves_dir = 'C:\\Users\\asus\\Desktop\\Crypto_Predictions\\saves'
    # Récupérer le numéro de la semaine dans le mois
    week_number = datetime.now().isocalendar()[1]
    # Nom du dossier correspondant à la semaine actuelle
    current_week_folder = f'modele_semaine{week_number}'
    # Chemin complet du dossier correspondant à la semaine actuelle
    current_week_folder_path = os.path.join(saves_dir, current_week_folder)
    # Chemin complet du dossier modele_existant
    existing_models_folder_path = os.path.join(saves_dir, 'modele_existant')
    # Chemin complet du modèle à importer
    model_path = ''
    # Vérifier si le dossier correspondant à la semaine actuelle existe
    if os.path.exists(current_week_folder_path):
        model_path = os.path.join(current_week_folder_path, 'model_gru_lstm_ltc.h5')
    else:
        model_path = os.path.join(existing_models_folder_path, 'model_gru_lstm_ltc.h5')
    # Importer le modèle
    gru_lstm_model = load_model(model_path)
    return gru_lstm_model

gru_lstm_model = import_model()
max_date = datetime(2024, 4, 24,0,0,0)


def predict_future_value(model, future_date, max_date):
    
    # Chemin du répertoire contenant les données
    data_dir = 'C:\\Users\\asus\\Desktop\\Crypto_Predictions\\data'
    # Liste des dossiers dans le répertoire
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    # Trie les dossiers par date de création (le plus récent en premier)
    folders.sort(key=lambda x: os.path.getctime(os.path.join(data_dir, x)), reverse=True)
    # Choix du dossier le plus récent
    recent_folder = folders[0]
    # Chemin du fichier BNB_data.csv dans le dossier récent
    bnb_data_path = os.path.join(data_dir, recent_folder, 'LTC_data.csv')
    # Charger les données BNB_data.csv
    df = pd.read_csv(bnb_data_path)
    print(f"Les données BNB les plus récentes ont été chargées à partir de {bnb_data_path}")
    
    
    df["date"] = pd.to_datetime(df["date"])
    scaler = MinMaxScaler()
    features = df.drop("date", axis=1)  
    scaled_features = scaler.fit_transform(features)
    max_date = df["date"].max()
    look_back = 24*30
    days_to_predict = (future_date - max_date).total_seconds() / 3600

    future_sequence = scaled_features[-look_back:, :]
    for _ in range(int(days_to_predict)):
        predicted_scaled_value = model.predict(future_sequence.reshape(1, look_back, scaled_features.shape[1]))
        future_sequence = np.append(future_sequence[1:, :], predicted_scaled_value, axis=0)
        predicted_values = scaler.inverse_transform(predicted_scaled_value)

    return predicted_values.flatten()


if __name__ == '__main__':
       
    future_date = datetime(2024, 4, 30, 18, 0, 0)
    max_date = datetime(2024, 4, 24,0,0,0)
    predicted_values = predict_future_value(gru_lstm_model, future_date,max_date)
    price_pred, high_pred, low_pred, volumefrom_pred, volumeto_pred = predicted_values
    print("Predicted values for", future_date, "by model_lstm:")
    print("Price:", price_pred)
    print("High:", high_pred)
    print("Low:", low_pred)
    print("Volume From:", volumefrom_pred)
    print("Volume To:", volumeto_pred)
