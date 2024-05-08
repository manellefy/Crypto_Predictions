import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# Chemin du répertoire contenant les données
data_dir = 'C:\\Users\\asus\\Desktop\\Crypto_Predictions\\data'
folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
folders.sort(key=lambda x: os.path.getctime(os.path.join(data_dir, x)), reverse=True)
recent_folder = folders[0]
btc_data_path = os.path.join(data_dir, recent_folder, 'BNB_data.csv')
df = pd.read_csv(btc_data_path, encoding='utf-8')  # Ensure utf-8 encoding is used

# Convertir la colonne date en format datetime pour faciliter la manipulation
df["date"] = pd.to_datetime(df["date"])

# Toutes les colonnes sans la colonne date (remplacée avec les séquences)
features = df.drop("date", axis=1)

# Instance pour la normalisation des inputs pour le modèle entre 0 et 1 (échelle commune)
scaler = MinMaxScaler()
# Normaliser les features
features = scaler.fit_transform(np.array(features).reshape(-1, features.shape[1]))

print(features.shape)

# Définir le train 90% de l'ensemble des données 
training_size = int(len(features) * 0.90)
# Le reste pour le test
test_size = len(features) - training_size
# Division en train et test 
train_data, test_data = features[0:training_size, :], features[training_size:len(features), :]

# Préparer les données pour le train (série temporelle en séquences)
def create_dataset(dataset, time_step=24*30):  # time-step un mois de recul pour voir
    dataX, dataY = [], []  # Initialisation
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), :]  # Entrée / caractéristique (train seq)
        dataX.append(a)
        dataY.append(dataset[i + time_step, :])  # Target à prédire (la ligne juste après)
    return np.array(dataX), np.array(dataY)

time_step = 24*30
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

model_gru_lstm = Sequential()
model_gru_lstm.add(GRU(10, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_gru_lstm.add(LSTM(10))
model_gru_lstm.add(Dense(features.shape[1]))
model_gru_lstm.compile(loss="mean_absolute_error", optimizer="adam")
model_gru_lstm.fit(X_train, y_train, epochs=100, batch_size=64)

def save_model(model, model_name):
    saves_dir = 'C:\\Users\\asus\\Desktop\\Crypto_Predictions\\saves'
    week_number = datetime.now().isocalendar()[1]
    folder_name = f'modele_semaine{week_number}'
    folder_path = os.path.join(saves_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, f'{model_name}.h5')
    model.save(save_path)
    print(f"Le modèle a été enregistré dans {save_path}")

save_model(model_gru_lstm, 'model_gru_lstm_bnb')
