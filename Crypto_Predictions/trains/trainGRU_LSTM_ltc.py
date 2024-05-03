#importer pandas pour manipuler les donnÃ©es
import pandas as pd 
import os
#charger les donnÃ©es .csv dans une DataFrame 

# Chemin du répertoire contenant les données
data_dir = 'C:\\Users\\asus\\Desktop\\Crypto_Predictions\\data'
folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
folders.sort(key=lambda x: os.path.getctime(os.path.join(data_dir, x)), reverse=True)
recent_folder = folders[0]
btc_data_path = os.path.join(data_dir, recent_folder, 'LTC_data.csv')
df = pd.read_csv(btc_data_path)

import pandas as pd
import numpy as np #pour les opÃ©rations sur les tableaux numÃ©riques
from sklearn.model_selection import train_test_split #pour diviser l'ensemble train et test
from tensorflow.keras.models import Sequential #construire un modÃ¨le avec des couches sÃ©quentielle 
from tensorflow.keras.layers import LSTM, Dense , GRU #deux types de modÃ¨le RNN 
                                                    #la couche Dense est une couche connectÃ©e 
from sklearn.preprocessing import MinMaxScaler #pour la normalisation des donnÃ©es entre 0 et 1 
from sklearn.metrics import mean_absolute_error #mesure la diffÃ©rence moyenne absolue entre 
                                                #les valeurs prÃ©dites et les valeurs rÃ©elles
from datetime import datetime, timedelta

#Convertir la colonne date en format datetime pour faciliter la manipulation
df["date"] = pd.to_datetime(df["date"])
#toutes les colonnes sans la colonne date (remplacée avec les séquences)
features = df.drop("date", axis=1)

#Instance pour la normalisation des inputs pour le modèle entre 0 et 1 (echelle commune)
scaler = MinMaxScaler()
#normaliser les features 
features=scaler.fit_transform(np.array(features).reshape(-1,5))

print(features.shape)
#definir le train 90% de l'ensemble des données 
training_size=int(len(features)*0.90)
#le reste pour le test
test_size=len(features)-training_size
#division en train et test 
#train data reçoit à partir de la premiere ligne jusqu'à 90% des lignes (avec toutes les colonnes)
#test data reçoit 10% des données restante jusqu'à la fin, avec toute les colonnes
train_data,test_data=features[0:training_size,:],features[training_size:len(features),:]

import numpy as np

#préparer les données pour le train (série temporelle en séquences)
def create_dataset(dataset, time_step=24*30): #time-step un mois de recul pour voir
    dataX, dataY = [], [] #initialisation
    #boucle pour itérer la dataset (longueur de dataset - un mois - 1 ) la valeur avant dernière
    #0 --> 8760 - 24*30 - 1 dans chaque step
    for i in range(len(dataset)-time_step-1): 
        #extraction d'une séquence du dataset 
        #en commençant à l'index `i` et en s'étendant jusqu'à `i+time_step`
        #exemple pour i=0, on va prendre à partir de la ligne 0 jusuq'à ligne 24*30 - 1 (toutes les colonnes)
        a = dataset[i:(i+time_step), :]   #entrée / caractéristique (train seq)
        #ajout à la variable dataX 
        dataX.append(a)
        dataY.append(dataset[i + time_step, :]) #target à prédire (la ligne juste après)
                                                #une seule ligne et toutes les colonnes (seq sortie)
    return np.array(dataX), np.array(dataY)
time_step=24*30
#création de l'ensemble des séquences de train sur les données de train
X_train, y_train = create_dataset(train_data, time_step)
#création de l'ensemble des séquences de test sur les données de test
X_test, y_test = create_dataset(test_data, time_step)
#préparer les inputs au modèle sous la forme 
#(nombre d'échantillons, nombre de pas de temps, nombre de features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

model_gru_lstm = Sequential()
model_gru_lstm.add(GRU(10, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_gru_lstm.add(LSTM(10)) #32
model_gru_lstm.add(Dense(features.shape[1]))
model_gru_lstm.compile(loss="mean_absolute_error", optimizer="adam")
model_gru_lstm.fit(X_train, y_train, epochs=100, batch_size=64)


def save_model(model, model_name):
    # Chemin du répertoire contenant les sauvegardes
    saves_dir = 'C:\\Users\\asus\\Desktop\\Crypto_Predictions\\saves'
    # Récupérer le numéro de la semaine dans le mois
    week_number = datetime.now().isocalendar()[1]
    # Nom du dossier
    folder_name = f'modele_semaine{week_number}'
    # Chemin complet du dossier
    folder_path = os.path.join(saves_dir, folder_name)
    # Créer le dossier s'il n'existe pas
    os.makedirs(folder_path, exist_ok=True)
    # Chemin complet de sauvegarde du modèle
    save_path = os.path.join(folder_path, f'model_gru_lstm_ltc.h5')
    # Enregistrer le modèle
    model.save(save_path)
    print(f"Le modèle a été enregistré dans {save_path}")
# Utilisation de la fonction
save_model(model_gru_lstm, 'model_gru_lstm_btc')
