from flask import Flask, request, jsonify
from datetime import datetime
import predictGRU_LSTM_btc as btc
import predictGRU_LSTM_eth as eth
import predictGRU_LSTM_sol as sol
import predictGRU_LSTM_ltc as ltc
import predictGRU_LSTM_bnb as bnb
import predictGRU_LSTM_mkr as mkr

app = Flask(__name__)

# Dictionnaire pour mapper les cryptos à leurs modules respectifs
crypto_models = {
    'btc': btc,
    'eth': eth,
    'sol': sol,
    'ltc': ltc,
    'bnb': bnb,
    'mkr': mkr,
    
}

@app.route('/predict/<crypto>', methods=['POST'])
def get_crypto_prediction(crypto):
    if crypto not in crypto_models:
        return jsonify({'error': f'Unsupported crypto type: {crypto}. Supported types are {list(crypto_models.keys())}'}), 404

    data = request.get_json()
    date_str = data.get('date', None)
    if not date_str:
        return jsonify({'error': 'Missing date in request body'}), 400

    try:
        future_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD HH:MM:SS'}), 400

    try:
        # Sélectionnez le modèle et la fonction de prédiction basée sur le crypto-type
        module = crypto_models[crypto]
        predictions = module.predict_future_value(module.gru_lstm_model, future_date, module.max_date)

        response = {
            "Price": float(predictions[0]),
            "High": float(predictions[1]),
            "Low": float(predictions[2]),
            "Volume From": float(predictions[3]),
            "Volume To": float(predictions[4])
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


