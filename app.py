
# ============================================================
# app.py
# Version corrigée — prédiction CNN + LSTM
# Fix : augmentation désactivée en inférence
# ============================================================

import io
import base64
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.cnn_model import CustomCNN
from models.rnn_model import StockLSTM
from utils.data_loader_rnn import SEQUENCE_LENGTH

# ============================================================
# Configuration Flask
# ============================================================

app  = Flask(__name__)
CORS(app)

# Noms des 10 classes CIFAR-10 dans l'ordre officiel Keras
CLASS_NAMES = [
    'avion',       # 0
    'automobile',  # 1
    'oiseau',      # 2
    'chat',        # 3
    'cerf',        # 4
    'chien',       # 5
    'grenouille',  # 6
    'cheval',      # 7
    'bateau',      # 8
    'camion'       # 9
]

CLASS_EMOJIS = {
    'avion':      '',
    'automobile': '',
    'oiseau':     '',
    'chat':       '',
    'cerf':       '',
    'chien':      '',
    'grenouille': '',
    'cheval':     '',
    'bateau':     '',
    'camion':     ''
}

# ============================================================
# Chargement CNN
# ============================================================

print("[app] Chargement du modele CNN...")

cnn_model = CustomCNN(num_classes=10)

# Warm-up OBLIGATOIRE avec training=False
# Initialise les couches BatchNorm en mode inférence
dummy_cnn = tf.zeros((1, 32, 32, 3))
cnn_model(dummy_cnn, training=False)

cnn_model.load_weights("models/cifar10_cnn.keras")

# Compilation pour évaluation éventuelle
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print("[app] Modele CNN pret !")

# ============================================================
# Chargement LSTM
# ============================================================

print("[app] Chargement du modele LSTM...")

lstm_model = StockLSTM(units_1=128, units_2=64, dropout_rate=0.3)
dummy_lstm = tf.zeros((1, SEQUENCE_LENGTH, 1))
lstm_model(dummy_lstm, training=False)
lstm_model.load_weights("models/tsla_lstm.keras")

# Chargement données TSLA + scaler
df_tsla    = pd.read_csv("data/TSLA_2019_2024.csv",
                          index_col=0, parse_dates=True)
prices_all = df_tsla['High'].values.astype('float32')
lstm_scaler = MinMaxScaler(feature_range=(0, 1))
lstm_scaler.fit(prices_all.reshape(-1, 1))

print("[app] Modele LSTM pret !")

# ============================================================
# Route principale
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

# ============================================================
# Route : /predict — Classification CNN
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Reçoit une image, la prétraite et retourne
    les probabilités de classification CNN.
    """

    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image reçue'}), 400

    file = request.files['image']

    try:
        # --------------------------------------------------
        # 1. Lecture et redimensionnement
        # --------------------------------------------------
        img = Image.open(file.stream)

        # Conversion RGB obligatoire
        img = img.convert('RGB')

        # Redimensionnement 32×32 (taille CIFAR-10)
        img = img.resize((32, 32), Image.LANCZOS)

        # --------------------------------------------------
        # 2. Conversion numpy + normalisation
        # Identique au prétraitement de data_loader.py
        # --------------------------------------------------

        # Conversion en float32
        img_array = np.array(img, dtype=np.float32)

        # Normalisation pixels [0,255] → [0.0, 1.0]
        img_array = img_array / 255.0

        # Ajout dimension batch : (32,32,3) → (1,32,32,3)
        img_batch = np.expand_dims(img_array, axis=0)

        # Conversion en tenseur TensorFlow
        img_tensor = tf.constant(img_batch, dtype=tf.float32)

        # --------------------------------------------------
        # 3. Prédiction
        # training=False désactive ABSOLUMENT :
        #   - Dropout (pas de neurones désactivés)
        #   - BatchNormalization (utilise stats globales)
        #   - Data Augmentation (pas de flip/rotation)
        # Sans training=False → toujours classe 0 !
        # --------------------------------------------------
        logits = cnn_model(img_tensor, training=False)

        # Softmax : logits bruts → probabilités [0,1]
        probabilities = tf.nn.softmax(logits).numpy()[0]

        # Classe prédite = indice avec probabilité max
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # --------------------------------------------------
        # 4. Construction du résultat JSON
        # --------------------------------------------------
        all_predictions = [
            {
                'class':       CLASS_NAMES[i],
                'emoji':       CLASS_EMOJIS[CLASS_NAMES[i]],
                'probability': float(probabilities[i]),
                'percentage':  f"{float(probabilities[i]) * 100:.1f}%"
            }
            for i in range(len(CLASS_NAMES))
        ]

        # Tri décroissant par probabilité
        all_predictions.sort(
            key=lambda x: x['probability'], reverse=True
        )

        # --------------------------------------------------
        # 5. Preview base64 de l'image 32×32
        # --------------------------------------------------
        buffer = io.BytesIO()
        preview = Image.fromarray(
            (img_array * 255).astype(np.uint8)
        )
        # Agrandissement 32→160 pour l'affichage
        preview = preview.resize((160, 160), Image.NEAREST)
        preview.save(buffer, format='PNG')
        img_b64 = base64.b64encode(
            buffer.getvalue()
        ).decode('utf-8')

        # Log terminal pour débogage
        print(f"[predict] Classe : {predicted_class} "
              f"({confidence*100:.1f}%) | "
              f"Top3 : {[p['class'] for p in all_predictions[:3]]}")

        return jsonify({
            'success':         True,
            'predicted_class': predicted_class,
            'emoji':           CLASS_EMOJIS[predicted_class],
            'confidence':      confidence,
            'confidence_pct':  f"{confidence * 100:.1f}%",
            'all_predictions': all_predictions,
            'preview_image':   f"data:image/png;base64,{img_b64}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================
# Route : /predict_lstm
# ============================================================

@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    try:
        prices        = df_tsla['High'].values.astype('float32')
        prices_scaled = lstm_scaler.transform(
            prices.reshape(-1, 1)
        ).flatten()

        last_window = prices_scaled[-SEQUENCE_LENGTH:]
        last_window = tf.constant(
            last_window.reshape(1, SEQUENCE_LENGTH, 1),
            dtype=tf.float32
        )

        pred_scaled = lstm_model(last_window, training=False).numpy()[0][0]
        pred_price = float(lstm_scaler.inverse_transform([[pred_scaled]])[0][0])
        last_price = float(prices[-1])
        last_date = str(df_tsla.index[-1].date())
        variation = ((pred_price - last_price) / last_price) * 100
        is_up = variation > 0

        return jsonify({
            'success':         True,
            'predicted_price': round(pred_price, 2),
            'last_price':      round(last_price, 2),
            'last_date':       last_date,
            'variation':       round(float(variation), 2),
            'is_up':           bool(is_up),
            'signal':          'HAUSSE ↑' if is_up else 'BAISSE ↓',
            'confidence':      round(float(1 - abs(pred_scaled - 0.5) * 0.5), 3)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================
# Route : /lstm_history
# ============================================================

@app.route('/lstm_history')
def lstm_history():
    try:
        df_recent = df_tsla.tail(120)
        dates = [str(d.date()) for d in df_recent.index]
        prices = [round(float(p), 2) for p in df_recent['High'].values]

        return jsonify({
            'success': True,
            'dates':   dates,
            'prices':  prices,
            'stats': {
                'min':  round(float(prices_all.min()), 2),
                'max':  round(float(prices_all.max()), 2),
                'mean': round(float(prices_all.mean()), 2),
                'last': round(float(prices_all[-1]), 2),
                'mae':  11.04,
                'mape': 7.91,
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# Route : /stats
# ============================================================

@app.route('/stats')
def stats():
    return jsonify({
        'test_accuracy':  84.83,
        'test_loss':      0.4599,
        'total_params':   '1.5M',
        'epochs_trained': 50,
        'batch_size':     64,
        'optimizer':      'Adam',
        'architecture':   'CustomCNN',
        'dataset':        'CIFAR-10',
        'num_classes':    10,
        'class_scores': [
            {'class': 'avion',      'precision': 86.5},
            {'class': 'automobile', 'precision': 90.6},
            {'class': 'oiseau',     'precision': 82.7},
            {'class': 'chat',       'precision': 78.0},
            {'class': 'cerf',       'precision': 82.0},
            {'class': 'chien',      'precision': 84.3},
            {'class': 'grenouille', 'precision': 79.4},
            {'class': 'cheval',     'precision': 90.6},
            {'class': 'bateau',     'precision': 91.3},
            {'class': 'camion',     'precision': 82.9},
        ]
    })

# ============================================================
# Lancement
# ============================================================


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
