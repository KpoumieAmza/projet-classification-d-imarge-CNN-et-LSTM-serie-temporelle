# Deep Learning — CNN + LSTM | CIFAR-10 & TSLA

> Projet Fil Rouge — Dr. Noulapeu N. A.
> TensorFlow 2.15 · Keras · Flask · Python 3.12

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Accuracy](https://img.shields.io/badge/CNN%20Accuracy-84.83%25-green)
![MAPE](https://img.shields.io/badge/LSTM%20MAPE-7.91%25-green)

---

## Présentation

Ce projet implémente deux modèles de Deep Learning complets :

- **Mission 1** — Classification d'images CIFAR-10 avec un CNN personnalisé
- **Mission 2** — Prédiction du prix boursier Tesla (TSLA) avec un LSTM

Les deux modèles sont accessibles via une **interface web dark mode** construite avec Flask + HTML/CSS/JS.

---

## Résultats

| Mission | Modèle | Métrique | Résultat |
|---------|--------|----------|----------|
| CNN | CustomCNN | Test Accuracy | **84.83 %** |
| LSTM | StockLSTM | MAPE | **7.91 %** |

---

## Structure du projet
```
cifar10_cnn/
├── models/
│   ├── cnn_model.py          # CustomCNN — API Subclassing Keras
│   ├── rnn_model.py          # StockLSTM — LSTM double couche
│   ├── cifar10_cnn.keras     # Poids CNN entraînés
│   └── tsla_lstm.keras       # Poids LSTM entraînés
├── utils/
│   ├── data_loader.py        # Pipeline CIFAR-10 tf.data
│   └── data_loader_rnn.py    # Pipeline TSLA + Sliding Window
├── templates/
│   └── index.html            # Interface dark mode
├── static/
│   ├── css/style.css         # Thème dark tech/hacker
│   └── js/main.js            # Logique interactive
├── data/
│   ├── TSLA_2019_2024.csv    # Données Tesla 5 ans
│   ├── training_curves.png   # Courbes CNN
│   ├── confusion_matrix.png  # Matrice de confusion
│   └── rnn_predictions.png   # Courbe réel vs prédit
├── train.py                  # Entraînement CNN
├── train_rnn.py              # Entraînement LSTM
├── evaluate.py               # Évaluation CNN
├── evaluate_rnn.py           # Évaluation LSTM
├── app.py                    # Serveur Flask API
└── requirements.txt          # Dépendances
```

---

## Installation
```bash
# 1. Cloner le dépôt
git clone https://github.com/TON_USERNAME/cifar10_cnn.git
cd cifar10_cnn

# 2. Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### Entraîner les modèles
```bash
# Mission 1 — CNN CIFAR-10
python train.py

# Mission 2 — LSTM Tesla
python train_rnn.py
```

### Évaluer les modèles
```bash
# CNN → matrice de confusion
python evaluate.py

# LSTM → courbe réelle vs prédite
python evaluate_rnn.py
```

### Lancer l'interface web
```bash
python app.py
```

Ouvre `http://127.0.0.1:5000` dans ton navigateur.

---

## Mission 1 — CNN CIFAR-10

### Architecture CustomCNN

| Bloc | Couches | Sortie |
|------|---------|--------|
| Augmentation | RandomFlip · Rotation · Zoom | (B, 32, 32, 3) |
| Conv Bloc 1 | Conv2D(32)×2 + BN + MaxPool + Dropout(0.25) | (B, 16, 16, 32) |
| Conv Bloc 2 | Conv2D(64)×2 + BN + MaxPool + Dropout(0.25) | (B, 8, 8, 64) |
| Conv Bloc 3 | Conv2D(128)×2 + BN + MaxPool + Dropout(0.25) | (B, 4, 4, 128) |
| MLP | Flatten + Dense(512) + Dense(256) + Dropout(0.5) | (B, 256) |
| Sortie | Dense(10) — logits | (B, 10) |

### Performances
```
Test Accuracy : 84.83 %   (objectif : 70 %)
Test Loss     : 0.4599
Macro F1      : 84.6 %
```

---

## Mission 2 — LSTM Tesla

### Architecture StockLSTM
```
Entrée (B, 60, 1)
→ LSTM(128, return_sequences=True) → Dropout(0.3)
→ LSTM(64,  return_sequences=False) → Dropout(0.3)
→ Dense(32, relu) → Dropout(0.1)
→ Dense(1)  ← prix normalisé J+1
```

### Performances
```
MAPE  : 7.91 %
MAE   : $11.04 / jour
RMSE  : $15.14 / jour
Convergence : époque 17 / 30
```

---

## Technologies utilisées

- **TensorFlow 2.15 / Keras** — API Subclassing
- **scikit-learn** — MinMaxScaler, métriques
- **Flask + Flask-CORS** — API REST
- **Pillow** — Traitement d'images
- **Chart.js** — Graphique historique TSLA
- **HTML/CSS/JS** — Interface dark mode

---

## Auteur

Projet réalisé dans le cadre du cours **Deep Learning** — Dr. Noulapeu N. A.
