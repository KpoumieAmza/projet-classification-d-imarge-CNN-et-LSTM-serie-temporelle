
# ============================================================
# train_rnn.py
# Rôle : Script d'entraînement du modèle LSTM
#        Compilation MSE + Callbacks + model.fit() + Sauvegarde
# Commande : python train_rnn.py
# ============================================================

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')              # Pas de fenêtre GUI
import matplotlib.pyplot as plt
import numpy as np

from models.rnn_model import StockLSTM
from utils.data_loader_rnn import load_stock_data, SEQUENCE_LENGTH


# ============================================================
# Hyperparamètres
# ============================================================

LEARNING_RATE   = 0.001
EPOCHS  = 20             # EarlyStopping arrêtera avant
MODEL_SAVE_PATH = "models/tsla_lstm.keras"


# ============================================================
# Fonction : plot_history_rnn()
# Rôle : Courbes Train Loss vs Validation Loss
# ============================================================

def plot_history_rnn(history):
    """
    Génère et sauvegarde les courbes de perte MSE
    Train vs Validation après l'entraînement.
    """

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    epochs_range = range(1, len(history.history['loss']) + 1)

    # Courbe Train Loss
    ax.plot(
        epochs_range,
        history.history['loss'],
        label='Train MSE',
        color='#00d4ff',
        linewidth=2
    )

    # Courbe Validation Loss
    ax.plot(
        epochs_range,
        history.history['val_loss'],
        label='Validation MSE',
        color='#f59e0b',
        linewidth=2,
        linestyle='--'
    )

    # Marqueur sur la meilleure époque (val_loss minimum)
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val   = min(history.history['val_loss'])
    ax.axvline(
        x=best_epoch,
        color='#10b981',
        linestyle=':',
        linewidth=1.5,
        label=f'Meilleure époque : {best_epoch}'
    )

    ax.set_title('Courbes MSE — StockLSTM TSLA',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Époque', color='white')
    ax.set_ylabel('MSE (Mean Squared Error)', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#1e2d4a')
    ax.spines['top'].set_color('#1e2d4a')
    ax.spines['left'].set_color('#1e2d4a')
    ax.spines['right'].set_color('#1e2d4a')
    ax.legend(facecolor='#12122a', labelcolor='white', fontsize=10)
    ax.grid(True, alpha=0.15, color='#94a3b8')

    plt.tight_layout()

    save_path = "data/rnn_training_curves.png"
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"[train_rnn] Courbes sauvegardées → {save_path}")


# ============================================================
# Fonction principale : train_rnn()
# ============================================================

def train_rnn():

    print("=" * 55)
    print("   ENTRAÎNEMENT — StockLSTM TSLA High")
    print("=" * 55)

    # ----------------------------------------------------------
    # ÉTAPE 1 : Chargement des données
    # ----------------------------------------------------------
    print("\n[1/5] Chargement des données TSLA...")
    train_ds, val_ds, test_ds, scaler, df, X_test, y_test = load_stock_data()
    print("      → Pipeline tf.data.Dataset prêt !")

    # ----------------------------------------------------------
    # ÉTAPE 2 : Instanciation du modèle
    # ----------------------------------------------------------
    print("\n[2/5] Création du modèle StockLSTM...")

    model = StockLSTM(
        units_1      = 128,   # Neurones LSTM couche 1
        units_2      = 64,    # Neurones LSTM couche 2
        dropout_rate = 0.3    # Taux de dropout
    )

    # Warm-up pour initialiser les poids
    # (batch=1, sequence=60 jours, features=1)
    dummy = tf.zeros((1, SEQUENCE_LENGTH, 1))
    model(dummy, training=False)

    print(f"      → Modèle créé : {model.count_params():,} paramètres")

    # ----------------------------------------------------------
    # ÉTAPE 3 : Compilation
    # ----------------------------------------------------------
    print("\n[3/5] Compilation du modèle...")

    model.compile(
        # Adam : optimiseur adaptatif standard
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),

        # MSE : Mean Squared Error — fonction de perte pour régression
        # Pénalise fortement les grandes erreurs de prédiction
        loss=tf.keras.losses.MeanSquaredError(),

        # MAE en métrique : plus interprétable que MSE
        # Erreur absolue moyenne en unités normalisées
        metrics=[
            tf.keras.metrics.MeanSquaredError(name='mse'),
            tf.keras.metrics.MeanAbsoluteError(name='mae')
        ]
    )
    print("      → Compilation OK (Adam + MSE)")

    # ----------------------------------------------------------
    # ÉTAPE 4 : Callbacks
    # ----------------------------------------------------------
    print("\n[4/5] Configuration des Callbacks...")

    # EarlyStopping : arrête si val_loss ne s'améliore plus
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,               # Plus patient qu'avec le CNN
        restore_best_weights=True, # Restaure le meilleur modèle
        verbose=1
    )

    # ModelCheckpoint : sauvegarde le meilleur modèle
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',        # Sur val_loss pour la régression
        save_best_only=True,
        verbose=1
    )

    # ReduceLROnPlateau : réduit le LR si stagnation
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,                # Divise par 2
        patience=6,
        min_lr=1e-6,
        verbose=1
    )

    print("      → EarlyStopping(12) + ModelCheckpoint + ReduceLROnPlateau")

    # ----------------------------------------------------------
    # ÉTAPE 5 : Entraînement
    # ----------------------------------------------------------
    print("\n[5/5] Lancement de l'entraînement...\n")

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1
    )

    # ----------------------------------------------------------
    # Résultats finaux
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("   ENTRAÎNEMENT TERMINÉ")
    print("=" * 55)

    best_val_loss = min(history.history['val_loss'])
    best_val_mae  = min(history.history['val_mae'])
    best_epoch    = np.argmin(history.history['val_loss']) + 1

    print(f"\n   Meilleure époque      : {best_epoch}")
    print(f"   Meilleur val_MSE      : {best_val_loss:.6f}")
    print(f"   Meilleur val_MAE      : {best_val_mae:.6f}")
    print(f"   Modèle sauvegardé     : {MODEL_SAVE_PATH}")

    # Génération des courbes
    plot_history_rnn(history)

    return history, model, scaler, X_test, y_test, df


# ============================================================
# Point d'entrée
# ============================================================

if __name__ == "__main__":
    train_rnn()
