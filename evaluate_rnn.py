# ============================================================
# evaluate_rnn.py
# Rôle : Charger le modèle LSTM sauvegardé, générer les
#        prédictions et tracer la courbe réelle vs prédite
#        avec inverse_transform pour afficher les vrais prix
# Commande : python evaluate_rnn.py
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.rnn_model import StockLSTM
from utils.data_loader_rnn import load_stock_data, SEQUENCE_LENGTH

MODEL_SAVE_PATH = "models/tsla_lstm.keras"


# ============================================================
# Fonction : plot_predictions()
# Rôle : Courbe prix réels vs prix prédits (en dollars)
# ============================================================

def plot_predictions(y_real, y_pred, title="TSLA — Prix High réel vs prédit"):
    """
    Génère le graphique superposant :
    - Courbe bleue : prix réels (données de test)
    - Courbe orange : prédictions du LSTM
    Les deux courbes sont en dollars réels (après inverse_transform)

    Args:
        y_real : tableau numpy des prix réels en $
        y_pred : tableau numpy des prix prédits en $
        title  : titre du graphique
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')

    days = np.arange(len(y_real))

    # ----------------------------------------------------------
    # Graphique 1 : Courbe complète réel vs prédit
    # ----------------------------------------------------------
    ax1.set_facecolor('#12122a')

    # Zone ombrée entre les deux courbes (écart)
    ax1.fill_between(days, y_real, y_pred,
                     alpha=0.15, color='#7c3aed',
                     label='Écart prédiction')

    # Courbe prix réels
    ax1.plot(days, y_real,
             label='Prix réel (High)',
             color='#00d4ff',
             linewidth=2,
             zorder=3)

    # Courbe prédictions
    ax1.plot(days, y_pred,
             label='Prédiction LSTM',
             color='#f59e0b',
             linewidth=1.8,
             linestyle='--',
             zorder=4)

    ax1.set_title(title, color='white', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Jours de trading (jeu de test)', color='white')
    ax1.set_ylabel('Prix High ($)', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1a1a35', labelcolor='white', fontsize=10)
    ax1.grid(True, alpha=0.12, color='#94a3b8')
    for spine in ax1.spines.values():
        spine.set_color('#1e2d4a')

    # ----------------------------------------------------------
    # Graphique 2 : Erreur absolue par jour
    # ----------------------------------------------------------
    ax2.set_facecolor('#12122a')

    errors = np.abs(y_real - y_pred)

    # Barres d'erreur colorées selon l'amplitude
    colors_bar = ['#10b981' if e < 5 else '#f59e0b' if e < 15 else '#ef4444'
                  for e in errors]

    ax2.bar(days, errors, color=colors_bar, alpha=0.8, width=1.0)

    # Ligne de l'erreur moyenne
    mae = np.mean(errors)
    ax2.axhline(y=mae, color='#00d4ff', linewidth=1.5,
                linestyle='--', label=f'MAE moyenne : ${mae:.2f}')

    ax2.set_title('Erreur absolue par jour (en $)',
                  color='white', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Jours de trading', color='white')
    ax2.set_ylabel('Erreur absolue ($)', color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1a1a35', labelcolor='white', fontsize=10)
    ax2.grid(True, alpha=0.12, color='#94a3b8')
    for spine in ax2.spines.values():
        spine.set_color('#1e2d4a')

    # Légende couleurs barres
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#10b981', label='Erreur < $5'),
        Patch(facecolor='#f59e0b', label='Erreur $5-$15'),
        Patch(facecolor='#ef4444', label='Erreur > $15'),
    ]
    ax2.legend(handles=legend_elements, facecolor='#1a1a35',
               labelcolor='white', fontsize=9, loc='upper right')

    plt.tight_layout()

    save_path = "data/rnn_predictions.png"
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"[evaluate_rnn] Graphique sauvegardé → {save_path}")


# ============================================================
# Fonction principale : evaluate_rnn()
# ============================================================

def evaluate_rnn():

    print("=" * 55)
    print("   ÉVALUATION — StockLSTM TSLA High")
    print("=" * 55)

    # ----------------------------------------------------------
    # ÉTAPE 1 : Chargement des données
    # ----------------------------------------------------------
    print("\n[1/4] Chargement des données...")
    train_ds, val_ds, test_ds, scaler, df, X_test, y_test = load_stock_data()
    print(f"      → {len(X_test)} séquences de test")

    # ----------------------------------------------------------
    # ÉTAPE 2 : Reconstruction + chargement des poids
    # Même stratégie que pour le CNN : on recrée puis load_weights
    # ----------------------------------------------------------
    print(f"\n[2/4] Chargement du modèle depuis {MODEL_SAVE_PATH}...")

    model = StockLSTM(units_1=128, units_2=64, dropout_rate=0.3)

    # Warm-up obligatoire avant load_weights
    dummy = tf.zeros((1, SEQUENCE_LENGTH, 1))
    model(dummy, training=False)

    # Chargement des poids uniquement
    model.load_weights(MODEL_SAVE_PATH)

    # Recompilation pour model.evaluate()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanSquaredError(name='mse'),
            tf.keras.metrics.MeanAbsoluteError(name='mae')
        ]
    )
    print("      → Poids chargés avec succès !")

    # ----------------------------------------------------------
    # ÉTAPE 3 : Évaluation sur le jeu de test
    # ----------------------------------------------------------
    print("\n[3/4] Évaluation sur le jeu de test...")

    results = model.evaluate(test_ds, verbose=0)
    test_mse = results[1]
    test_mae = results[2]

    # ----------------------------------------------------------
    # ÉTAPE 4 : Prédictions + inverse_transform
    # On reconvertit les valeurs normalisées [0,1]
    # en prix réels en dollars
    # ----------------------------------------------------------
    print("\n[4/4] Génération des prédictions...")

    # Prédictions normalisées : shape (n_test, 1)
    y_pred_scaled = model.predict(test_ds, verbose=0)

    # Aplatissement : (n_test, 1) → (n_test,)
    y_pred_scaled = y_pred_scaled.flatten()

    # inverse_transform : [0,1] → prix en dollars réels
    # reshape(-1,1) car sklearn attend une matrice 2D
    y_pred_real = scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()

    y_real_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # ----------------------------------------------------------
    # Métriques en prix réels (dollars)
    # ----------------------------------------------------------
    mae_real  = mean_absolute_error(y_real_real, y_pred_real)
    mse_real  = mean_squared_error(y_real_real,  y_pred_real)
    rmse_real = np.sqrt(mse_real)

    # MAPE : Mean Absolute Percentage Error
    mape = np.mean(
        np.abs((y_real_real - y_pred_real) / y_real_real)
    ) * 100

    print(f"\n   ┌──────────────────────────────────────┐")
    print(f"   │ Métriques sur données normalisées    │")
    print(f"   │   Test MSE  : {test_mse:.6f}               │")
    print(f"   │   Test MAE  : {test_mae:.6f}               │")
    print(f"   ├──────────────────────────────────────┤")
    print(f"   │ Métriques en dollars réels ($)       │")
    print(f"   │   MAE  : ${mae_real:.2f} / jour              │")
    print(f"   │   RMSE : ${rmse_real:.2f} / jour              │")
    print(f"   │   MAPE : {mape:.2f}%                        │")
    print(f"   └──────────────────────────────────────┘")

    # Exemples de prédictions vs réalité
    print(f"\n--- Exemples de prédictions (10 premiers jours de test) ---")
    print(f"{'Jour':<6} {'Réel ($)':>10} {'Prédit ($)':>12} {'Écart ($)':>10}")
    print("-" * 42)
    for i in range(min(10, len(y_real_real))):
        ecart = y_pred_real[i] - y_real_real[i]
        print(f"{i+1:<6} {y_real_real[i]:>10.2f} {y_pred_real[i]:>12.2f} {ecart:>+10.2f}")

    # Génération du graphique réel vs prédit
    plot_predictions(y_real_real, y_pred_real)

    print("\n=== Évaluation terminée ! ===")
    print("=== Fichiers sauvegardés dans data/ ===")
    print("    - data/rnn_predictions.png")
    print("    - data/rnn_training_curves.png")

    return y_real_real, y_pred_real


# ============================================================
# Point d'entrée
# ============================================================

if __name__ == "__main__":
    evaluate_rnn()
