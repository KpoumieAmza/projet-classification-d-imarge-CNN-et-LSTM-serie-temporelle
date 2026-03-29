# ============================================================
# evaluate.py
# Version corrigée pour Windows — matplotlib backend forcé
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')          # Force la sauvegarde fichier
# sans ouvrir de fenêtre GUI
# (évite les blocages sur Windows)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from models.cnn_model import CustomCNN
from utils.data_loader import CLASS_NAMES

MODEL_SAVE_PATH = "models/cifar10_cnn.keras"

# Fonction : plot_confusion_matrix()


def plot_confusion_matrix(cm, class_names):

    # Normalisation : comptages → pourcentages par ligne
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    im = ax.imshow(cm_normalized, interpolation='nearest',
                   cmap='Blues', vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Taux de prédiction', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.set_title('Matrice de Confusion — CustomCNN CIFAR-10',
                 color='white', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Classe prédite',  color='white', fontsize=12)
    ax.set_ylabel('Classe réelle',   color='white', fontsize=12)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45,
                       ha='right', color='white', fontsize=10)
    ax.set_yticklabels(class_names, color='white', fontsize=10)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            color = 'white' if pct < 0.5 else '#1a1a2e'
            ax.text(j, i - 0.15, f"{pct:.0%}",
                    ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')
            ax.text(j, i + 0.2, f"({count})",
                    ha='center', va='center',
                    color=color, fontsize=7, alpha=0.8)

    plt.tight_layout()

    # Sauvegarde directe sans fenêtre popup
    save_path = "data/confusion_matrix.png"
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()   # Libère la mémoire
    print(f"[evaluate] Matrice sauvegardée → {save_path}")

# Fonction : plot_sample_predictions()


def plot_sample_predictions(model, x_test, y_test, class_names):

    # Sélection de 20 images aléatoires
    np.random.seed(42)   # Graine fixe pour la reproductibilité
    indices = np.random.choice(len(x_test), 20, replace=False)
    images = x_test[indices]
    labels = y_test[indices]

    # Prédiction sur les 20 images
    logits = model(images, training=False)
    predictions = tf.argmax(tf.nn.softmax(logits), axis=1).numpy()

    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle("Prédictions du modèle sur 20 images de test",
                 color='white', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], interpolation='bilinear')
        ax.axis('off')
        true_label = class_names[int(labels[i])]
        pred_label = class_names[int(predictions[i])]
        color = '#4CAF50' if true_label == pred_label else '#F44336'
        symbol = 'OK' if true_label == pred_label else 'X'
        ax.set_title(
            f"{symbol} {pred_label}\n(reel: {true_label})",
            color=color, fontsize=8, fontweight='bold'
        )

    plt.tight_layout()

    save_path = "data/sample_predictions.png"
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"[evaluate] Exemples sauvegardes  → {save_path}")


# Fonction principale : evaluate()


def evaluate():

    print("=" * 55)
    print("   EVALUATION — CustomCNN sur CIFAR-10")
    print("=" * 55)

    # ETAPE 1 : Chargement des données de test

    print("\n[1/4] Chargement des donnees...")
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.flatten()
    print(f"  -> {len(x_test)} images de test chargees")

    # ETAPE 2 : Reconstruction du modèle + chargement des poids
    # On recrée le modèle depuis zéro puis on charge les poids

    print(f"\n[2/4] Reconstruction du modele...")

    model = CustomCNN(num_classes=10)

    # Warm-up pour initialiser les poids
    dummy = tf.zeros((1, 32, 32, 3))
    model(dummy, training=False)

    # Chargement des poids uniquement
    model.load_weights(MODEL_SAVE_PATH)
    print("      -> Poids charges avec succes !")

    # Recompilation pour model.evaluate()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # ETAPE 3 : Evaluation globale sur le jeu de test

    print("\n[3/4] Evaluation sur le jeu de test...")

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"   Test Loss  : {test_loss:.4f}")

    # ETAPE 4 : Matrice de confusion + rapport par classe
  
    print("\n[4/4] Generation de la matrice de confusion...")

    logits_all = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(logits_all, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Rapport par classe ---")
    print(classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=3
    ))

    # Génération et sauvegarde des graphiques
    plot_confusion_matrix(cm, CLASS_NAMES)
    plot_sample_predictions(model, x_test, y_test, CLASS_NAMES)

    print("\n=== Evaluation terminee ! ===")
    print("=== Fichiers sauvegardes dans data/ ===")
    print("    - data/confusion_matrix.png")
    print("    - data/sample_predictions.png")

    return test_acc

# Point d'entree


if __name__ == "__main__":
    evaluate()
