import os
import tensorflow as tf
import matplotlib.pyplot as plt
# Importation de notre model personnel
from models.cnn_model import CustomCNN 
# importation de notre pipelen de donne 
from utils.data_loader import load_cifar10, BATCH_SIZE

# hyperparametre d'entrainement centraliser ici pour faciliter l'entrainement
LEARNING_RATE = 0.001     # vitesse d'aprentissage de l'optimiseur Adam
EPOCHS = 50     # nombre max d'epochs
MODEL_SAVE_PATH = "models/cifar10_cnn.keras"

# Fonction : plot_history()
# Rôle : Tracer et sauvegarder les courbes Train vs Validation
#        Loss et Accuracy après l'entraînement


def plot_history(history):
    # Création d'une figure avec 2 sous-graphiques côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("courbe d'entrainement - CustomCNN CIFAR-10", fontsize=14, fontweight='bold')
    
    # Graphique 1 : Courbes de perte (Loss)
    ax1.plot(history.history['val_loss'], label="validation Loss",
        color='#2196f3', linewidth=2)
    # perte sur validation 
    ax1.plot(
        history.history['val_loss'],
        label='Validation Loss',
        color='#F44336',           
        linestyle='--'
    )
    ax1.set_title("Perte (Loss)")
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Valeur de la perte")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Graphique 2 : Courbes de précision (Accuracy)
    ax2.plot(
        history.history['accuracy'],      # Précision sur train
        label='Train Accuracy',
        linewidth=2
    )
    ax2.plot(
        history.history['val_accuracy'],  # Précision sur validation
        label='Validation Accuracy',
        color='#FF9800',                  
        linewidth=2,
        linestyle='--'
    )
    ax2.set_title("Précision (Accuracy)")
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Précision (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Ajustement automatique des espacements entre les sous-graphiques
    plt.tight_layout()
  
    # Sauvegarde de la figure dans le dossier data/
    save_path = "data/training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[train] Courbes sauvegardées → {save_path}")
    plt.show()
  
# Fonction principale : train()


def train():

    print("=" * 55)
    print("   ENTRAÎNEMENT — CustomCNN sur CIFAR-10")
    print("=" * 55)
  
    # ÉTAPE 1 : Chargement des données
    # On appelle notre pipeline tf.data.Dataset
    print("\n[1/5] chargement des donnees CIFAR-10...")
    train_ds, val_ds, test_ds = load_cifar10()
    print("  -> donnees chargees avec succes")

    # ÉTAPE 2 : Instanciation du modèle
    print("\n[2/5] creation du modele CustomCNN..")

    model = CustomCNN(num_classes=10)
    # (nécessaire avant compile() avec l'API Subclassing)
    dummy = tf.zeros((1, 32, 32, 3))
    model(dummy, training=False)
    print(f"    → Modèle créé : {model.count_params():,} paramètres")
    # ÉTAPE 3 : Compilation du modèle
    print("\n[3/5] Compilation du modèle...")
    # Optimiseur Adam avec learning rate personnalisé ajuste automatiquement le learning rate
    #  pour chaque paramètre individuellement
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        # Fonction de perte : mesure l'erreur du modèle
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # Métriques à surveiller pendant l'entraînement
        metrics=['accuracy']
    )
    print("   → Compilation OK (Adam + SparseCategoricalCrossentropy)")
    # ÉTAPE 4 : Définition des Callbacks
    print("\n[4/5] configuration des callbacks")
    #  Callback 1 : EarlyStopping  Arrête l'entraînement si val_loss ne s'améliore plus
    # pendant 'patience' époques consécutives
    # Évite le surapprentissage et le gaspillage de temps
    early_stopping = tf.keras.callbacks.EarlyStopping(
        # Surveille la perte de validation
        monitor='val_loss',
        # Attend 8 époques sans amélioration  
        patience=8,
        # Restaure les meilleurs poids         
        restore_best_weights=True, 
        # Affiche un message à l'arrêt
        verbose=1              
    )
    # --- Callback 2 : ModelCheckpoint ---
    # Sauvegarde automatiquement le meilleur modèle
    # à chaque fois que val_accuracy s'améliore
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,   # Chemin de sauvegarde
        monitor='val_accuracy',     # Surveille la précision de validation
        save_best_only=True,        # Sauvegarde SEULEMENT si amélioration
        verbose=1                   # Affiche un message à chaque sauvegarde
    )
    
    # --- Callback 3 : ReduceLROnPlateau ---
    # Réduit le learning rate si val_loss stagne
    # Permet au modèle de "affiner" sa convergence
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',     # Surveille la perte de validation
        factor=0.5,             # Divise le LR par 2 si stagnation
        patience=4,             # Attend 4 époques avant de réduire
        min_lr=1e-6,            # Learning rate minimum autorisé
        verbose=1               # Affiche un message à chaque réduction
    )

    print("  → EarlyStopping + ModelCheckpoint + ReduceLROnPlateau")
    # ÉTAPE 5 : Entraînement avec model.fit()

    print("\n[5/5] Lancement de l'entraînement...\n")

    history = model.fit(
        train_ds,                   # Pipeline d'entraînement
        epochs=EPOCHS,              # Nombre max d'époques
        validation_data=val_ds,     # Pipeline de validation
        callbacks=[                 # Liste des callbacks actifs
            early_stopping,
            checkpoint,
            reduce_lr
        ],
        verbose=1                   # Affiche la progression époque par époque
    )
    # Résultats finaux
    
    print("\n" + "=" * 55)
    print("   ENTRAÎNEMENT TERMINÉ")
    print("=" * 55)

    # Récupération de la meilleure précision de validation atteinte
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n   Meilleure val_accuracy : {best_val_acc * 100:.2f}%")
    print(f"   Modèle sauvegardé     : {MODEL_SAVE_PATH}")

    # Vérification de l'objectif du projet (70% minimum)
    if best_val_acc >= 0.70:
        print("   ✓ Objectif 70% atteint !")
    else:
        print("   ✗ Objectif 70% non atteint — essaie plus d'époques")

    # Génération des courbes d'entraînement
    plot_history(history)
    return history


if __name__ == "__main__":
    train()
