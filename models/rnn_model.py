# ============================================================
# models/rnn_model.py
# Rôle : Définir l'architecture du réseau LSTM pour la
#        prédiction du prix High de Tesla (TSLA)
# API : tf.keras.Model (Subclassing — standard industrie)
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# Classe StockLSTM
# Hérite de tf.keras.Model (API Subclassing)
# ============================================================

class StockLSTM(tf.keras.Model):
    """
    Réseau LSTM pour la prédiction de série temporelle boursière.

    Architecture :
        Entrée (batch, 60, 1)
        → LSTM(128, return_sequences=True)
        → Dropout(0.3)
        → LSTM(64, return_sequences=False)  ← dernier état caché uniquement
        → Dropout(0.3)
        → Dense(32, relu)
        → Dense(1)  ← prédiction du prix normalisé J+1
    """

    def __init__(self, units_1=128, units_2=64, dropout_rate=0.3):
        """
        Constructeur : déclaration de toutes les couches.

        Args:
            units_1      : neurones de la 1ère couche LSTM (128)
            units_2      : neurones de la 2ème couche LSTM (64)
            dropout_rate : taux de dropout entre les couches (0.3)
        """

        # Appel obligatoire au constructeur parent
        super(StockLSTM, self).__init__()

        # Sauvegarde des hyperparamètres pour get_config()
        self.units_1      = units_1
        self.units_2      = units_2
        self.dropout_rate = dropout_rate

        # ----------------------------------------------------------
        # COUCHE LSTM 1 : extraction des patterns temporels longs
        # units=128 : 128 cellules mémoire (états cachés)
        # return_sequences=True : retourne la séquence complète
        #   → sortie : (batch, 60, 128)
        #   → nécessaire pour alimenter la 2ème couche LSTM
        # ----------------------------------------------------------
        self.lstm1 = layers.LSTM(
            units=units_1,
            return_sequences=True,   # Garde toute la séquence temporelle
            name="lstm1"
        )

        # Dropout après LSTM1
        # Désactive aléatoirement 30% des connexions
        # pendant l'entraînement pour éviter l'overfitting
        self.drop1 = layers.Dropout(rate=dropout_rate, name="drop1")

        # ----------------------------------------------------------
        # COUCHE LSTM 2 : synthèse finale de la séquence
        # units=64 : 64 cellules mémoire
        # return_sequences=False : retourne UNIQUEMENT le dernier
        #   état caché (comme demandé dans l'énoncé)
        #   → sortie : (batch, 64)
        #   → vecteur résumant toute la séquence de 60 jours
        # ----------------------------------------------------------
        self.lstm2 = layers.LSTM(
            units=units_2,
            return_sequences=False,  # Seulement le dernier état caché
            name="lstm2"
        )

        # Dropout après LSTM2
        self.drop2 = layers.Dropout(rate=dropout_rate, name="drop2")

        # ----------------------------------------------------------
        # COUCHE DENSE INTERMÉDIAIRE
        # Apprend des combinaisons non-linéaires de l'état LSTM
        # relu : introduit la non-linéarité
        # ----------------------------------------------------------
        self.dense1 = layers.Dense(
            units=32,
            activation='relu',
            name="dense1"
        )

        # Dropout léger avant la sortie
        self.drop3 = layers.Dropout(rate=0.1, name="drop3")

        # ----------------------------------------------------------
        # COUCHE DE SORTIE
        # units=1 : prédit une seule valeur (prix High J+1 normalisé)
        # Pas d'activation : régression → valeur continue entre 0 et 1
        # ----------------------------------------------------------
        self.output_layer = layers.Dense(
            units=1,
            activation=None,   # Pas d'activation pour la régression
            name="output"
        )

    # ----------------------------------------------------------
    # Méthode call() : forward pass
    # Décrit comment les données traversent le réseau
    #
    # Args:
    #   inputs   : (batch, 60, 1) — fenêtres temporelles
    #   training : True pendant fit(), False pendant predict()
    # ----------------------------------------------------------

    def call(self, inputs, training=False):

        # LSTM 1 : extrait les patterns sur toute la séquence
        x = self.lstm1(inputs)                    # → (batch, 60, 128)
        x = self.drop1(x, training=training)

        # LSTM 2 : synthétise en un vecteur unique
        x = self.lstm2(x)                         # → (batch, 64)
        x = self.drop2(x, training=training)

        # Dense intermédiaire
        x = self.dense1(x)                        # → (batch, 32)
        x = self.drop3(x, training=training)

        # Sortie : prédiction scalaire normalisée
        x = self.output_layer(x)                  # → (batch, 1)

        return x

    # ----------------------------------------------------------
    # get_config() et from_config() : sérialisation du modèle
    # Nécessaires pour sauvegarder/charger avec .keras
    # ----------------------------------------------------------

    def get_config(self):
        config = super().get_config()
        config.update({
            "units_1":      self.units_1,
            "units_2":      self.units_2,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Filtre les arguments Keras internes (trainable, dtype)
        return cls(
            units_1=config.get("units_1",      128),
            units_2=config.get("units_2",      64),
            dropout_rate=config.get("dropout_rate", 0.3),
        )


# ============================================================
# Test rapide
# Commande : python models/rnn_model.py
# ============================================================

if __name__ == "__main__":
    print("=== Test architecture StockLSTM ===")

    model = StockLSTM(units_1=128, units_2=64, dropout_rate=0.3)

    # Warm-up : (1 batch, 60 jours, 1 feature)
    dummy = tf.zeros((1, 60, 1))
    out = model(dummy, training=False)

    print(f"Entrée  : {dummy.shape}")   # (1, 60, 1)
    print(f"Sortie  : {out.shape}")     # (1, 1)

    model.summary()
    print("=== Architecture OK ! ===")
