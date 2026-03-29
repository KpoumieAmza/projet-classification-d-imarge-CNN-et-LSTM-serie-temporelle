import tensorflow as tf
# on importe les couches de des donnes
from tensorflow.keras import layers

# get_config() : sérialise la configuration du modèle
# Keras appelle cette méthode lors de la sauvegarde .keras
# Elle doit retourner TOUS les arguments du constructeur


def get_config(self):
    # On retourne un dictionnaire avec les paramètres
    # nécessaires pour reconstruire le modèle
    config = super().get_config()
    config.update({
        "num_classes": 10   # Seul argument de notre __init__
    })
    return config

    # ----------------------------------------------------------
    # from_config() : reconstruit le modèle depuis la config
    # Keras appelle cette méthode lors du chargement .keras
    # On filtre les arguments inconnus (trainable, dtype)
    # pour éviter l'erreur "unexpected keyword argument"
    # ----------------------------------------------------------
@classmethod
def from_config(cls, config):
    # On garde SEULEMENT num_classes et on ignore
    # les arguments Keras internes (trainable, dtype, etc.)
    return cls(num_classes=config.get("num_classes", 10))


# classe customCNN


class CustomCNN(tf.keras.Model):
    def __init__(self, num_classes=10):

        # Appel obligatoire au constructeur parent
        super(CustomCNN, self).__init__()
        # Augmentation des donnes 
        # ces couhes ne s'active que pendant l'entrainement
        self.augment = tf.keras.Sequential([
            # retournement horizontal aleatoire
            layers.RandomFlip("horizontal"),
            # rotation aleatoire leger
            layers.RandomRotation(0.1),
            # zoom aleatoire le leger
            layers.RandomZoom(0.1)  
        ], name="augmentation")
        # Premiere bloc convolutif
        # Couche conv2D
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', name="conv1")
        # BatchNormalization : normaliser les activation apres conv
        # Accelere l'entrainement et stabilise le gradient
        self.bn1 = layers.BatchNormalization(name="bn1")
        # deux convolutions consecutive = plus de puissance d'extration
        self.conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", name="conv2")
        self.bn2 = layers.BatchNormalization(name="bn2")

        # MaxPooling2D : réduit la taille spatiale de moitié (32→16)
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")

        # Dropout: desactive alearatoirement des neurones
        self.drop1 = layers.Dropout(rate=0.25, name='drop1')
        # deuxieme bloc convolutif
        # But detecter des feacture de niveau intermediaire on double le nombre

        self.conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name="conv3")
        self.bn3 = layers.BatchNormalization(name="bn3")

        self.conv4 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name="conv4")
        self.bn4 = layers.BatchNormalization(name="bn4")
        # Maxpooling reduit encore de moitie
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")
        self.drop2 = layers.Dropout(rate=0.25, name="pool2")
        # BLOC 3 : Troisième bloc convolutif  détecter des features haut niveau (objets entiers)
     
        self.conv5 = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name="conv5"
        )
        self.bn5 = layers.BatchNormalization(name="bn5")

        self.conv6 = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name="conv6"
        )
        self.bn6 = layers.BatchNormalization(name="bn6")
        # MaxPooling : réduit encore de moitié (8→4)
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")

        self.drop3 = layers.Dropout(rate=0.25, name="drop3")

        # CLASSIFICATEUR FINAL MLP — Multi-Layer Perceptron
        self.flatten = layers.Flatten(name="flatten")
        # Première couche Dense
        self.dense1 = layers.Dense(units=512, activation="relu", name="dense1")
        self.bn7 = layers.BatchNormalization(name="bn7")
        # Dropout plus fort (50%) sur le classificateur
        self.drop4 = layers.Dropout(rate=0.5, name="drop4")
        # deuxieme couche dense
        self.dense2 = layers.Dense(units=256, activation="relu", name="dense2")
        self.drop5 = layers.Dropout(rate=0.5, name="drop5")
        #
        # couche de sortie
        # Pas d'activation ici — la fonction de perte
        self.output_layer = layers.Dense(units=num_classes, activation=None, name="output")
        # Méthode call() : définit le forward pass

    def call(self, inputs, training=False):
        # Augmentation (active seulement en entraînement)
        x = self.augment(inputs, training=training)
        # bloc convolutif 1
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool1(x)
        x = self.drop1(x, training=training)
        # bloc convolution 2
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        # bloc de convolution 3
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.conv6(x)
        x = self.bn6(x, training=training)
        x = self.pool3(x)
        x = self.drop3(x, training=training)
        # Classificateur MLP
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn7(x, training=training)
        x = self.drop4(x, training=training)
        x = self.dense2(x)
        x = self.drop5(x, training=training)
        # sortie finale logits bruts
        x = self.output_layer(x)
        return x
            
            
if __name__ == "__main__":
    print("=== Test de l'architecture CustomCNN ===")

    # Instancier le modèle
    model = CustomCNN(num_classes=10)

    # Construire le modèle avec un faux batch pour initialiser
    # les poids (obligatoire avant model.summary())
    dummy_input = tf.zeros((1, 32, 32, 3))  # 1 image RGB 32×32
    model(dummy_input, training=False)

    # Afficher le résumé complet (couches + paramètres)
    model.summary()
    print("=== Architecture OK ! ===")

