# importation des bibiotheque
import tensorflow as tf
import numpy as np


# la taille des image CIFAR-10
IMG_SIZE = (32, 32, 3)
# nombre des classe a predire 
NUM_CLASSES = 10

# Taille d'un batch nombre  d'image traite en meme en temps

BATCH_SIZE = 64
# Nom lisibles dans des 10 classes cifar-10

CLASS_NAMES = ['avoin', 'automobile', 'oiseau',
               'chat', 'cerf', 'chien', 'grenouille',
               'cheval', 'bateau', 'camoin']

# couche d'augmentation de donne
# on cree un modele sequentiel keras contenant les couches
data_augmentation = tf.keras.Sequential([
   # on retourne notre image de facon aliatoire
   tf.keras.layers.RandomFlip("horizontal"),
    # faire pivoter l'image d'un angle aleatoire simule des angle de prise de vue differents
    tf.keras.layers.RandomRotation(0.1),

    # Zoom aleatoire leger
    tf.keras.layers.RandomZoom(0.1)
], name ="data_augmentation")

# Fonction loadata


def load_cifar10():
    # chargement brut des donnees depuis keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalisation des pixels
    x_train = x_train.astype("float32")/255.0
    x_test = x_test.astype("float32")/255.0
    # decoupage train/valiadion
    x_val = x_train[40000:]
    y_val = y_train[40000:]
    x_train = x_train[:40000]
    y_train = y_train[:40000]

    # affichages des verification des dimensions
    print(f"[data_loader] Train  :{x_train.shape} --- Labels :{y_train}")
    print(f"[data_loader] validation  :{x_val.shape} --- Labels :{y_val}")
    print(f"[data_loader] Test  :{x_test.shape} --- Labels :{y_test}")
    # Pipeline d'entrainement

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        # Melange aleatoire de 40 000 exemple a chaque epoque
        .shuffle(buffer_size=40000)
        # regroupement en batch_size image
        .batch(BATCH_SIZE)
        # Prechargement en parallele pendant que le GPU trait le batch N le cpu traite N-1
        .prefetch(tf.data.AUTOTUNE)
    )

    # Pipeline de validation
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        # pas shuffle sur laa validation resultat stables
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)

        # Pipelin de test
    )
    test_ds = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            # Pas de shufflet sur le test
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    print("== Test du data_loader ==")
    train_ds, val_ds, test_ds = load_cifar10()
    # Inspecter le premier batch pour vérifier les dimensions
    for images, labels in train_ds.take(1):
        print(f"Batch images : {images.shape}")   # (64, 32, 32, 3)
        print(f"Batch labels : {labels.shape}")   # (64, 1)
        print(f"Pixel min    : {images.numpy().min():.4f}")  # ~0.0
        print(f"Pixel max    : {images.numpy().max():.4f}")  # ~1.0
    print("=== Test réussi ! ===")
