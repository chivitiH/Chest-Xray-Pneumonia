# train_deep_learning.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
DATA_DIR = os.path.join("data", "raw", "chest_xray")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32 # On traite les images par lots de 32
MODEL_SAVE_PATH = os.path.join("models", "chest_xray_deep_learning_model.keras") # Nouveau format !

# 1. Chargement efficace des données avec Keras
print("Chargement des datasets...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'test'),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes trouvées :", class_names)

# 2. Augmentation des données pour améliorer la robustesse
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# 3. Préparation du modèle de base (Transfer Learning)
# On utilise MobileNetV2, un modèle léger et performant pré-entraîné sur des millions d'images
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False, # On ne garde pas sa dernière couche de classification
    weights='imagenet'
)

base_model.trainable = False # On "gèle" l'expert pour ne pas détruire son savoir

# 4. Construction de notre propre modèle par-dessus
model = Sequential([
  layers.Rescaling(1./255), # Normaliser les pixels entre 0 et 1
  data_augmentation,
  base_model,
  layers.GlobalAveragePooling2D(),
  layers.Dropout(0.2), # Une couche pour éviter le sur-apprentissage
  layers.Dense(1, activation='sigmoid') # Notre couche de décision finale (Pneumonia ou Normal)
])

# 5. Compilation du modèle
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# 6. Entraînement
print("\n--- DÉBUT DE L'ENTRAÎNEMENT DU MODÈLE DEEP LEARNING ---")
epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 7. Évaluation finale sur le set de test
print("\n--- ÉVALUATION SUR LE SET DE TEST ---")
loss, accuracy = model.evaluate(test_ds)
print(f"Précision (Accuracy) sur le set de test : {accuracy:.4f}")

# 8. Sauvegarde du modèle
print(f"Sauvegarde du modèle dans : {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)

print("\n--- FIN DU PROCESSUS ---")