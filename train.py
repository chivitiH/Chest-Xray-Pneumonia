# train_random_forest_final.py (corrigé)

# --- OPTIMISATION INTEL AJOUTÉE ---
from sklearnex import patch_sklearn
patch_sklearn()
# ------------------------------------

import os
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# --- CONFIGURATION ---
DATA_DIR = os.path.join("data", "raw", "chest_xray")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")
MODEL_SAVE_PATH = os.path.join("models", "chest_xray_final_rf_model.pkl")
IMAGE_SIZE = (224, 224)

def load_images_from_folder(folder, label):
    """Charge et prépare les images."""
    images = []
    labels = []
    print(f"Chargement des images du dossier : {folder}")
    for filename in os.listdir(folder):
        if filename.startswith('.'):
            continue

        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert('L').resize(IMAGE_SIZE)
                images.append(np.array(img).flatten())
                labels.append(label)
        except IOError:
            print(f"Erreur en lisant le fichier (ignoré) : {img_path}")
    return images, labels

def main():
    """Entraîne et sauvegarde le modèle RandomForest optimisé."""
    print("--- ENTRAÎNEMENT FINAL DU MODÈLE RANDOMFOREST (OPTIMISÉ PAR INTEL) ---")

    # 1. Chargement des données d'entraînement
    train_normal_images, train_normal_labels = load_images_from_folder(os.path.join(TRAIN_DATA_DIR, 'NORMAL'), 0)
    train_pneumonia_images, train_pneumonia_labels = load_images_from_folder(os.path.join(TRAIN_DATA_DIR, 'PNEUMONIA'), 1)
    
    X_train = np.array(train_normal_images + train_pneumonia_images)
    y_train = np.array(train_normal_labels + train_pneumonia_labels)
    
    # 2. Entraînement du modèle RandomForest
    print("\nEntraînement du RandomForest (hyperparamètres optimisés)...")
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=350,
        class_weight='balanced',
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"Entraînement terminé en {end_time - start_time:.2f} secondes.")

    # 3. Évaluation sur le set de TEST
    test_normal_images, test_normal_labels = load_images_from_folder(os.path.join(TEST_DATA_DIR, 'NORMAL'), 0)
    test_pneumonia_images, test_pneumonia_labels = load_images_from_folder(os.path.join(TEST_DATA_DIR, 'PNEUMONIA'), 1)
    
    # --- LIGNES CORRIGÉES ICI ---
    X_test = np.array(test_normal_images + test_pneumonia_images)
    y_test = np.array(test_normal_labels + test_pneumonia_labels)
    # ---------------------------

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nPrécision Globale (Accuracy) sur le set de test : {accuracy:.4f}")
    print("\nRapport de Classification Détaillé :")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia']))

    # 4. Sauvegarde
    print(f"Sauvegarde du modèle dans : {MODEL_SAVE_PATH}")
    joblib.dump(model, MODEL_SAVE_PATH)

if __name__ == '__main__':
    main()