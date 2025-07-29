import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Tableau de Bord d'Analyse de Pneumonie",
    page_icon="🩺",
    layout="wide"
)

# --- FONCTIONS DE CHARGEMENT DES MODÈLES ---
@st.cache_resource
def load_all_models():
    """Charge tous les modèles une seule fois au démarrage."""
    print("Chargement des modèles...")
    # Modèle ML Classique
    rf_path = os.path.join("models", "chest_xray_final_rf_model.pkl")
    rf_model = joblib.load(rf_path)
    
    # Modèles Deep Learning
    dl1_path = os.path.join("models", "EfficientNetV2B3_model.keras")
    dl2_path = os.path.join("models", "InceptionResNetV2_model.keras")
    dl_model1 = tf.keras.models.load_model(dl1_path)
    dl_model2 = tf.keras.models.load_model(dl2_path)
    print("Modèles chargés.")
    
    return rf_model, dl_model1, dl_model2

# Chargement des modèles
rf_model, dl_model1, dl_model2 = load_all_models()

# --- INTERFACE UTILISATEUR ---
st.title("🩺 Tableau de Bord d'Analyse de Radiographies Thoraciques")
st.markdown("Comparez un modèle de Machine Learning classique optimisé avec un ensemble de modèles de Deep Learning de pointe.")

# --- SÉLECTION DU MODÈLE ---
model_choice = st.selectbox(
    "Choisissez une approche pour le diagnostic :",
    (
        "Deep Learning - Ensemble (Précision : 92.47%)",
        "Machine Learning - RandomForest (Précision : 85.26%)"
    )
)
st.write("---")

col1, col2 = st.columns([2, 3])

with col1:
    # --- UPLOAD DE L'IMAGE ---
    uploaded_file = st.file_uploader(
        "Chargez une radiographie thoracique...", 
        type=["jpeg", "jpg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée.", use_container_width=True)

with col2:
    if uploaded_file:
        # --- PRÉDICTION ET AFFICHAGE ---
        if st.button(f"Lancer le diagnostic avec l'approche '{model_choice.split(' - ')[0]}'"):
            with st.spinner('Analyse en cours...'):
                
                # --- Logique pour le Deep Learning Ensemble ---
                if "Deep Learning" in model_choice:
                    st.subheader("Diagnostic par l'Ensemble de Modèles Deep Learning")
                    
                    # Prétraitement de l'image
                    img_resized = image.resize((224, 224))
                    img_array = np.array(img_resized)
                    if img_array.ndim == 2:
                        img_array = np.stack((img_array,)*3, axis=-1)
                    img_batch = np.expand_dims(img_array, axis=0)

                    # Prédiction avec chaque modèle DL
                    pred_score1 = dl_model1.predict(img_batch)[0][0]
                    pred_score2 = dl_model2.predict(img_batch)[0][0]
                    
                    final_score = (pred_score1 + pred_score2) / 2.0
                    label = "PNEUMONIA" if final_score > 0.5 else "NORMAL"
                    confidence = final_score if label == "PNEUMONIA" else 1 - final_score

                # --- Logique pour le Machine Learning Classique ---
                else:
                    st.subheader("Diagnostic par le Modèle RandomForest")
                    
                    # Prétraitement de l'image
                    img_rf = image.convert('L').resize((224, 224))
                    img_flat = np.array(img_rf).flatten().reshape(1, -1)
                    
                    pred_idx = rf_model.predict(img_flat)[0]
                    label = "PNEUMONIA" if pred_idx == 1 else "NORMAL"
                    proba = rf_model.predict_proba(img_flat)[0]
                    confidence = proba.max()

            # Affichage des résultats
            if label == 'PNEUMONIA':
                st.error(f"**Diagnostic final : {label}**")
            else:
                st.success(f"**Diagnostic final : {label}**")
            
            st.metric(label="Confiance du diagnostic", value=f"{confidence:.2%}")
            st.progress(float(confidence))

            if "Deep Learning" in model_choice:
                st.write("**Avis des experts individuels :**")
                st.write(f"- *EfficientNet* pensait à 'Pneumonie' avec une probabilité de {pred_score1:.1%}")
                st.write(f"- *InceptionResNet* pensait à 'Pneumonie' avec une probabilité de {pred_score2:.1%}")
    else:
        st.info("Veuillez charger une image pour activer le diagnostic.")