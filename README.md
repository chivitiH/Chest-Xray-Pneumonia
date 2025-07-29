🩺 Détection de Pneumonie par IA - Analyse Comparative
Démo en direct de l'application : https://chest-xray-pneumonia-57xyx2a2khe6x6hsunxa4k.streamlit.app/

🎯 Objectif du Projet
Ce projet vise à développer et comparer plusieurs approches de Machine Learning et de Deep Learning pour la détection automatique de pneumonie à partir de radiographies thoraciques. L'objectif final est de construire une application web interactive permettant de visualiser et de comparer les diagnostics de ces modèles.

🔬 La Démarche : De la Baseline au Modèle de Pointe
Ce projet a suivi une démarche itérative complète, allant d'un modèle de base solide à des techniques d'intelligence artificielle de pointe pour maximiser la précision.

1. Approche par Machine Learning Classique
La première étape a consisté à établir une baseline robuste en utilisant des algorithmes classiques.

Modèle Testé : RandomForest.

Optimisations : Utilisation de class_weight pour gérer le déséquilibre des classes, optimisation des hyperparamètres et accélération matérielle avec scikit-learn-intelex.

Résultat Obtenu : 85.26% de précision sur le jeu de test.

Conclusion : Une solution extrêmement rapide et efficace, constituant une excellente baseline.

2. Approche par Deep Learning (Transfer Learning)
Pour dépasser les limites du Machine Learning classique, une approche par Transfer Learning a été mise en œuvre.

Architectures Testées : MobileNetV2, EfficientNetV2B3, InceptionResNetV2.

Techniques : Fine-tuning, augmentation de données, callbacks intelligents (EarlyStopping, ReduceLROnPlateau), et gestion des poids de classe.

Résultat Obtenu (Modèle Unique) : ~87.5% de précision.

3. Approche par Ensemble de Modèles (État de l'Art) ✅
Pour atteindre la performance maximale, les deux meilleurs modèles de Deep Learning ont été combinés en un "comité d'experts" (Ensembling).

Composition : EfficientNetV2B3 + InceptionResNetV2.

Méthode : La prédiction finale est la moyenne des scores de confiance des deux modèles.

Résultat Obtenu : 92.47% de précision sur le jeu de test.

Conclusion : Cette approche, bien que plus lourde en calcul, offre une précision et une robustesse de niveau professionnel, la qualifiant comme la solution de pointe pour ce problème.

🏆 Solution Finale
Le projet se conclut sur une application Streamlit comparative qui déploie les deux meilleures approches :

Le Champion du Deep Learning : Un ensemble de modèles (EfficientNetV2B3 + InceptionResNetV2) atteignant 92.47% de précision.

Le Champion du ML Classique : Un RandomForest optimisé atteignant 85.26% de précision en un temps record.

🚀 Déploiement & Utilisation
L'application est déployée sur Streamlit Cloud et accessible publiquement.

Lancer le projet en local
Clonez le dépôt : git clone https://github.com/chivitiH/Chest-Xray-Pneumonia.git

Installez les dépendances : pip install -r requirements.txt

Lancez l'application : streamlit run app.py

💼 Compétences Démontrées
Maîtrise Complète du Cycle MLOps : De l'analyse de données à l'entraînement, la gestion des dépendances (requirements.txt), la gestion de version (Git, Git-LFS), jusqu'au déploiement continu sur une plateforme cloud (Streamlit Cloud).

Expertise en Deep Learning : Implémentation de techniques avancées comme le Transfer Learning, le Fine-Tuning, la Data Augmentation et l'Ensembling de modèles.

Analyse Comparative Rigoureuse : Évaluation et comparaison de multiples approches (ML vs DL) pour identifier la meilleure solution.

Problem Solving & Débogage : Résolution de problèmes complexes de compatibilité de versions, de dépendances et de déploiement multi-plateformes (Local/Cloud).

🎯 Apprentissages Clés
L'Ensembling est une technique de pointe pour maximiser la robustesse et la précision.

Une baseline solide en ML classique est essentielle pour mesurer les gains réels du Deep Learning.

La gestion d'environnement (requirements.txt, versions Python) est la clé d'un déploiement réussi.