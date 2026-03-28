# 🚀 Pipeline Big Data dans le cloud — AWS EMR

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-3.x-orange?logo=apachespark)
![AWS EMR](https://img.shields.io/badge/AWS-EMR-yellow?logo=amazonaws)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-orange?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![RGPD](https://img.shields.io/badge/RGPD-Compliant-blue)

---

## 🎯 Objectif

Concevoir et déployer un **pipeline Big Data distribué en production** pour l'extraction de features visuelles à grande échelle, en vue d'un futur moteur de classification de fruits.

Le pipeline traite **22 688 images** sur un cluster AWS EMR, extrait des features via **MobileNetV2**, réduit la dimensionnalité par **PCA**, et stocke les résultats en **Parquet sur S3**.

---

## 📊 Résultats

| Indicateur | Valeur |
|-----------|--------|
| Images traitées | 22 688 |
| Features extraites (MobileNetV2) | 1 280 par image |
| Composantes PCA retenues | 50 |
| Variance expliquée | 83,7 % |
| Fichiers Parquet produits | 24 + 1 `_SUCCESS` |
| Durée d'exécution (EMR) | ~19 min |
| Région AWS | eu-west-3 (Paris) — RGPD ✅ |

---

## 🏗️ Architecture du pipeline

```
S3 (images Fruits-360)
        │
        ▼
  Chargement PySpark
  (SparkContext + S3)
        │
        ▼
  Extraction features
  MobileNetV2 (Broadcast)
  via Pandas UDF
        │
        ▼
  Matérialisation Parquet
  (évite timeout socket)
        │
        ▼
  StandardScaler
        │
        ▼
  PCA k=50
  (83,7% variance)
        │
        ▼
  Résultats Parquet → S3
```

---

## ⚙️ Stack technique

| Composant | Technologie |
|-----------|------------|
| Traitement distribué | PySpark 3.x |
| Feature extraction | TensorFlow 2.11 · MobileNetV2 |
| Réduction dimensionnalité | PCA (Spark MLlib) |
| Normalisation | StandardScaler (Spark MLlib) |
| Stockage | AWS S3 · Format Parquet |
| Infrastructure | AWS EMR (eu-west-3) |
| Sérialisation modèle | Broadcast Variable |
| UDF | Pandas UDF |

---

## 🔑 Choix techniques clés

**MobileNetV2 via Broadcast Variable**
Le modèle est sérialisé et distribué à chaque nœud du cluster via une Broadcast Variable — évite le rechargement du modèle à chaque appel et optimise les performances réseau.

**Matérialisation intermédiaire**
Les features extraites sont écrites en Parquet avant l'étape StandardScaler — indispensable pour éviter les timeouts socket liés à la durée d'inférence MobileNetV2.

**Conformité RGPD**
Cluster déployé en région `eu-west-3` (Paris) — toutes les données restent sur le territoire européen.

---

## 🗂️ Structure du projet

```
OC_DS_P8_Deployer_modele_cloud/
│
├── notebooks/
│   └── P8_Pipeline_EMR.ipynb        # Exploration et développement
│
├── src/
│   └── P8_Fruits_Pipeline_EMR.py    # Script de production (EMR)
│   └── P8_Fruits_Local_Demo.py      # Démo locale (validation)
│
├── docs/
│   └── procedure_AWS_EMR.pdf        # Procédure de déploiement
│   └── notes_methodologiques.md     # Choix techniques documentés
│
└── README.md
```

---

## 🚀 Déploiement AWS EMR

### Prérequis
- Compte AWS avec rôle EMR approprié
- Bucket S3 configuré
- Cluster EMR avec bootstrap TensorFlow

### Lancement
```bash
# Via AWS Console — Cloner le cluster existant
# Cluster configuré : EMR 6.x · m5.xlarge · 3 nœuds
# Step : spark-submit P8_Fruits_Pipeline_EMR.py
```

### Variables d'environnement
```python
PATH_Data   = "s3://votre-bucket/Test"
PATH_Result = "s3://votre-bucket/Results"
```

---

## 💻 Démo locale

```bash
# Configurer l'environnement
set HADOOP_HOME=C:\hadoop
set PYSPARK_PYTHON=<path_to_python>

# Lancer la démo
python P8_Fruits_Local_Demo.py
```

Résultat attendu : 494 lignes · 4 colonnes · ~88% variance expliquée (5 classes locales vs 131 en cloud)

---

## 📚 Formation

Projet réalisé dans le cadre de la formation **Data Scientist** — [OpenClassrooms](https://openclassrooms.com)
Accréditation universitaire **WSCUC** (Western Association of Schools and Colleges — USA) · Niveau Master / Bac+5

---

## 👤 Auteur

**Stéphane Barré**
Data Scientist | PySpark · AWS · ML · NLP | Double profil Ingénieur · Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-stephane--barre--data-blue?logo=linkedin)](https://www.linkedin.com/in/stephane-barre-data)
[![GitHub](https://img.shields.io/badge/GitHub-stephanebarre13--boop-black?logo=github)](https://github.com/stephanebarre13-boop)
