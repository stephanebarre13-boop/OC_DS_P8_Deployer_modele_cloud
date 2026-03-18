# Déployez un modèle dans le cloud — Projet 8

**OpenClassrooms · Data Scientist · Mars 2026**  
**Auteur : Stéphane BARRÉ**

---

## Objectif

Mettre en place une chaîne de traitement Big Data dans le cloud AWS pour extraire les features de 22 688 images de fruits, en vue d'alimenter un futur modèle de classification.

La start-up **Fruits!** souhaite développer une application mobile de reconnaissance de fruits. Ce projet en constitue la première brique technique.

---

## Résultats clés

| Métrique | Valeur |
|---|---|
| Images traitées | 22 688 |
| Classes de fruits | 131 |
| Features par image (MobileNetV2) | 1 280 |
| Composantes PCA retenues | 50 |
| Variance expliquée | 83,7 % |
| Taille sortie Parquet | 92 MB |
| Région AWS | eu-west-3 Paris (RGPD) |
| Cluster EMR | j-10D17Y4KI6SBM |
| Step | s-00367443M91CBHEI65G8 — COMPLETED |

---

## Architecture AWS

```
IAM (p8-user + rôles EMR)
    |
    v
S3 (s3://p8-fruits-bs/)
    ├── /Test/      -> 22 688 images JPG sources
    ├── /scripts/   -> bootstrap.sh + P8_Fruits_Cloud_v2.py
    ├── /logs/      -> journaux EMR
    └── /Results/   -> 92 MB Parquet Snappy (sortie PCA)
    |
    v
EMR (emr-6.15.0 · Spark 3.4.1 · 2x m5.xlarge · eu-west-3)
```

---

## Pipeline PySpark — 7 étapes

```
1. binaryFile      -> chargement images depuis S3
2. Label           -> extraction classe depuis chemin (element_at)
3. Broadcast       -> diffusion poids MobileNetV2 (14 MB x 1)  *
4. Pandas UDF      -> featurisation SCALAR_ITER (1 280 features)
5. StandardScaler  -> normalisation µ=0, σ=1
6. PCA k=50        -> réduction 1 280 -> 50 dims (83,7 % variance)
7. write.parquet   -> export 92 MB sur S3 (lazy evaluation)
```

\* Elément ajouté par rapport au notebook de l'alternant

---

## Points techniques

- **Broadcast des poids** : le driver charge MobileNetV2 une seule fois (~14 MB) et distribue à tous les workers via réseau local. Sans broadcast : N x 14 MB de trafic réseau.
- **Pandas UDF SCALAR_ITER** : le modèle est chargé une seule fois par worker (pas par image). Optimal pour le Deep Learning distribué.
- **repartition(24)** : surprovisionnement ~3 partitions/coeur physique. Anti-bottleneck.
- **StandardScaler obligatoire avant PCA** : MobileNetV2 produit des features d'échelles très différentes. Sans normalisation, la PCA est biaisée.
- **PCA k optimal** : recherche automatique du premier k depassant 80 % de variance. k=40 -> 79,4 % (insuffisant). k=50 -> 83,7 % (retenu).
- **RGPD** : eu-west-3 Paris. Données jamais hors d'Europe.

---

## Configuration EMR

```
Version     : emr-6.15.0
Spark       : 3.4.1
Hadoop      : 3.3.6
Instances   : 2x m5.xlarge (4 vCPU, 16 GB RAM chacune)
Region      : eu-west-3 Paris
Bootstrap   : tensorflow-cpu==2.11.0 + pandas + pillow + pyarrow + urllib3<2.0
```

---

## Livrables

| Fichier | Description |
|---|---|
| `Barre_Stephane_1_notebook_032026.ipynb` | Pipeline complet local + cloud avec outputs |
| `P8_Fruits_Cloud_v2.py` | Script spark-submit production |

---

## Structure du repo

```
OC_DS_P8_Deployer_modele_cloud/
├── README.md
├── Barre_Stephane_1_notebook_032026.ipynb
└── P8_Fruits_Cloud_v2.py
```

---

## Accès S3 (lecture)

Les résultats sont disponibles dans le bucket S3 :
```
s3://p8-fruits-bs/Results/   -> 24 fichiers Parquet Snappy (92 MB)
Region : eu-west-3 Paris
```

---

## Projet précédent

[Projet 7 — Scoring Credit](https://github.com/stephanebarre13-boop/Barre_Stephane_P7)
