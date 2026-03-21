"""
P8 — Déployez un modèle dans le cloud
Pipeline PySpark : Images S3 → MobileNetV2 → StandardScaler → PCA → Parquet S3
Aligné sur le mode opératoire P8_Notebook_Linux_EMR_PySpark_V1.0
Cluster : P8-Fruits-Cluster (emr-6.15.0)
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import io
import os

import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf

# ─────────────────────────────────────────────
# 1. CHEMINS S3
# ─────────────────────────────────────────────
PATH        = 's3://p8-fruits-bs'
PATH_Data   = PATH + '/Test'
PATH_Result = PATH + '/Results'

print('PATH:        ' + PATH)
print('PATH_Data:   ' + PATH_Data)
print('PATH_Result: ' + PATH_Result)

# ─────────────────────────────────────────────
# 2. SPARKSESSION
# ─────────────────────────────────────────────
spark = (SparkSession
             .builder
             .appName('P8')
             .config("spark.sql.parquet.writeLegacyFormat", 'true')
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .getOrCreate()
)
sc = spark.sparkContext
spark.sparkContext.setLogLevel("WARN")
print("Spark demarre :", spark.version)

# ─────────────────────────────────────────────
# 3. CHARGEMENT DES IMAGES DEPUIS S3
# ─────────────────────────────────────────────
print("Chargement des images depuis S3...")

images = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .option("recursiveFileLookup", "true") \
    .load(PATH_Data)

# Extraction du label depuis le dossier parent
images = images.withColumn('label', element_at(split(images['path'], '/'), -2))

print(images.printSchema())
print(images.select('path', 'label').show(5, False))

# ─────────────────────────────────────────────
# 4. BROADCAST DES POIDS MOBILENETV2
# ─────────────────────────────────────────────
print("Chargement et broadcast des poids MobileNetV2...")

model = MobileNetV2(weights='imagenet',
                    include_top=True,
                    input_shape=(224, 224, 3))

new_model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)

brodcast_weights = sc.broadcast(new_model.get_weights())
print("Poids broadcastes")

# ─────────────────────────────────────────────
# 5. FONCTIONS DE PREPROCESSING ET FEATURISATION
# ─────────────────────────────────────────────
def model_fn():
    """
    Reconstruit MobileNetV2 sur chaque worker avec les poids broadcastes.
    """
    model = MobileNetV2(weights='imagenet',
                        include_top=True,
                        input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    new_model = Model(inputs=model.input,
                      outputs=model.layers[-2].output)
    new_model.set_weights(brodcast_weights.value)
    return new_model


def preprocess(content):
    """
    Pretraite les bytes d'une image pour la prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)


def featurize_series(model, content_series):
    """
    Featurise une pd.Series d'images brutes.
    Retourne une pd.Series de vecteurs de features.
    """
    input_arr = np.stack(content_series.map(preprocess))
    preds = model.predict(input_arr)
    output = [p.flatten() for p in preds]
    return pd.Series(output)


@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)

# ─────────────────────────────────────────────
# 6. EXTRACTION DES FEATURES (MOBILENETV2)
# ─────────────────────────────────────────────
print("Extraction des features MobileNetV2...")

features_df = images.repartition(24).select(
    col("path"),
    col("label"),
    featurize_udf("content").alias("features")
)

print("Features extraites")

# ─────────────────────────────────────────────
# 7. STANDARDSCALER + PCA
# ─────────────────────────────────────────────
print("StandardScaler + PCA...")

# Conversion array<float> -> vecteur Spark ML
array_to_vector = udf(lambda arr: Vectors.dense(arr), VectorUDT())
features_df = features_df.withColumn("features_vec", array_to_vector(col("features")))
features_df.cache()

# StandardScaler
scaler = StandardScaler(
    inputCol="features_vec",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)
scaler_model = scaler.fit(features_df)
df_scaled = scaler_model.transform(features_df)

# PCA — k optimal pour ~80% de variance
pca = PCA(k=50, inputCol="features_scaled", outputCol="pca_features")
pca_model = pca.fit(df_scaled)
df_pca = pca_model.transform(df_scaled)

explained = float(sum(pca_model.explainedVariance))
print(f"PCA terminee — variance expliquee : {explained:.1%} avec 50 composantes")

# ─────────────────────────────────────────────
# 8. EXPORT PARQUET VERS S3
# ─────────────────────────────────────────────
print(f"Export Parquet -> {PATH_Result}")

# Conversion vecteur PCA -> array pour export
vec_to_array = udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))

df_export = df_pca.select(
    col("path"),
    col("label"),
    col("features").alias("features_mobilenet"),
    vec_to_array(col("pca_features")).alias("pca_features")
)

df_export.write.mode("overwrite").parquet(PATH_Result)

print("Parquet exporte sur S3.")

# ─────────────────────────────────────────────
# 9. VALIDATION
# ─────────────────────────────────────────────
df_check = spark.read.parquet(PATH_Result)
print(f"Résumé : {df_check.count()} lignes, {len(df_check.columns)} colonnes")
df_check.show(3, truncate=50)

spark.stop()
print("Pipeline complet. Resultats dans s3://p8-fruits-bs/Results")
