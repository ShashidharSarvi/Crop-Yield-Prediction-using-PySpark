import pandas as pd
import numpy as np
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder.appName("DiseaseRiskPrediction").getOrCreate()

# Sample dataset (replace with real data)
data = [
    {"ndvi": 0.65, "humidity": 75, "leaf_wetness": 6.2, "disease_risk": 1},
    {"ndvi": 0.42, "humidity": 85, "leaf_wetness": 8.1, "disease_risk": 2},
    {"ndvi": 0.38, "humidity": 90, "leaf_wetness": 9.5, "disease_risk": 3},
    {"ndvi": 0.71, "humidity": 60, "leaf_wetness": 4.3, "disease_risk": 0},
    {"ndvi": 0.53, "humidity": 78, "leaf_wetness": 7.0, "disease_risk": 1}
]

# Create DataFrame
df = spark.createDataFrame(data)

# Feature engineering pipeline
assembler = VectorAssembler(
    inputCols=["ndvi", "humidity", "leaf_wetness"],
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

# Risk levels: 0=None, 1=Low, 2=Medium, 3=High
classifier = RandomForestClassifier(
    featuresCol="features",
    labelCol="disease_risk",
    numTrees=50,
    maxDepth=5,
    seed=42
)

# Create pipeline
pipeline = Pipeline(stages=[assembler, scaler, classifier])
model = pipeline.fit(df)

# Save model
model.write().overwrite().save("models/disease_risk_model")

# Prediction function
def predict_disease_risk(ndvi, humidity, leaf_wetness):
    """Predict disease risk level (0-3)"""
    data = [(float(ndvi), float(humidity), float(leaf_wetness))]
    columns = ["ndvi", "humidity", "leaf_wetness"]
    df = spark.createDataFrame(data, columns)
    
    model = PipelineModel.load("models/disease_risk_model")
    prediction = model.transform(df).collect()[0]["prediction"]
    
    risk_levels = {
        0: "No risk",
        1: "Low risk",
        2: "Medium risk",
        3: "High risk"
    }
    
    return risk_levels.get(prediction, "Unknown")

if __name__ == "__main__":
    # Test prediction
    risk = predict_disease_risk(0.48, 82, 7.8)
    print(f"Disease Risk: {risk}")