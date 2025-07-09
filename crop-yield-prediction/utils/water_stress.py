import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Initialize Spark
spark = SparkSession.builder.appName("WaterStressAnalysis").getOrCreate()

def calculate_water_stress(ndvi, temperature):
    """Calculate water stress index"""
    # Normalize parameters
    norm_ndvi = (ndvi + 1) / 2  # Scale to 0-1
    norm_temp = (temperature - 10) / 30  # Scale 10-40°C to 0-1
    
    # Calculate stress index (higher = more stress)
    if norm_ndvi < 0.01:  # Prevent division by zero
        return 10.0
    return (1 - norm_ndvi) * (norm_temp ** 2) * 10

# Register UDF
water_stress_udf = udf(calculate_water_stress, FloatType())

# Analysis function
def analyze_water_stress(data_path="data/field_data.csv"):
    """Analyze water stress across fields"""
    # Load field data
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    # Calculate stress
    result = df.withColumn("water_stress", 
                          water_stress_udf(col("mean_ndvi"), col("temperature")))
    
    # Convert to Pandas
    stress_df = result.toPandas()
    
    # Generate visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(stress_df["mean_ndvi"], stress_df["temperature"], 
               c=stress_df["water_stress"], cmap="RdYlGn_r", s=100)
    plt.colorbar(label="Water Stress Index")
    plt.xlabel("Mean NDVI")
    plt.ylabel("Temperature (°C)")
    plt.title("Field Water Stress Analysis")
    plt.grid(True)
    plt.savefig("output/water_stress.png")
    
    return stress_df

if __name__ == "__main__":
    stress_data = analyze_water_stress()
    print(stress_data[["field_id", "water_stress"]])