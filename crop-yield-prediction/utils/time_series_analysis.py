import os
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType

# Initialize Spark
spark = SparkSession.builder.appName("TimeSeriesAnalysis").getOrCreate()

def compute_ndvi_trend(field_id, date_paths):
    """Calculate NDVI trend slope for a field over time"""
    ndvi_values = []
    dates = []
    
    for date_str, path in date_paths:
        with rasterio.open(path) as src:
            ndvi = src.read(1)
            ndvi = np.clip(ndvi, -1, 1)
            ndvi_values.append(np.mean(ndvi))
            dates.append(datetime.strptime(date_str, "%Y%m%d").toordinal())
    
    # Calculate linear trend (slope)
    if len(dates) > 1:
        slope = np.polyfit(dates, ndvi_values, 1)[0]
        return float(slope * 1000)  # Convert to per-thousand days
    return 0.0

# Register UDF
ndvi_trend_udf = udf(compute_ndvi_trend, FloatType())

# Main analysis function
def analyze_field_trends(data_dir="data/time_series"):
    """Process all fields and calculate NDVI trends"""
    # Load field metadata
    fields = []
    for field_dir in os.listdir(data_dir):
        field_path = os.path.join(data_dir, field_dir)
        if os.path.isdir(field_path):
            date_paths = []
            for file in os.listdir(field_path):
                if file.endswith(".tif"):
                    date_str = file.split("_")[0]
                    date_paths.append((date_str, os.path.join(field_path, file)))
            if date_paths:
                fields.append((field_dir, date_paths))
    
    # Create DataFrame
    df = spark.createDataFrame(fields, ["field_id", "date_paths"])
    
    # Calculate trends
    result = df.withColumn("ndvi_trend", ndvi_trend_udf(col("field_id"), col("date_paths")))
    
    # Convert to Pandas for visualization
    trends_df = result.toPandas()
    
    # Generate trend visualization
    plt.figure(figsize=(10, 6))
    plt.bar(trends_df["field_id"], trends_df["ndvi_trend"], 
            color=np.where(trends_df["ndvi_trend"] > 0, 'green', 'red'))
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Field NDVI Trends (per 1000 days)")
    plt.ylabel("Trend Slope")
    plt.xlabel("Field ID")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/field_trends.png")
    
    return trends_df

if __name__ == "__main__":
    trends = analyze_field_trends()
    print(trends)