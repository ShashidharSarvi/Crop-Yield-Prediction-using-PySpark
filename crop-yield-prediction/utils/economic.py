import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType

# Initialize Spark
spark = SparkSession.builder.appName("EconomicImpact").getOrCreate()

def calculate_profit(yield_value, fertilizer_cost, labor_cost=150, crop_price=180):
    """Calculate profit from crop yield"""
    revenue = yield_value * crop_price
    total_cost = fertilizer_cost + labor_cost
    return revenue - total_cost

def calculate_roi(yield_value, fertilizer_cost, crop_price=180):
    """Calculate return on investment for fertilizer"""
    revenue = yield_value * crop_price
    return (revenue - fertilizer_cost) / fertilizer_cost * 100

# Register UDFs
profit_udf = udf(calculate_profit, FloatType())
roi_udf = udf(calculate_roi, FloatType())

# Analysis function
def analyze_economic_impact(data_path="data/yield_data.csv"):
    """Analyze economic impact of farming decisions"""
    # Load yield data
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    # Calculate economic metrics
    result = df.withColumn("profit", profit_udf(col("yield"), col("fertilizer_cost"))) \
              .withColumn("roi", roi_udf(col("yield"), col("fertilizer_cost")))
    
    # Convert to Pandas
    economic_df = result.toPandas()
    
    # Generate visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Profit vs Fertilizer
    ax1.scatter(economic_df["fertilizer_cost"], economic_df["profit"], 
               c=economic_df["roi"], cmap="RdYlGn", s=100)
    ax1.set_xlabel("Fertilizer Cost ($)")
    ax1.set_ylabel("Profit ($)")
    ax1.set_title("Profit vs Fertilizer Investment")
    
    # ROI Distribution
    ax2.hist(economic_df["roi"], bins=20, color="skyblue", edgecolor="black")
    ax2.axvline(economic_df["roi"].mean(), color="red", linestyle="--")
    ax2.set_xlabel("ROI (%)")
    ax2.set_ylabel("Number of Fields")
    ax2.set_title("Return on Investment Distribution")
    
    plt.tight_layout()
    plt.savefig("output/economic_impact.png")
    
    return economic_df

if __name__ == "__main__":
    economic_data = analyze_economic_impact()
    print(economic_data[["field_id", "profit", "roi"]])