from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

# Load data with new parameters
df = pd.merge(features_df, labels_df, on="filename")
spark_df = spark.createDataFrame(df)

# Convert region to numeric index
indexer = StringIndexer(inputCol="region", outputCol="region_index")

# Assemble all features
assembler = VectorAssembler(
    inputCols=["mean_ndvi", "rainfall", "temperature", "soil_ph", "fertilizer", "region_index"],
    outputCol="features"
)

# Use more powerful algorithm
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="yield",
    numTrees=100,
    maxDepth=5,
    seed=42
)

# Create pipeline
pipeline = Pipeline(stages=[indexer, assembler, rf])
model = pipeline.fit(spark_df)

# Save model
model.write().overwrite().save("artifacts/advanced_yield_model")