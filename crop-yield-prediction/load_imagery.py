from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import rasterio
import os

def extract_metadata(path):
    """Reads a GeoTIFF and returns basic metadata."""
    with rasterio.open(path) as src:
        b = src.bounds
        return {
            "path": path,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "bounds": {
                "left": b.left,
                "bottom": b.bottom,
                "right": b.right,
                "top": b.top
            }
        }


if __name__ == "__main__":
    # 1. Start Spark
    spark = SparkSession.builder \
        .appName("CropYieldDataIngest") \
        .getOrCreate()

    # 2. List all GeoTIFF files
    data_dir = "data"
    files = [os.path.join(data_dir, f)
             for f in os.listdir(data_dir)
             if f.lower().endswith((".tif", ".tiff"))]

    # 3. Parallelize and extract metadata
    rdd = spark.sparkContext.parallelize(files)
    meta_rdd = rdd.map(extract_metadata)

    # 4. Define schema for a DataFrame
    schema = StructType([
        StructField("path", StringType(), False),
        StructField("width", IntegerType(), False),
        StructField("height", IntegerType(), False),
        StructField("crs", StringType(), False),
        StructField("bounds", 
            StructType([
                StructField("left", DoubleType(), False),
                StructField("bottom", DoubleType(), False),
                StructField("right", DoubleType(), False),
                StructField("top", DoubleType(), False),
            ]), False),
    ])

    # 5. Convert to DataFrame
    df = spark.createDataFrame(meta_rdd, schema=schema)

    # 6. Show the ingested metadata
    df.show(truncate=False)

    spark.stop()
