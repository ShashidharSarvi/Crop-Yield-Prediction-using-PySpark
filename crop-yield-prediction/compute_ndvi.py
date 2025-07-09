import os
import numpy as np
import rasterio
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import FloatType, StructType, StructField
import matplotlib.pyplot as plt

def calculate_ndvi(red_band, nir_band):
    """Robust NDVI calculation with error handling"""
    # Convert to float and handle potential issues
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    
    # Mask invalid values
    red[red <= 0] = np.nan
    nir[nir <= 0] = np.nan
    
    # Calculate denominator safely
    denominator = nir + red
    denominator[denominator == 0] = np.nan
    
    # Calculate NDVI
    ndvi = (nir - red) / denominator
    
    # Clip to valid range and fill NaNs
    ndvi = np.clip(ndvi, -1, 1)
    ndvi = np.nan_to_num(ndvi, nan=-1)
    
    return ndvi

def process_tile(path):
    """Process a single tile to compute NDVI"""
    try:
        with rasterio.open(path) as src:
            # Try different band configurations
            for band_config in [
                (4, 8),   # Sentinel-2: Red=Band4, NIR=Band8
                (3, 4),   # Landsat 8: Red=Band4, NIR=Band5 -> (4,5) but 1-based index
                (3, 4),   # Common: Red=Band3, NIR=Band4
                (1, 2)    # Fallback: First two bands
            ]:
                try:
                    red_band = src.read(band_config[0])
                    nir_band = src.read(band_config[1])
                    ndvi = calculate_ndvi(red_band, nir_band)
                    
                    # Create NDVI output path
                    output_path = path.replace(".tif", "_NDVI.tif")
                    
                    # Save as new GeoTIFF
                    profile = src.profile
                    profile.update(
                        dtype=rasterio.float32,
                        count=1,
                        nodata=-9999
                    )
                    
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(ndvi.astype(rasterio.float32), 1)
                    
                    print(f"Successfully processed: {path}")
                    return output_path
                
                except (IndexError, rasterio.RasterioIOError):
                    continue
            
            print(f"Could not find suitable bands for: {path}")
            return None
    
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return None

def main():
    spark = SparkSession.builder.appName("NDVIComputation").getOrCreate()
    
    # Get list of TIFF files
    data_dir = "data"
    tif_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if f.lower().endswith((".tif", ".tiff")) and "_NDVI" not in f]
    
    # Parallelize processing
    rdd = spark.sparkContext.parallelize(tif_files)
    results = rdd.map(process_tile).collect()
    
    # Print results
    print("\nProcessing Summary:")
    success = [r for r in results if r is not None]
    failed = [tif_files[i] for i, r in enumerate(results) if r is None]
    
    print(f"Successfully processed: {len(success)} files")
    print(f"Failed to process: {len(failed)} files")
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f" - {f}")
    
    spark.stop()

if __name__ == "__main__":
    main()