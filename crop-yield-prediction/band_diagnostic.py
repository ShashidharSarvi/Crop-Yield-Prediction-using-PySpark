import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_bands(path):
    """Visualize and analyze bands in a GeoTIFF"""
    with rasterio.open(path) as src:
        print(f"\nFile: {os.path.basename(path)}")
        print(f"Bands: {src.count}")
        print(f"Dimensions: {src.width}x{src.height}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        
        # Create subplots
        fig, axes = plt.subplots(1, src.count, figsize=(15, 5))
        fig.suptitle(f"Band Visualization: {os.path.basename(path)}")
        
        for i in range(src.count):
            band = src.read(i+1)
            axes[i].imshow(band, cmap='viridis')
            axes[i].set_title(f"Band {i+1}")
            
            # Print band stats
            print(f"  Band {i+1}: min={np.nanmin(band):.2f}, "
                  f"max={np.nanmax(band):.2f}, "
                  f"mean={np.nanmean(band):.2f}")
        
        plt.tight_layout()
        plt.savefig(f"band_analysis_{os.path.basename(path)}.png")
        plt.show()

if __name__ == "__main__":
    data_dir = "data"
    for file in os.listdir(data_dir):
        if file.lower().endswith((".tif", ".tiff")):
            try:
                analyze_bands(os.path.join(data_dir, file))
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                