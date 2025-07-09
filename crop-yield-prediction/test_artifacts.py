from pyspark.sql import SparkSession
import os

# Initialize Spark
spark = SparkSession.builder \
    .appName("FileCreationTest") \
    .getOrCreate()

# Create artifacts directory if needed
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# Test file creation
test_file_path = os.path.join(artifacts_dir, "test_file.txt")
try:
    with open(test_file_path, "w") as f:
        f.write("test")
    print(f"✅ File created at: {os.path.abspath(test_file_path)}")
    
    # Verify
    if os.path.exists(test_file_path):
        print(f"✅ Verification: File exists! Size: {os.path.getsize(test_file_path)} bytes")
        print(f"✅ Directory listing: {os.listdir(artifacts_dir)}")
    else:
        print(f"❌ Error: File not found at {test_file_path}")
        
except Exception as e:
    print(f"❌ Failed to create file: {str(e)}")

# Cleanup
spark.stop()