import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors
from utils.field_analysis import *

# Set Python paths
VENV_PYTHON = sys.executable
os.environ['PYSPARK_PYTHON'] = VENV_PYTHON
os.environ['PYSPARK_DRIVER_PYTHON'] = VENV_PYTHON

# Initialize Spark
spark = SparkSession.builder \
    .appName("AdvancedCropAnalyzer") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executorEnv.PYSPARK_PYTHON", VENV_PYTHON) \
    .config("spark.executorEnv.PYSPARK_DRIVER_PYTHON", VENV_PYTHON) \
    .getOrCreate()

# Load model
MODEL_PATH = "artifacts/crop_yield_model"
model = PipelineModel.load(MODEL_PATH)

# Page configuration
st.set_page_config(
    page_title="🌾 Advanced Crop Yield Analyzer",
    page_icon="🌱",
    layout="wide"
)

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Analysis Parameters")
    
    # Environmental factors
    st.subheader("Environmental Factors")
    rainfall = st.slider("💧 Rainfall (mm)", 0.0, 300.0, 100.0)
    temperature = st.slider("🌡️ Average Temperature (°C)", 10.0, 40.0, 25.0)
    soil_ph = st.slider("🧪 Soil pH", 4.0, 9.0, 6.5)
    fertilizer = st.slider("🧴 Fertilizer Usage (kg/ha)", 0.0, 300.0, 100.0)
    
    # Disease risk parameters
    st.subheader("Disease Risk Parameters")
    humidity = st.slider("Humidity (%)", 0, 100, 65)
    leaf_wetness = st.slider("Leaf Wetness (hours)", 0.0, 24.0, 5.0)
    
    # Economic parameters
    st.subheader("Economic Parameters")
    crop_price = st.number_input("Crop Price ($/ton)", 100, 300, 180)
    labor_cost = st.number_input("Labor Cost ($/ha)", 50, 500, 150)

# Main content
st.title("🌾 Advanced Crop Yield Analyzer")
st.markdown("Predict crop yields using satellite imagery and environmental parameters")

# File upload section
st.header("1. Satellite Imagery Analysis")
uploaded_file = st.file_uploader("Upload NDVI GeoTIFF", type=["tif", "tiff"], 
                                 help="Upload GeoTIFF containing NDVI data")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
    st.session_state.trend_data = {"dates": [], "ndvi_values": []}

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        with rasterio.open(tmp_path) as src:
            ndvi = src.read(1)
            ndvi = np.clip(ndvi, -1, 1)
            
            # Calculate statistics
            ndvi_min = np.min(ndvi)
            ndvi_max = np.max(ndvi)
            ndvi_mean = np.mean(ndvi)
            ndvi_median = np.median(ndvi)
            ndvi_std = np.std(ndvi)
            
            # Create columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("NDVI Visualization")
                fig, ax = plt.subplots()
                im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax, label="NDVI Value")
                st.pyplot(fig)
                
                # Add to time-series data
                if len(st.session_state.trend_data["dates"]) < 10:
                    st.session_state.trend_data["dates"].append(datetime.now())
                    st.session_state.trend_data["ndvi_values"].append(ndvi_mean)
            
            with col2:
                st.subheader("Field Analysis")
                
                # Statistics table
                stats_data = {
                    "Metric": ["Minimum", "Maximum", "Mean", "Median", "Std Dev"],
                    "Value": [f"{ndvi_min:.4f}", f"{ndvi_max:.4f}", 
                             f"{ndvi_mean:.4f}", f"{ndvi_median:.4f}", 
                             f"{ndvi_std:.4f}"]
                }
                st.dataframe(pd.DataFrame(stats_data), hide_index=True)
                
                # Health indicator
                health = (ndvi_mean + 1) / 2
                st.progress(health, text=f"Crop Health: {health*100:.1f}%")
                
                # Prepare for prediction
                features = Vectors.dense([ndvi_mean])
                df = spark.createDataFrame([(features,)], ["features"])
                
                # Predict
                prediction = model.transform(df).collect()[0]["prediction"]
                st.subheader("Yield Prediction")
                st.metric("Predicted Yield", f"{prediction:.2f} tons/ha")
                
                # Health assessment text
                if health > 0.7:
                    st.success("✅ Excellent crop health - optimal growing conditions")
                elif health > 0.5:
                    st.info("🟢 Good crop health - normal conditions")
                elif health > 0.3:
                    st.warning("🟡 Moderate crop health - monitor closely")
                else:
                    st.error("🔴 Poor crop health - intervention needed")
                
                # Save prediction to history
                st.session_state.predictions.append({
                    "timestamp": datetime.now(),
                    "ndvi": ndvi_mean,
                    "yield": prediction,
                    "health": health
                })
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
    finally:
        os.unlink(tmp_path)

# Advanced Analytics Section
st.divider()
st.header("🔍 Advanced Analytics")

if 'predictions' in st.session_state and st.session_state.predictions:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disease Risk Assessment")
        risk = predict_disease_risk(
            st.session_state.predictions[-1]["ndvi"],
            humidity,
            leaf_wetness
        )
        if "High" in risk:
            st.error(f"🚨 {risk}")
        elif "Moderate" in risk:
            st.warning(f"⚠️ {risk}")
        else:
            st.success(f"✅ {risk}")
        
        st.subheader("Water Stress Analysis")
        stress = calculate_water_stress(
            st.session_state.predictions[-1]["ndvi"],
            temperature
        )
        st.metric("Stress Index", f"{stress:.2f}")
        if stress > 7:
            st.error("Severe water stress detected! Increase irrigation.")
        elif stress > 4:
            st.warning("Moderate water stress - monitor soil moisture")
        else:
            st.success("Normal water conditions")
    
    with col2:
        st.subheader("Economic Impact")
        if st.session_state.predictions:
            last_prediction = st.session_state.predictions[-1]["yield"]
            profit = calculate_profit(last_prediction, fertilizer, labor_cost, crop_price)
            roi = calculate_roi(last_prediction, fertilizer, crop_price)
            
            st.metric("Estimated Profit", f"${profit:.2f}/ha")
            st.metric("Return on Investment", f"{roi:.1f}%")
            
            if roi > 200:
                st.success("Excellent ROI - optimal fertilizer usage")
            elif roi > 100:
                st.info("Good ROI - consider minor adjustments")
            else:
                st.warning("Low ROI - review fertilizer strategy")

# Time-Series Trend Analysis
st.divider()
st.header("📈 Time-Series Trend Analysis")

if len(st.session_state.trend_data["dates"]) > 1:
    trend = analyze_trend(
        st.session_state.trend_data["dates"],
        st.session_state.trend_data["ndvi_values"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("NDVI Trend")
        st.metric("Trend Slope", f"{trend:.4f} per 1000 days")
        if trend > 0:
            st.success("Positive trend - crop health improving")
        else:
            st.warning("Negative trend - crop health declining")
    
    with col2:
        st.subheader("Historical NDVI Values")
        fig, ax = plt.subplots()
        ax.plot(st.session_state.trend_data["dates"], 
                st.session_state.trend_data["ndvi_values"], 
                'o-')
        ax.set_xlabel("Date")
        ax.set_ylabel("NDVI Mean")
        ax.set_title("NDVI Trend Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Scenario Testing Section
st.divider()
st.header("🧪 Scenario Testing")

scenario_col1, scenario_col2 = st.columns(2)

with scenario_col1:
    st.subheader("Environmental Factors")
    sim_rainfall = st.slider("💧 Rainfall (mm)", 0.0, 300.0, 100.0, key="scenario_rain")
    sim_temp = st.slider("🌡️ Temperature (°C)", 10.0, 40.0, 25.0, key="scenario_temp")
    sim_ph = st.slider("🧪 Soil pH", 4.0, 9.0, 6.5, key="scenario_ph")
    sim_fert = st.slider("🧴 Fertilizer (kg/ha)", 0.0, 300.0, 100.0, key="scenario_fert")

with scenario_col2:
    st.subheader("Crop Health")
    sim_ndvi = st.slider("NDVI", -1.0, 1.0, 0.4, key="scenario_ndvi")
    st.write("")  # Spacer
    
    if st.button("Simulate This Scenario", type="primary", use_container_width=True):
        try:
            features = Vectors.dense([sim_ndvi])
            df = spark.createDataFrame([(features,)], ["features"])
            prediction = model.transform(df).collect()[0]["prediction"]
            
            # Calculate analytics
            health = (sim_ndvi + 1) / 2
            stress = calculate_water_stress(sim_ndvi, sim_temp)
            risk = predict_disease_risk(sim_ndvi, humidity, leaf_wetness)
            profit = calculate_profit(prediction, sim_fert, labor_cost, crop_price)
            roi = calculate_roi(prediction, sim_fert, crop_price)
            
            # Display results
            st.success(f"🌾 Predicted Yield: {prediction:.2f} tons/ha")
            st.info(f"🧪 Crop Health: {health:.1%}")
            st.metric("Water Stress", f"{stress:.2f}")
            st.metric("Disease Risk", risk)
            st.metric("Estimated Profit", f"${profit:.2f}/ha")
            st.metric("ROI", f"{roi:.1f}%")
            
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")

# Footer
st.divider()
st.caption("Advanced Crop Yield Prediction System | Powered by PySpark and Satellite Imagery")

# Cleanup
spark.stop()