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

# Configuration
st.set_page_config(
    page_title="🌾 Advanced Crop Analytics Platform",
    page_icon="🌱",
    layout="wide"
)

# Set Python paths
VENV_PYTHON = sys.executable
os.environ['PYSPARK_PYTHON'] = VENV_PYTHON
os.environ['PYSPARK_DRIVER_PYTHON'] = VENV_PYTHON

# Initialize Spark
spark = SparkSession.builder \
    .appName("CropAnalytics") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load model
MODEL_PATH = "artifacts/crop_yield_model"
model = PipelineModel.load(MODEL_PATH)

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Environmental Parameters")
    rainfall = st.slider("💧 Rainfall (mm)", 0.0, 300.0, 100.0)
    temperature = st.slider("🌡️ Temperature (°C)", 10.0, 40.0, 25.0)
    soil_ph = st.slider("🧪 Soil pH", 4.0, 9.0, 6.5)
    fertilizer = st.slider("🧴 Fertilizer (kg/ha)", 0.0, 300.0, 100.0)
    
    st.header("🦠 Disease Risk Parameters")
    humidity = st.slider("Humidity (%)", 0, 100, 65)
    leaf_wetness = st.slider("Leaf Wetness (hours)", 0.0, 24.0, 5.0)
    
    st.header("💰 Economic Parameters")
    crop_price = st.number_input("Crop Price ($/ton)", 100, 300, 180)
    labor_cost = st.number_input("Labor Cost ($/ha)", 50, 500, 150)

# Main dashboard
st.title("🌾 Advanced Crop Analytics Platform")
st.markdown("Predict crop performance using satellite imagery and environmental data")

# Initialize session state
if 'field_history' not in st.session_state:
    st.session_state.field_history = {
        'dates': [],
        'ndvi_values': [],
        'yield_predictions': []
    }

# File upload section
st.header("📡 Satellite Imagery Analysis")
uploaded_file = st.file_uploader("Upload raw satellite image (GeoTIFF)", 
                                type=["tif", "tiff"])

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        with rasterio.open(tmp_path) as src:
            # Process different band configurations
            ndvi = None
            for band_config in [(4, 8), (3, 4), (1, 2)]:
                try:
                    red = src.read(band_config[0]).astype(float)
                    nir = src.read(band_config[1]).astype(float)
                    
                    # Calculate NDVI
                    denom = nir + red
                    denom[denom == 0] = 0.01
                    ndvi = (nir - red) / denom
                    ndvi = np.clip(ndvi, -1, 1)
                    break
                except (IndexError, rasterio.RasterioIOError):
                    continue
            
            if ndvi is None:
                st.error("❌ Could not process image: Unsupported band configuration")
                st.stop()
            
            # Calculate statistics
            ndvi_mean = np.nanmean(ndvi)
            valid_pixels = np.sum((ndvi > -1) & (ndvi < 1))
            total_pixels = ndvi.size
            health = (ndvi_mean + 1) / 2
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("NDVI Visualization")
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax, label="NDVI Value")
                st.pyplot(fig)
                
                # Add to time-series data
                st.session_state.field_history['dates'].append(datetime.now())
                st.session_state.field_history['ndvi_values'].append(ndvi_mean)
            
            with col2:
                st.subheader("Field Analysis")
                
                # Statistics
                metrics = {
                    "Mean NDVI": ndvi_mean,
                    "Min NDVI": np.nanmin(ndvi),
                    "Max NDVI": np.nanmax(ndvi),
                    "Std Dev": np.nanstd(ndvi),
                    "Healthy Pixels": f"{valid_pixels/total_pixels:.1%}"
                }
                st.dataframe(pd.DataFrame(list(metrics.items()), 
                                        columns=["Metric", "Value"]))
                
                # Health indicator
                st.metric("Crop Health Index", f"{health:.1%}")
                st.progress(health)
                
                # Prepare for prediction
                features = Vectors.dense([
                    ndvi_mean,
                    rainfall,
                    temperature,
                    soil_ph,
                    fertilizer
                ])
                
                # Predict yield
                df = spark.createDataFrame([(features,)], ["features"])
                prediction = model.transform(df).collect()[0]["prediction"]
                st.session_state.field_history['yield_predictions'].append(prediction)
                
                st.subheader("Yield Prediction")
                st.metric("Predicted Yield", f"{prediction:.2f} tons/ha", 
                         delta="Optimal" if prediction > 2.5 else "Below Average")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
    finally:
        os.unlink(tmp_path)

# Advanced Analytics Section
st.divider()
st.header("🔍 Advanced Field Analytics")

if st.session_state.field_history['dates']:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disease Risk Assessment")
        risk = predict_disease_risk(
            st.session_state.field_history['ndvi_values'][-1],
            humidity,
            leaf_wetness,
            temperature
        )
        if "High" in risk:
            st.error(f"🚨 {risk}")
            st.info("Recommended action: Apply fungicide treatment")
        elif "Moderate" in risk:
            st.warning(f"⚠️ {risk}")
            st.info("Recommended action: Monitor closely and improve airflow")
        else:
            st.success(f"✅ {risk}")
        
        st.subheader("Water Stress Analysis")
        stress = calculate_water_stress(
            st.session_state.field_history['ndvi_values'][-1],
            temperature,
            rainfall
        )
        st.metric("Stress Index", f"{stress:.2f}/10")
        if stress > 7:
            st.error("Severe water stress! Increase irrigation by 30%")
        elif stress > 4:
            st.warning("Moderate stress - consider 15% irrigation increase")
        else:
            st.success("Optimal water conditions")
    
    with col2:
        st.subheader("Economic Impact")
        if st.session_state.field_history['yield_predictions']:
            last_yield = st.session_state.field_history['yield_predictions'][-1]
            profit = calculate_profit(last_yield, fertilizer, labor_cost, crop_price)
            roi = calculate_roi(last_yield, fertilizer, crop_price)
            
            st.metric("Estimated Profit", f"${profit:.2f}/ha")
            st.metric("Return on Investment", f"{roi:.1f}%")
            
            # Recommendation engine
            if roi < 50:
                st.warning("Low ROI - optimize fertilizer usage")
            elif roi < 100:
                st.info("Moderate ROI - consider precision agriculture techniques")
            else:
                st.success("Excellent ROI - maintain current practices")
            
            # Cost breakdown
            cost_data = {
                "Revenue": last_yield * crop_price,
                "Fertilizer Cost": fertilizer,
                "Labor Cost": labor_cost
            }
            st.bar_chart(cost_data)

# Time-Series Analysis
st.divider()
st.header("📈 Historical Field Performance")

if len(st.session_state.field_history['dates']) > 1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crop Health Trend")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(st.session_state.field_history['dates'], 
                st.session_state.field_history['ndvi_values'], 
                'o-', color='green')
        ax.set_title("NDVI Trend Over Time")
        ax.set_ylabel("NDVI Value")
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Yield Projections")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(st.session_state.field_history['dates'], 
                st.session_state.field_history['yield_predictions'], 
                's-', color='blue')
        ax.set_title("Yield Prediction Trend")
        ax.set_ylabel("Tons per Hectare")
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Trend analysis
    trend = analyze_trend(
        st.session_state.field_history['dates'],
        st.session_state.field_history['ndvi_values']
    )
    trend_direction = "improving" if trend > 0 else "declining"
    st.info(f"Field health is {trend_direction} at a rate of {abs(trend):.2f} NDVI units per year")

# What-If Scenarios
st.divider()
st.header("🧪 What-If Scenario Analysis")

scenario_col1, scenario_col2 = st.columns(2)

with scenario_col1:
    st.subheader("Environmental Factors")
    sim_rain = st.slider("💧 Simulated Rainfall (mm)", 0.0, 300.0, 100.0)
    sim_temp = st.slider("🌡️ Simulated Temperature (°C)", 10.0, 40.0, 25.0)
    sim_ph = st.slider("🧪 Simulated Soil pH", 4.0, 9.0, 6.5)
    sim_fert = st.slider("🧴 Simulated Fertilizer (kg/ha)", 0.0, 300.0, 100.0)
    sim_ndvi = st.slider("🌿 Simulated NDVI", -1.0, 1.0, 0.4)

with scenario_col2:
    st.subheader("Simulation Results")
    if st.button("Run Simulation", type="primary", use_container_width=True):
        try:
            # Create feature vector
            features = Vectors.dense([
                sim_ndvi,
                sim_rain,
                sim_temp,
                sim_ph,
                sim_fert
            ])
            
            # Predict yield
            df = spark.createDataFrame([(features,)], ["features"])
            prediction = model.transform(df).collect()[0]["prediction"]
            
            # Calculate analytics
            health = (sim_ndvi + 1) / 2
            stress = calculate_water_stress(sim_ndvi, sim_temp, sim_rain)
            risk = predict_disease_risk(sim_ndvi, humidity, leaf_wetness, sim_temp)
            profit = calculate_profit(prediction, sim_fert, labor_cost, crop_price)
            roi = calculate_roi(prediction, sim_fert, crop_price)
            
            # Display results
            st.success(f"🌾 Predicted Yield: {prediction:.2f} tons/ha")
            st.info(f"🧪 Crop Health: {health:.1%}")
            st.metric("Water Stress", f"{stress:.2f}/10")
            st.metric("Disease Risk", risk)
            st.metric("Estimated Profit", f"${profit:.2f}/ha")
            st.metric("ROI", f"{roi:.1f}%")
            
            # Recommendations
            if prediction < 2.0:
                st.warning("⚠️ Low yield predicted - consider soil amendments")
            elif stress > 6:
                st.warning("⚠️ High water stress - optimize irrigation")
            elif "High" in risk:
                st.warning("⚠️ Disease risk high - apply preventative measures")
            
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")

# Footer
st.divider()
st.caption("Advanced Crop Analytics Platform | © 2025 | Powered by PySpark and Satellite Intelligence")

# Cleanup
spark.stop()