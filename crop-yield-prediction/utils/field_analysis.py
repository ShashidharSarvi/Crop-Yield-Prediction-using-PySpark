import numpy as np
import pandas as pd

def calculate_water_stress(ndvi, temperature, rainfall):
    """Enhanced water stress calculation"""
    # Normalize parameters
    ndvi_norm = np.clip((ndvi + 1) / 2, 0.01, 1)  # Scale to 0-1 with min 0.01
    temp_norm = np.clip((temperature - 10) / 30, 0, 1)  # Scale 10-40°C to 0-1
    rain_norm = np.clip(1 - (rainfall / 300), 0, 1)  # Inverse rainfall
    
    # Calculate composite index
    stress = (temp_norm * rain_norm) / ndvi_norm
    return np.clip(stress, 0, 10)

def predict_disease_risk(ndvi, humidity, leaf_wetness, temperature):
    """Comprehensive disease risk prediction"""
    # Base risk factors
    risk_score = (1 - ndvi) * (humidity/100) * (leaf_wetness/10)
    
    # Temperature modifier
    if 22 <= temperature <= 28:
        risk_score *= 1.5  # Ideal temp for diseases
    elif temperature > 30:
        risk_score *= 0.7  # Too hot for many diseases
    
    # Classify risk
    if risk_score < 0.15:
        return "Low risk"
    elif risk_score < 0.3:
        return "Moderate risk"
    else:
        return "High risk"

def calculate_profit(yield_value, fertilizer_cost, labor_cost=150, crop_price=180):
    """Profit calculation with validation"""
    revenue = yield_value * crop_price
    total_cost = fertilizer_cost + labor_cost
    profit = revenue - total_cost
    return max(profit, -total_cost)  # Prevent negative beyond costs

def calculate_roi(yield_value, fertilizer_cost, crop_price=180):
    """ROI calculation with error handling"""
    if fertilizer_cost <= 0:
        return 0.0
    revenue = yield_value * crop_price
    return max((revenue - fertilizer_cost) / fertilizer_cost * 100, -100)

def analyze_trend(dates, values):
    """Trend analysis with validation"""
    if len(dates) < 2:
        return 0.0
    try:
        date_ordinals = [d.toordinal() for d in dates]
        slope = np.polyfit(date_ordinals, values, 1)[0]
        return slope * 365  # Convert to yearly change
    except:
        return 0.0