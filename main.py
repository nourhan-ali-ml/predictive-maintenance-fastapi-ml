"""
Predictive Maintenance API for Medical Equipment
Version: 2.0
Author: Nourhan Ali
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Literal

# ============================================
# LOAD MODEL AND FEATURES
# ============================================

try:
    model = joblib.load("predictive_maintenance_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    print("✅ Model and features loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Features: {feature_names}")
except FileNotFoundError as e:
    print(f"❌ Error: Model files not found - {e}")
    print("Run the Jupyter notebook first to generate model files!")
    model = None
    feature_names = None

# ============================================
# INITIALIZE FASTAPI APP
# ============================================

app = FastAPI(
    title="Predictive Maintenance API",
    description="AI-powered system that predicts maintenance needs for medical equipment based on real-time sensor data",
    version="2.0",
    contact={
        "name": "Nourhan Ali",
        "email": "bio.eng.nourhanali@gmail.com",
        "url": "https://linkedin.com/in/nourhan-ali-71289415b"
    }
)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class MaintenanceRequest(BaseModel):
    """Input schema for maintenance prediction"""
    temperature: float = Field(
        ..., 
        description="Equipment operating temperature in Celsius", 
        ge=50, 
        le=100,
        examples=[78.5]
    )
    vibration: float = Field(
        ..., 
        description="Mechanical vibration level (normalized)", 
        ge=0, 
        le=1.5,
        examples=[0.72]
    )
    operating_hours: int = Field(
        ..., 
        description="Cumulative operating hours since last maintenance", 
        ge=0, 
        le=2000,
        examples=[850]
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "temperature": 78.5,
                    "vibration": 0.72,
                    "operating_hours": 850
                }
            ]
        }
    }


class MaintenanceResponse(BaseModel):
    """Output schema for maintenance prediction"""
    maintenance_needed: Literal[0, 1] = Field(
        ..., 
        description="Prediction: 0 = No maintenance needed, 1 = Maintenance required"
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        ..., 
        description="Calculated risk category based on sensor thresholds"
    )
    message: str = Field(
        ..., 
        description="Human-readable recommendation for maintenance team"
    )
    confidence: float = Field(
        ...,
        description="Model prediction confidence (0-1)",
        ge=0,
        le=1
    )


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["Health"])
def root():
    """
    Root endpoint - API health check
    """
    if model is None:
        return {
            "status": "error",
            "message": "Model not loaded. Please ensure model files exist.",
            "version": "2.0"
        }
    
    return {
        "status": "active",
        "message": "Predictive Maintenance API is running",
        "model": type(model).__name__,
        "version": "2.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Detailed health check with model status
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features": feature_names if feature_names else None,
        "api_version": "2.0"
    }


@app.post("/predict", response_model=MaintenanceResponse, tags=["Prediction"])
def predict_maintenance(data: MaintenanceRequest):
    """
    Predict maintenance needs based on equipment sensor data
    
    This endpoint analyzes real-time sensor readings (temperature, vibration, operating hours)
    and predicts whether medical equipment requires maintenance.
    
    **Returns:**
    - `maintenance_needed`: 0 (no action) or 1 (schedule maintenance)
    - `risk_level`: low, medium, or high based on sensor thresholds
    - `message`: Actionable recommendation for maintenance team
    - `confidence`: Model prediction probability
    
    **Example Request:**
    ```json
    {
        "temperature": 78.5,
        "vibration": 0.72,
        "operating_hours": 850
    }
    ```
    """
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        # Create DataFrame with proper feature names
        input_df = pd.DataFrame([[
            data.temperature,
            data.vibration,
            data.operating_hours
        ]], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get prediction probability for confidence score
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            confidence = float(max(proba))
        else:
            confidence = 0.85  # Default confidence if probability not available
        
        # Calculate risk level based on thresholds
        risk_score = 0
        
        if data.temperature > 75:
            risk_score += 2
        elif data.temperature > 72:
            risk_score += 1
        
        if data.vibration > 0.65:
            risk_score += 2
        elif data.vibration > 0.55:
            risk_score += 1
        
        if data.operating_hours > 800:
            risk_score += 2
        elif data.operating_hours > 600:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 4:
            risk_level = "high"
        elif risk_score >= 2:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Generate actionable message
        if prediction == 1:
            if risk_level == "high":
                message = "⚠️ URGENT: High risk detected. Schedule immediate inspection and maintenance."
            elif risk_level == "medium":
                message = "⚠️ Maintenance recommended. Risk level moderate. Schedule inspection within 48 hours."
            else:
                message = "⚠️ Maintenance needed. Risk currently low but monitoring required."
        else:
            message = f"✅ Equipment operating normally. Risk level: {risk_level}. Continue regular monitoring."
        
        return {
            "maintenance_needed": int(prediction),
            "risk_level": risk_level,
            "message": message,
            "confidence": round(confidence, 3)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )


# ============================================
# STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    print("\n" + "="*70)
    print("🚀 Predictive Maintenance API Started")
    print("="*70)
    if model:
        print(f"✅ Model: {type(model).__name__}")
        print(f"✅ Features: {feature_names}")
    else:
        print("⚠️ Warning: Model files not found!")
    print(f"📚 Interactive docs: http://localhost:8000/docs")
    print("="*70 + "\n")


# ============================================
# EXAMPLE USAGE (for testing)
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
