# 🏥 Predictive Maintenance System for Medical Equipment

A machine learning-powered system that predicts maintenance needs for medical devices based on real-time sensor data, helping healthcare facilities prevent equipment failures and ensure continuous patient care.

---

## 📊 Project Overview

This project demonstrates an end-to-end predictive maintenance solution combining:
- **Machine Learning Models** for failure prediction
- **Production-Ready API** built with FastAPI
- **Real-time Monitoring** of medical equipment health

### Key Features
✅ Analyzes sensor data (temperature, vibration, operating hours)  
✅ Predicts equipment at high risk of failure  
✅ Prioritizes recall to minimize missed failures (critical in healthcare)  
✅ RESTful API for easy integration into existing systems

---

## 🎯 Problem Statement

In healthcare environments, unexpected equipment failures can:
- Disrupt critical patient care
- Lead to costly emergency repairs
- Reduce operational efficiency

**Solution:** Proactive maintenance scheduling based on ML-driven risk assessment.

---

## 🧪 Model Performance

After comparing multiple algorithms, **Logistic Regression** achieved the best balance:

| Metric | Score | Why It Matters |
|--------|-------|----------------|
| **Recall** | 73% | Catches most devices needing maintenance |
| **Precision** | 47% | Some false alarms, but acceptable tradeoff |
| **F1-Score** | 57% | Balanced performance |
| **Accuracy** | 60% | Overall correctness |

### Why Prioritize Recall?
In healthcare, **missing a real failure (false negative) is far more costly** than generating a false alarm (false positive). High recall ensures we catch devices at risk.

---

## 🔬 Methodology

### 1. Data Generation
- Simulated sensor data for 500 medical devices
- Realistic failure patterns based on equipment thresholds:
  - High temperature (>75°C) → increased failure risk
  - High vibration (>0.65) → increased failure risk
  - High operating hours (>800h) → increased failure risk

### 2. Feature Engineering
- **Temperature**: Equipment operating temperature
- **Vibration**: Mechanical vibration levels
- **Operating Hours**: Cumulative usage time

### 3. Model Comparison
Tested three algorithms with optimized hyperparameters:
- ✅ **Logistic Regression** (Best - 57% F1-Score)
- Random Forest (52% F1-Score)
- XGBoost (51% F1-Score)

### 4. Production Deployment
- Built RESTful API with **FastAPI**
- Real-time prediction endpoint
- Easy integration with hospital management systems

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/nourhan-ali-ml/predictive-maintenance-fastapi-ml.git
cd predictive-maintenance-fastapi-ml
pip install -r requirements.txt
```

### Run the Notebook
Open `predictive_maintenance.ipynb` in Jupyter or Google Colab to see:
- Data generation and exploration
- Model training and comparison
- Performance evaluation and visualization

### Run the API (Local Development)
```bash
# Install dependencies first
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

The API will start at: `http://localhost:8000`

**Interactive Documentation:** Visit `http://localhost:8000/docs` for automatic API documentation powered by FastAPI.

### Test the API

#### Using cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 78.5,
    "vibration": 0.72,
    "operating_hours": 850
  }'
```

#### Using Python:
```python
import requests

data = {
    "temperature": 78.5,
    "vibration": 0.72,
    "operating_hours": 850
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

**Example Response:**
```json
{
  "maintenance_needed": 1,
  "risk_level": "high",
  "message": "⚠️ URGENT: High risk detected. Schedule immediate inspection and maintenance.",
  "confidence": 0.847
}
```

---

## 📁 Project Structure

```
predictive-maintenance-fastapi-ml/
├── predictive_maintenance.ipynb   # Model development & experiments
├── main.py                        # FastAPI application
├── model.pkl                      # Trained model (saved)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🔍 Key Insights

### Feature Importance
1. **Operating Hours** (44.7%) - Most predictive feature
2. **Vibration** (29.7%) - Strong indicator of mechanical issues
3. **Temperature** (25.6%) - Reflects thermal stress

### Model Behavior
- **High Recall Strategy**: Designed to catch most at-risk devices
- **Acceptable False Positives**: Maintenance teams prefer investigating false alarms over missing real failures
- **Class Imbalance Handling**: Used class weighting and balanced sampling

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **scikit-learn** - ML models and evaluation
- **XGBoost** - Gradient boosting
- **FastAPI** - Production API framework
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization

---

## 📈 Future Enhancements

- [ ] Time-series forecasting (predict failures X days in advance)
- [ ] Integration with real hospital equipment data
- [ ] Dashboard for maintenance team monitoring
- [ ] Automated alert system (email/SMS notifications)
- [ ] A/B testing with different risk thresholds
- [ ] Model retraining pipeline with new data

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

---

## 📧 Contact

**Nourhan Ali**  
AI & Machine Learning Engineer  
📧 bio.eng.nourhanali@gmail.com  
💼 [LinkedIn](http://linkedin.com/in/nourhan-ali-71289415b)  
🔗 [GitHub Portfolio](https://github.com/nourhan-ali-ml/ml-portfolio.git)

---

## 📜 License

This project is open source and available under the MIT License.

---

## ⭐ Acknowledgments

Built as part of my transition into AI/ML engineering, applying machine learning to real-world healthcare challenges.

If you found this project helpful, please consider giving it a ⭐!

---

**Made for Healthcare Innovation**
