# Predictive Maintenance API

A **production-style Machine Learning project** that demonstrates how to train a model for **predictive maintenance** and expose it via a **FastAPI** endpoint.

The project is designed to bridge the gap between **Jupyter/Colab notebooks** and **real-world ML APIs**, following best practices used in junior-to-mid ML engineering roles.

---

## 📌 Project Overview

This project predicts whether **maintenance is needed** for industrial equipment based on:

* Temperature
* Vibration level
* Operating hours

A **RandomForestClassifier** is trained on simulated sensor data and served through a FastAPI-based API.

---

## 🧠 Machine Learning Model

* Algorithm: Random Forest Classifier
* Library: scikit-learn
* Training Data: Simulated sensor readings
* Output:

  * `0` → No maintenance needed
  * `1` → Maintenance needed

The trained model is saved as:

```
rf_model.pkl
```

---

## 🚀 API Endpoints

### `POST /predict`

Predicts whether maintenance is required.

**Request Body (JSON):**

```json
{
  "temperature": 72,
  "vibration": 0.55,
  "operating_hours": 500
}
```

**Response:**

```json
{
  "maintenance_needed": 0
}
```

---

## 🧪 Running in Google Colab

This project was designed to run **fully inside Google Colab** without external servers or tunneling tools.

To test the API inside Colab:

* The FastAPI app is defined normally
* Requests are tested using FastAPI's `TestClient`
* No Uvicorn or ngrok required

This avoids common asyncio and Python 3.12 issues in notebook environments.

---

## 📂 Project Structure

```
project/
│
├── train.py              # Model training script
├── rf_model.pkl          # Saved trained model
├── api_colab.py          # FastAPI app (Colab-compatible)
├── requirements.txt      # Dependencies
└── README.md
```

---

## 🛠 Tech Stack

* Python 3.12
* FastAPI
* scikit-learn
* pandas
* NumPy
* Joblib

---

## 🎯 Why This Project Matters

This project demonstrates:

* End-to-end ML workflow (training → saving → inference)
* Clean API design using FastAPI
* Understanding of production constraints in notebook environments
* Practical ML engineering skills beyond notebooks

It is suitable for:

* Junior ML Engineer roles
* AI Engineer internships
* ML portfolio / GitHub showcase

---

## 👤 Author

**Nourhan Ali**
Biomedical Engineer | ML & AI Enthusiast

---

## 📌 Future Improvements

* Replace simulated data with real sensor datasets
* Add input validation & logging
* Dockerize the API
* Deploy on cloud (AWS / GCP / Azure)

---

⭐ If you found this project useful, feel free to star the repository!
