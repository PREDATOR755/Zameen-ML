# Zameen-ML
Automated Real Estate Prediction
# Zameen-ML-Pipeline: Automated Real Estate Price Prediction

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange.svg)

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline that predicts residential property prices in Islamabad. It moves beyond static Jupyter notebooks by implementing automated data ingestion, live geospatial feature engineering, and a decoupled inference web application.

---

## 🏗️ System Architecture

The pipeline is structured into four highly decoupled core components:

### 1. Automated Data Ingestion (GitHub Actions)
A custom, defensively programmed web scraper (`Scrapper.py`) navigates Zameen.com to extract property listings. 
* Uses rotating User-Agents, intelligent pagination delays, and Cloudflare-evasion techniques.
* Orchestrated via **GitHub Actions** (`automated_scraper.yml`) to run autonomously every week, ensuring the model's training data never goes stale.

### 2. Geospatial Feature Engineering
Instead of relying solely on text-based neighborhood target encoding, the pipeline dynamically calculates spatial relationships.
* Utilizes the `geopy` library and Nominatim API to resolve exact geographic coordinates (Latitude/Longitude) for unique property sectors.
* Calculates physical "as-the-crow-flies" Haversine distances to major economic hubs (e.g., Islamabad Blue Area) and transit lines to inject real-world valuation metrics into the feature matrix.

### 3. Predictive Modeling (Leakage-Free Training)
The core ML engine is built to handle the high right-skew inherent in real estate pricing.
* **Algorithm:** A highly regularized `RandomForestRegressor` ensemble.
* **Data Integrity:** Strict isolation of train/test partitions prior to Target Encoding prevents data leakage. Log transformations (`log1p`) are applied to both target variables and feature interaction cross-products to stabilize variance and prevent scale explosion.
* **Serialization:** The champion model and geographic mappings are serialized via `joblib` into lightweight artifacts.

### 4. Interactive Deployment (Streamlit)
A lightweight frontend application (`app.py`) decouples training from serving. It loads the pre-trained `.pkl` artifacts into cache to provide sub-second inference. The app matches the exact feature engineering pipeline used during training to ensure symmetry and prevents runtime crashes on missing input data.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Git
- (Optional) A virtual environment tool like `venv` or `conda`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Zameen-ML-Pipeline.git
   cd Zameen-ML-Pipeline
