# Bengaluru-House-Prediction

# 🏠 Bengaluru House Price Prediction

A machine learning project that predicts the price of a house in Bengaluru, India, based on its **location**, **total area (sq. ft.)**, **number of bathrooms**, and **number of bedrooms (BHK)**.

The model is trained on the Bengaluru House Price dataset using **Linear Regression** (scikit-learn), after extensive data cleaning, feature engineering and outlier removal.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📖 Project Overview

Buying a home is one of the biggest financial decisions people make, and real estate prices in a city like Bengaluru vary heavily by locality, size and configuration. This project builds a regression model that estimates a fair price for a property, helping buyers and sellers get a quick data-driven estimate.

**Inputs**

| Feature | Description |
|---------|-------------|
| `location` | Locality in Bengaluru (e.g. Whitefield, Indira Nagar, JP Nagar) |
| `total_sqft` | Total area of the property in square feet |
| `bath` | Number of bathrooms |
| `bhk` | Number of bedrooms |

**Output:** Estimated price in **Indian Lakhs (₹ 1 Lakh = ₹ 100,000)**.

---

## 📊 Dataset

- **File:** `Bengaluru_House_Data.csv`
- **Size:** 13,320 rows × 9 columns (raw)
- **Columns:** `area_type`, `availability`, `location`, `size`, `society`, `total_sqft`, `bath`, `balcony`, `price`

After cleaning and outlier removal, **7,251 rows** remain for model training.

---

## 🛠 Tech Stack

- **Language:** Python 3
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn
- **Environment:** Jupyter Notebook
- **Web framework (planned / optional):** Flask

---

## 📂 Project Structure

```
Bengaluru-House-Prediction/
│
├── Bengaluru_House_Data.csv            # Raw dataset
├── model.ipynb                         # Data cleaning, EDA, model building & evaluation
├── bangulur_house_prices_model.pkl     # Trained Linear Regression model (pickle)
├── columns.json                        # Feature columns used by the model (for predictions)
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔬 Methodology

The full workflow is in [`model.ipynb`](model.ipynb).

### 1. Data Cleaning
- Dropped less useful columns: `area_type`, `society`, `balcony`, `availability`
- Removed rows with missing values (13,320 → 13,246 rows)

### 2. Feature Engineering
- Extracted a numeric **`bhk`** feature from the `size` column (e.g. "2 BHK", "4 Bedroom" → `2`, `4`)
- Converted `total_sqft` to numbers, handling ranges (e.g. `2100 - 2850` → average) and dropping unconvertible values (e.g. "34.46Sq. Meter")
- Created **`price_per_sqft`** to help detect outliers
- **Dimensionality reduction:** locations with 10 or fewer listings were grouped into an `other` category

### 3. Outlier Removal
- Removed properties with less than **300 sq. ft. per bedroom**
- Removed price-per-sqft outliers per location (beyond ±1 standard deviation)
- Removed cases where, in the same location, a larger-BHK flat was priced lower per sq. ft. than a smaller-BHK flat
- Removed unrealistic bathroom counts (bathrooms ≥ BHK + 2)

Rows after each step: `13,246 → 12,502 → 10,241 → 7,329 → 7,251`

### 4. Encoding
- Applied **one-hot encoding** to the `location` column (final feature matrix: 7,251 × 245 including the target)

### 5. Model Building & Selection
- 80/20 train–test split
- Compared **Linear Regression**, **Lasso Regression** and **Decision Tree Regressor** using `GridSearchCV` with `ShuffleSplit` cross-validation
- Selected **Linear Regression** as the best performer and exported it with `pickle`

---

## 📈 Model Performance

| Model | Best Cross-Validation R² Score |
|-------|-------------------------------|
| **Linear Regression** ✅ | **0.818** |
| Decision Tree Regressor | 0.720 |
| Lasso Regression | 0.687 |

- **Test-set R² (Linear Regression):** ≈ **0.845**
- **5-fold cross-validation R²:** 0.77 – 0.85

---

## ⚙️ Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/Bengaluru-House-Prediction.git
cd Bengaluru-House-Prediction
```

**2. Create and activate a virtual environment** (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch Jupyter Notebook**

```bash
jupyter notebook
```

Open `model.ipynb` and run all cells to reproduce the analysis and model.

---

## 🚀 Usage

### Predict using the saved model

```python
import json
import pickle
import numpy as np

# Load model and feature columns
with open("bangulur_house_prices_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]


def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    location = location.lower()
    if location in data_columns:
        x[data_columns.index(location)] = 1

    return round(model.predict([x])[0], 2)


print(predict_price("1st Phase JP Nagar", 1000, 2, 2), "Lakhs")
print(predict_price("Whitefield", 1500, 3, 3), "Lakhs")
```

> **Note:** Location names in `columns.json` are lowercase. If a location isn't found, the model treats it as `other` (all location flags stay `0`).
> You may see a scikit-learn warning about feature names — it is harmless here.

### Example from the notebook

| Location | Sq. Ft. | Bath | BHK | Predicted Price |
|----------|---------|------|-----|-----------------|
| 1st Phase JP Nagar | 1000 | 1 | 2 | ~₹ 78.4 Lakhs |
| 1st Phase JP Nagar | 1000 | 2 | 3 | ~₹ 81.7 Lakhs |

---

## 🔮 Future Improvements

- Build a **Flask** web app / REST API so users can enter details and get predictions in the browser
- Add a front-end (HTML/CSS/JS) with a location dropdown
- Try more advanced models (Random Forest, XGBoost, Gradient Boosting) and tune hyperparameters
- Add more features (amenities, age of property, proximity to metro/IT hubs)
- Deploy the app on Render / Heroku / AWS

---

## 👤 Author

**Amey**

- GitHub: ByteAmey

---

⭐ If you found this project useful, please consider giving it a star!
