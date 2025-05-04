

# 🌦️ GRU-VAE for Weather Time Series Anomaly Detection & Forecasting

This notebook implements a **robust, modular, and scalable pipeline** for detecting anomalies and forecasting in weather time series data using:

* 🔍 **Variational Autoencoder (VAE)** for anomaly detection
* 🔁 **Sequence-to-Sequence GRU** for multi-step forecasting

---

## 🧱 1. Environment Setup

* 🔄 Resets the Python environment and clears cache for a clean slate.
* 📦 Loads essential libraries like `numpy`, `pandas`, `torch`, `matplotlib`, and `scikit-learn`.

---

## 📥 2. Data Loading & Preprocessing

* 📁 Loads a large-scale **Brazilian weather dataset** from Google Drive using `gdown`.
* 🔄 **Samples 0.1%** of the dataset for efficient experimentation.
* 🔢 Converts time columns, sorts chronologically, and sets datetime index.
* ⚙️ Automatically selects CPU/GPU and sets seeds for reproducibility.

---

## ⚙️ 3. Data Processing Pipeline

### 🔧 Key Functions

| Function               | Purpose                                                              |
| ---------------------- | -------------------------------------------------------------------- |
| `load_data`            | Loads and parses datetime; sorts and indexes chronologically         |
| `get_column_mapping`   | Translates Portuguese column names to English                        |
| `perform_eda`          | Provides summary stats, outlier detection, and visualizations        |
| `preprocess_data`      | Fills missing values, scales features, encodes categorical variables |
| `create_sequences`     | Creates past→future sequences for time series models                 |
| `split_data`           | Splits dataset into train/val/test chronologically                   |
| `engineer_features`    | Adds cyclic time features, weather indexes, and rolling stats        |
| `reduce_dataset_size`  | Shrinks data via chunking/striding/random sampling                   |
| `TimeSeriesDataset`    | Custom PyTorch dataset for time-series windowed data                 |
| `process_weather_data` | Runs the full processing pipeline end-to-end                         |

---

## 🧠 4. Variational Autoencoder (VAE) – Anomaly Detection

### 🏗️ Model Architecture

* **Encoder**: MLP → latent mean & log-variance
* **Decoder**: MLP reconstructs input from latent vector
* **Latent Sampling**: Uses reparameterization trick
* **Loss**: Combines 🔄 MSE (reconstruction) and 📉 KL divergence

### 🏃 Training & Evaluation

* Early stopping on validation loss
* Detects anomalies where reconstruction error > (mean + 2×std)

#### 🔍 **Step 3: Detecting anomalies in test data**

| **Metric**                  | **Value**         |
| --------------------------- | ----------------- |
| Anomaly Score (Mean)        | 7.13              |
| Anomaly Detection Threshold | 152.63            |
| Anomalies Detected          | 78 / 1149 samples |
| Anomaly Detection Rate      | \~6.79%           |


### 📊 Visualizations

* Anomaly score histograms
* Time series plots with anomaly points
* PCA-reduced latent space
* 🔬 Synthetic data from latent space

---

## 🔁 5. Sequence-to-Sequence GRU – Forecasting

### 🏗️ Model Architecture

| Component    | Description                                                     |
| ------------ | --------------------------------------------------------------- |
| `GRUModel`   | Simple GRU for single-step prediction                           |
| `Seq2SeqGRU` | Encoder-decoder GRU with optional teacher forcing for multistep |

### 🏃 Training Log

```plaintext
Step 5: Training GRU model for forecasting...
Epoch 1/30: Train Loss: 0.8246, Val Loss: 0.8187, Teacher Forcing: 0.80  
Epoch 2/30: Train Loss: 0.8057, Val Loss: 0.8185, Teacher Forcing: 0.79  
Epoch 3/30: Train Loss: 0.7997, Val Loss: 0.8183, Teacher Forcing: 0.78  
Epoch 4/30: Train Loss: 0.7901, Val Loss: 0.8177, Teacher Forcing: 0.77  
...
Early stopping at epoch 11
```

### 📈 Evaluation

```plaintext
Forecasting RMSE: 1.3610  
Forecasting MAE: 0.7273  
```

---

## 🔄 6. Integrated Pipeline

✅ Full end-to-end flow:

1. **Train VAE**
2. **Detect anomalies**
3. **Filter training anomalies**
4. **Train GRU on clean data**
5. **Forecast and evaluate**

---

## 🖼️ 7. Visualizations

| Type               | Purpose                                      |
| ------------------ | -------------------------------------------- |
| 📉 Loss Curves     | Training vs. Validation Loss                 |
| 🟢 Forecast Plots  | Actual vs. Predicted Sequences               |
| 🔴 Anomaly Overlay | Anomaly markers over original time series    |
| 🔵 Latent Space    | PCA projection of VAE latent representations |

---

## 📊 Summary Table of Main Components

| Component              | Role in Pipeline                                          |
| ---------------------- | --------------------------------------------------------- |
| 🔗 `VAE`               | Detects anomalies via reconstruction errors               |
| 🔁 `Seq2SeqGRU`        | Learns temporal patterns and forecasts future steps       |
| 🧰 Feature Engineering | Enhances input data with cyclic & derived weather metrics |
| 📊 EDA & Visuals       | Provides data understanding and interpretability          |
| 🔄 `combined_pipeline` | Orchestrates anomaly detection → filtering → forecasting  |

---

## ✅ Final Results Snapshot

| Metric                     | Value      |
| -------------------------- | ---------- |
| Anomalies Detected         | 78 / 1149  |
| Forecasting RMSE           | **1.3610** |
| Forecasting MAE            | **0.7273** |
| VAE Threshold (mean+2×std) | **152.63** |

---

## 💬 Conclusion

This notebook showcases a **deep learning-powered modular workflow** for **real-world weather time series**:

* 🎯 **Accurate anomaly detection** with a VAE
* 🔮 **Multi-step forecasting** using GRU-based sequence modeling
* 📊 Extensive EDA, engineering, and visualization
* ⚙️ Built for **scalability, interpretability, and reproducibility**

---

