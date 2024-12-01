# Sales Prediction Using LSTM: Monthly Pharmaceutical Forecasting

## Overview

This project predicts monthly pharmaceutical sales for Innovix in Elbonie using an LSTM-based neural network. The model uses historical sales data and other relevant features to forecast future sales volumes. The solution ensures effective planning and resource allocation, helping to prevent medication shortages.

---

## Features
1. **Data Preprocessing**: Extracts relevant features from a dataset and converts them into tensors suitable for model input.
2. **LSTM Model**: A neural network trained to predict sales trends based on past performance.
3. **Reproducibility**: Ensures consistent results with fixed seeds for randomness.
4. **Visualization**: Plots the predicted sales for easier analysis and interpretation.

---

## How It Works

1. **Dataset**:
   - Input: Excel file (`IN_E_FULL.xlsx`) containing historical sales data and related features.
   - Features used: Sales data (`Value Innovix`, `Value Yrex`) and indications (e.g., `Indication 1`, `Indication 2`).

2. **Model**:
   - LSTM (Long Short-Term Memory) is employed for sequence prediction, as it excels in handling time-series data.

3. **Prediction**:
   - The model predicts the next 14 months of sales based on the latest 20 months of data.
   - The forecast is denormalized to return to the original scale.

4. **Visualization**:
   - Monthly sales trends are plotted to compare predicted values over time.

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required Libraries: `pandas`, `torch`, `numpy`, `matplotlib`
- Dataset: Ensure `IN_E_FULL.xlsx` is in the `dataset2` folder.
- Pre-trained Model: Ensure `model_weights.pth` is in the `project` folder.

### Steps
1. Install dependencies:
   ```bash
   pip install pandas torch numpy matplotlib
   ```
2. Run the script:
   ```bash
   python predict_sales.py
   ```

---

## Code Breakdown

### Data Preprocessing
- Reads the Excel file and extracts the specified columns.
- Converts the data into PyTorch tensors for model compatibility.
- Prepares the last `context_size` months of data for prediction.

### Model Architecture
- **Input size**: 20 features per time step.
- **Hidden size**: 32 units in the LSTM layer.
- **Output size**: Predicts sales for the next 14 months.

### Prediction Workflow
1. Normalizes input data using the training mean and standard deviation.
2. Feeds the prepared data into the LSTM model.
3. Denormalizes the output to produce human-readable sales predictions.

### Visualization
- Combines historical and predicted sales data.
- Displays a line graph of monthly sales volumes.

---

## Example Output

- **Predicted Tensor Shape**: `(1, 20, 21)`
- **Predicted Sales (Next 14 Months)**: 
  ```plaintext
  [9507680, 8649920, 9342280, ...]
  ```
- **Sales Plot**:
  A graph showing trends over the months, with predictions extending beyond historical data.

---

## Future Work
- Enhance the model by adding external factors like seasonality or economic indicators.
- Implement a user-friendly interface for easy deployment in real-world scenarios.

