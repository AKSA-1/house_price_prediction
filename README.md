# Stock Price Prediction Using Hybrid GRU + LSTM Model

## Project Overview
This project implements a hybrid **GRU** and **LSTM** neural network model to predict stock prices based on historical data. The dataset consists of daily stock prices, and the model predicts future prices using a combination of the **GRU layer** (for fast feature extraction) and the **LSTM layer** (for sequence learning). This hybrid approach helps improve prediction accuracy while maintaining faster execution times.

### Features
- **Hybrid Model:** Combines GRU for faster feature extraction and LSTM for better sequence learning.
- **Batch Normalization and Dropout:** Applied to improve model performance and reduce overfitting.
- **Early Stopping & Learning Rate Scheduling:** Implemented to prevent overfitting and optimize training.
- **Future Prediction:** The model can predict future stock prices based on the last known sequence of data.

---

## Installation and Requirements

To run this project, ensure you have the following libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`

You can install all dependencies using the following `pip` command:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

Alternatively, install the dependencies from `requirements.txt` if provided:

```bash
pip install -r requirements.txt
```

---

## Dataset

The model uses a CSV file (`infolimpioavanzadoTarget.csv`) containing stock price data. The dataset should include the following columns:
- **Date:** Dates for each stock record.
- **Close Price:** Closing price of the stock, which will be used as the target for predictions.

Ensure your dataset is formatted correctly. Here's a sample structure:

| Date       | Close  |
|------------|--------|
| 2021-01-01 | 150.75 |
| 2021-01-02 | 151.30 |

---

## Code Explanation

### Data Preprocessing

- **Loading the Dataset:** The stock data is loaded from the CSV file and sorted by date.
- **Normalization:** The `close` prices are normalized using `MinMaxScaler` to a range between 0 and 1 for faster convergence in neural networks.
- **Sequence Generation:** Time-series data is converted into sequences using the `create_sequences()` function. These sequences are used as input (X), and the next stock price in the sequence is the label (y).

### Model Architecture

The hybrid GRU + LSTM model consists of:
- **GRU (Gated Recurrent Unit):** Extracts faster features from the sequences.
- **LSTM (Long Short-Term Memory):** Used for learning long-term dependencies in the sequence data.
- **Batch Normalization:** Normalizes the outputs from each layer to speed up convergence.
- **Dropout:** Adds regularization to reduce overfitting.

The model is compiled using the **Adam optimizer** and **mean squared error (MSE)** as the loss function. **Early stopping** is used to halt training when the model stops improving.

---

## Setup

1. **Clone the Repository:**

```bash
git clone https://github.com/your-repo/stock-price-prediction-gru-lstm.git
cd stock-price-prediction-gru-lstm
```

2. **Prepare the Data:**
   - Place your dataset (`infolimpioavanzadoTarget.csv`) in the same directory or update the code to point to the correct path.

3. **Run the Script:**

```bash
python stock_price_prediction.py
```

4. **Modify Parameters (Optional):**
   - You can adjust hyperparameters like `sequence_length`, `epochs`, and `batch_size` in the code to experiment with different configurations.

---

## Example Usage

### Training the Model

The following example demonstrates how to train the hybrid GRU + LSTM model:

```python
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr])
```

### Predicting Future Stock Prices

To predict future prices based on the last known data points, use the `predict_future_prices()` function:

```python
# Predict future stock prices
future_prices = predict_future_prices(model, scaled_data, num_predictions=10)
print("Future Predictions:", future_prices.flatten())
```

### Visualizing Predictions

You can plot the actual vs predicted stock prices to visualize model performance:

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(data['date'].iloc[train_size + seq_length:], actual_prices, color='blue', label='Actual Stock Price')
plt.plot(data['date'].iloc[train_size + seq_length:], predicted_prices, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```

---

## Evaluation

After training, the model is evaluated on the test data using various metrics:
- **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions.
- **Root Mean Squared Error (RMSE):** Sensitive to large prediction errors.
- **R-squared (R²):** Indicates the proportion of variance in stock prices explained by the model.
- **Accuracy:** Calculated as `100 - Mean Absolute Error (%)` to give an intuitive sense of the model's performance.

---

## Future Work

- **Hyperparameter Optimization:** Experiment with different sequence lengths, hidden units, and layers to further optimize performance.
- **Additional Features:** Use other features like trading volume, opening prices, and high/low prices to improve model accuracy.
- **Advanced Architectures:** Explore more advanced neural network architectures like Transformers or Attention mechanisms to enhance prediction capabilities.

---

## Author

- **Team:** AI Avengers
- **Name:** Aditya Kumar Singh
- **Email:** 22bec008@smvdu.ac.in
- **GitHub:** [Aditya Kumar Singh](https://github.com/adityakumarsingh)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

Special thanks to the open-source community for providing excellent resources and support for machine learning projects. Additionally, gratitude to all those who contributed to the libraries used in this project, such as TensorFlow, NumPy, Pandas, and Matplotlib.

---

This enhanced README file provides a complete guide for anyone looking to understand, install, and run the stock price prediction project with the hybrid GRU + LSTM model.
