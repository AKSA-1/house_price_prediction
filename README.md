# Stock Price Prediction Using Hybrid GRU + LSTM Model

## Project Overview
This project implements a hybrid **GRU** and **LSTM** neural network model to predict stock prices based on historical data. The dataset consists of daily stock prices, and the model predicts future prices using a combination of the GRU layer (for fast feature extraction) and the LSTM layer (for sequence learning). This hybrid approach helps improve prediction accuracy while maintaining faster execution times.

### Features
- **Hybrid Model:** Combines GRU for faster feature extraction and LSTM for better sequence learning.
- **Batch Normalization and Dropout:** Used to improve model performance and reduce overfitting.
- **Early Stopping & Learning Rate Scheduling:** To prevent overfitting and optimize training.
- **Future Prediction:** Model can predict future stock prices based on the last known sequence of data.

---

## Installation and Requirements
To run the code, the following libraries are required:
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`

You can install the required libraries using `pip`:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

---

## Dataset
The model uses a CSV file (`infolimpioavanzadoTarget.csv`) containing stock prices. The dataset should include at least a **date** column and a **close** price column. The `close` price is used for predictions.

---

## Code Explanation

### Data Preprocessing
- **Loading Dataset:** The stock data is loaded and sorted by date.
- **Normalization:** The `close` prices are normalized using `MinMaxScaler` to bring them within a range of 0 to 1, which helps with faster convergence in neural networks.
- **Sequence Generation:** Time-series data is converted into sequences using the `create_sequences()` function. These sequences (X) are used as input, and the corresponding label (y) is the next stock price in the sequence.

### Model Architecture
The model combines:
- **GRU (Gated Recurrent Unit):** Used to extract faster features from the sequences.
- **LSTM (Long Short-Term Memory):** Used for sequence learning to predict future prices based on historical data.
- **Batch Normalization:** Normalizes the outputs after each layer, which helps speed up convergence.
- **Dropout:** Regularization technique to reduce overfitting.

The model is compiled using the **Adam optimizer** with a reduced learning rate and **mean squared error (MSE)** as the loss function. The model also includes **early stopping** to halt training when the model stops improving.

### Future Prediction
The function `predict_future_prices()` predicts future stock prices by taking the last sequence of data points and generating future values one at a time.

---

## Evaluation

After training, the model is evaluated on the test data using the following metrics:
- **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions.
- **Root Mean Squared Error (RMSE):** A more sensitive measure of prediction error.
- **R-squared (R²):** Measures the proportion of variance in the dependent variable that is predictable from the independent variables.
- **Accuracy:** Calculated as `100 - Mean Absolute Error (%)` to give an intuitive sense of model performance.

### Visualization
The actual stock prices are compared with the predicted stock prices on a graph to visualize the model's performance over time.

---

## Example Usage

```python
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

# Predict future stock prices
future_prices = predict_future_prices(model, scaled_data, num_predictions=10)
print("Future Predictions:", future_prices.flatten())

# Plot the results
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

## Future Work
- **Hyperparameter Optimization:** Experiment with different sequence lengths, model layers, and units to optimize performance.
- **Incorporating Additional Features:** Use more features like trading volume, open price, high/low prices to improve model accuracy.
- **Implement Other Architectures:** Experiment with other neural network architectures like Transformers or Attention mechanisms.

---

## License
This project is open-source and free to use under the MIT License.

