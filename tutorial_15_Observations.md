# Tutorial 15 — Long Short-Term Memory (LSTM)

## Overview

This tutorial focused on **Long Short-Term Memory (LSTM)** networks. It demonstrates stock-price sequence prediction using historical `GOOGL` close prices, then extends the tutorial with a Simple RNN comparison and sentiment analysis.

## Notebook Structure

| Section | Description |
|---|---|
| Cell 1 | TensorFlow/Keras code copied from the tutorial PDF screenshots |
| Cell 2 | PyTorch implementation of the same stock-price LSTM workflow |
| Task 1 | Compare LSTM with Simple RNN |
| Task 2 | Build an LSTM sentiment-analysis model |

## Cell 1 — PDF Screenshot Code

Cell 1 downloads `GOOGL` stock data, creates 60-day sequences, trains a two-layer Keras LSTM model, plots actual vs predicted prices, and predicts the next day's price.

## Cell 2 — PyTorch Implementation

Cell 2 implements the same workflow in PyTorch using `nn.LSTM`, `nn.Linear`, MSE loss, and Adam optimizer.

The PyTorch workflow is:

`past 60 close prices → LSTM → predicted next close price`

## Guardrails

* stock data is split chronologically, not randomly
* the PyTorch scaler is fitted only on the training split
* test data is not used for training
* RMSE and MAE are reported on the test split
* sentiment-analysis train and test splits are kept separate
* stock prediction is treated as a learning exercise, not financial advice

## Task 1 — Compare LSTM with Simple RNN

Task 1 trains a Simple RNN on the same stock-price sequences used by the LSTM. Both models are compared using RMSE, MAE, and test MSE loss.

## Task 2 — LSTM for Sentiment Analysis

Task 2 builds a PyTorch LSTM classifier for sentiment analysis using the IMDb review dataset. The dataset is loaded through Keras for convenience, but the model and training loop are PyTorch.

The output is binary sentiment:

* `0` = negative
* `1` = positive

## Key Takeaways

* LSTMs can model sequential dependencies
* stock prices can be converted into supervised sequences using sliding windows
* Simple RNN and LSTM can be compared on the same time-series task
* LSTMs can also be applied to text sequences for sentiment analysis
* chronological splitting is important for time-series prediction
* test data should not be used when fitting scalers or training models

## Task Completion

| PDF Task | Status |
|---|---|
| Compare LSTM with Simple RNN | Completed |
| Make an LSTM model for sentiment analysis | Completed |

