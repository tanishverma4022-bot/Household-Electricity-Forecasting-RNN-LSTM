# Household-Electricity-Forecasting-RNN-LSTM
Deep learning time-series forecasting — comparing SimpleRNN vs LSTM to predict next-hour household electricity consumption from 24-hour usage windows.
A deep learning capstone project that builds a short-horizon electricity demand forecaster — given the previous 24 hours of household power usage, the model predicts global active power for the next hour. Two recurrent architectures, SimpleRNN and LSTM, were built under identical conditions and compared to see which better captures temporal dependencies in real-world energy data.

📌 Problem Statement

Predicting short-term household electricity demand can help with smarter grid load balancing, cost optimization, and consumer energy-saving apps. This project tests whether a more complex gated architecture (LSTM) meaningfully outperforms a simpler recurrent baseline (SimpleRNN) for a 24-hour forecast horizon.

📊 Dataset

Individual Household Electric Power Consumption dataset — minute-level readings from a single household (~2.07M rows), including voltage, global active/reactive power, and sub-metering data.

🧹 Data Preparation
Filled missing values (marked '?') using forward-fill to preserve time-series continuity
Checked and handled duplicate rows
Merged separate Date and Time columns into a unified Datetime index
Resampled minute-level data into hourly averages → 34,589 hourly records
Selected Global_active_power as the sole forecasting target
Scaled target to [0, 1] using MinMaxScaler for stable RNN training

🔢 Sequence Construction
Sliding-window approach: T = 24 hours input → predict hour 25
Generated 34,565 sequence-target pairs
Chronological (non-shuffled) train/test split:
Training: 27,652 sequences (80%)
Test: 6,913 sequences (20%)

🏗️ Model Architectures
Architecture	Recurrent Layer	Trainable Params
Simple RNN	SimpleRNN(50, activation='tanh')	3,901
LSTM	LSTM(50, activation='tanh')	11,701

Both models share an identical shape: recurrent layer (50 units) → Dropout (0.2) → Dense (25, ReLU) → Dense (1, output), isolating the recurrent cell type as the only real variable between them.

⚙️ Training Configuration
Optimizer: Adam (lr = 0.001)
Loss: MSE (tracked with MAE)
Early Stopping: patience = 5, restores best weights
Epochs/Batch: up to 30 epochs, batch size 32, 10% validation split (order preserved)

📈 Results
Model	MAE (kW)	RMSE (kW)
Simple RNN	0.3739	0.5102
LSTM	0.3565	0.5068

The LSTM outperforms the SimpleRNN on both metrics (~4.7% lower MAE), consistent with its gating mechanism better retaining information across the 24-hour window and mitigating vanishing gradients. The margin is real but modest — showing SimpleRNN remains a competitive, cheaper baseline at this sequence length.

📉 Visualizations
Full-range line plot of hourly Global Active Power (trend/seasonality check)
Actual vs. Predicted overlay charts for both models on the first 300 test samples

🚀 Deployment Readiness
Trained LSTM model serialized to .keras format
Fitted MinMaxScaler pickled alongside it
Streamlit set up as the target framework for an interactive forecasting demo

🛠️ Tools & Libraries

Python · TensorFlow / Keras · Pandas · NumPy · Scikit-learn · Matplotlib · Streamlit

🧾 Conclusion

This project demonstrates a full end-to-end time-series forecasting pipeline — from cleaning raw sensor data to framing it as supervised sequence prediction and benchmarking two recurrent architectures. Results confirm LSTM's edge over SimpleRNN for this task, while showing the improvement is incremental rather than dramatic at a 24-hour forecast horizon.
