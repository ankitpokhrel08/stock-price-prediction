import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

# Function to process the data and prepare for prediction
def preprocess_data(df):
    #Scroll bar to select company name and then assign that company name to the variable 'Company'
    company = st.selectbox("Select Company", df['Company'].unique())
    df = df[df['Company'] == company]


    # Use only the 'Close' column for predictions
    df = df[['Date', 'Close']]
    

    df['Date'] = df['Date'].apply(str_to_datetime)
    #sort the date by ascending order
    df = df.sort_values('Date')
    df.index = df.pop('Date')
    # Wrap data table in expander
    with st.expander("View Processed Data"):
        st.dataframe(df.head(10))
    
    st.subheader(f"Historical Stock Prices for {company}")
    st.line_chart(df['Close'])
    return df

def str_to_datetime(s):
    """
    Converts a string in the format 'YYYY-MM-DD HH:MM:SS-TZ' to a date object (YYYY-MM-DD).
    
    Args:
    - s (str): The input string containing the timestamp.
    
    Returns:
    - datetime.date: A date object with year, month, and day.
    """
    # Extract the date part (before the space)
    date_part = s.split(' ')[0]
    year, month, day = map(int, date_part.split('-'))
    return datetime.date(year, month, day)

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = pd.to_datetime(first_date_str)  # Convert to datetime
    last_date = pd.to_datetime(last_date_str)

    # Ensure the DataFrame has a DateTimeIndex
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        dataframe.index = pd.to_datetime(dataframe.index)

    target_date = first_date
    dates = []
    X, Y = [], []
    last_time = False

    while True:
        # Get the last (n+1) rows up to the target date
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            # Skip if not enough data
            print(f"Skipping target date {target_date}: insufficient data")
            if last_time:
                break
            next_target_date = target_date + datetime.timedelta(days=1)
            if next_target_date > last_date:
                break
            target_date = next_target_date
            continue

        # Extract features (X) and target (Y)
        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        # Find the next target date (7 days later)
        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        if len(next_week) < 2:
            break 

        next_target_date = next_week.index[1]  # Get the next valid date
        if last_time:
            break

        target_date = next_target_date

        if target_date >= last_date:
            last_time = True

    # Create a new DataFrame for the results
    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    # Add feature columns
    X = np.array(X)
    for i in range(n):
        ret_df[f'Target-{n - i}'] = X[:, i]

    # Add target column
    ret_df['Target'] = Y
    #convert the 'Target Date' from Timestamp object to string
    ret_df['Target Date'] = ret_df['Target Date'].apply(lambda x: x.strftime('%Y-%m-%d'))

    #Convert the 'Target Date' column to datetime
    ret_df['Target Date'] = ret_df['Target Date'].apply(str_to_datetime)
    # Wrap windowed data in expander
    with st.expander("View Windowed Data"):
        st.dataframe(ret_df.head(10))
    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    # Convert the dataframe to a NumPy array for easier slicing
    df_as_np = windowed_dataframe.to_numpy()

    # Extract the first column (dates) to keep track of time
    dates = df_as_np[:, 0]

    # Extract all columns except the first (dates) and last (target price)
    middle_matrix = df_as_np[:, 1:-1]

    # Reshape X to match LSTM input format: (samples, time steps, features)
    # Here, 'time steps' = window size (n) and 'features' = 1 (stock price)
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    # Extract the last column (Target price for prediction)
    Y = df_as_np[:, -1]

    # Convert X and Y to float32 for TensorFlow compatibility
    return dates, X.astype(np.float32), Y.astype(np.float32)

def splitting(dates, X, y):
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]

    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
    #Graph of train, validation and test data
    fig, ax = plt.subplots()
    ax.plot(dates_train, y_train, label='Train')
    ax.plot(dates_val, y_val, label='Validation')
    ax.plot(dates_test, y_test, label='Test')
    ax.legend()
    st.pyplot(fig)
    return dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test

def create_lstm_model():
    model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

    model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    st.subheader("Model Training")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get user prediction while model trains
    st.subheader("While the model is training...")
    user_prediction = st.radio(
        "Based on the historical data shown above, do you think the stock price will go up or down?",
        options=['Up ↑', 'Down ↓']
    )
    
    epochs = 50
    history = {'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Training Progress: {int(progress * 100)}%")
        
        history['loss'].extend(hist.history['loss'])
        history['val_loss'].extend(hist.history['val_loss'])
    
    st.success("Training completed!")
    
    # Show training metrics
    st.subheader("Training Metrics")
    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='Training Loss')
    ax.plot(history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)
    
    return model

def plottingeverything(model, X_train, y_train, X_val, y_val, X_test, y_test):
    train_predictions = model.predict(X_train).flatten()
    val_predictions = model.predict(X_val).flatten()
    test_predictions = model.predict(X_test).flatten()

    st.subheader("Model Predictions vs Actual Values")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates_train, train_predictions)
    ax.plot(dates_train, y_train)
    ax.plot(dates_val, val_predictions)
    ax.plot(dates_val, y_val)
    ax.plot(dates_test, test_predictions)
    ax.plot(dates_test, y_test)
    ax.legend(['Training Predictions', 'Training Observations', 'Validation Predictions', 'Validation Observations', 'Test Predictions', 'Test Observations'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Model Performance on Training, Validation, and Test Sets')
    st.pyplot(fig)

def futureprediction(df):
    future_prices = []
    last_3_days = df['Close'].iloc[-3:].to_numpy().reshape(1, 3, 1)

    # Predict the next 5 days (next week) stock prices
    for _ in range(5):
        next_day_price = model.predict(last_3_days).flatten()[0]
        future_prices.append(next_day_price)
    
    # Update last_3_days by adding new prediction and removing the oldest value
        last_3_days = np.append(last_3_days[:, 1:, :], [[[next_day_price]]], axis=1)

    return future_prices

def futuredates(df):

    last_date = df.index[-1]

# Create dates for the next week
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=5, freq='D')

    return future_dates


def final_prediction(df, future_dates, future_prices):
    st.subheader("Future Price Predictions")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-30:], df['Close'].iloc[-30:], label='Historical Prices (Last 30 Days)')
    ax.plot(future_dates, future_prices, label='Predicted Prices (Next 5 Days)', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Price Prediction for Next 5 Days')
    ax.legend()
    st.pyplot(fig)
    
    # Display predicted values in a table
    st.subheader("Predicted Prices for Next 5 Days")
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_prices
    })
    st.table(pred_df)

   

# Streamlit app UI
st.title('Stock Price Prediction App')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file with 'Date' and 'Close' columns", type='csv')

if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the uploaded data
    st.write("Data preview:", df.head())

    # Process the data
    df_processed = preprocess_data(df)
    #Ask user to select the date range for prediction
    st.subheader("Select Date Range for Prediction. Format: YYYY-MM-DD")
    #finding the latest date and storing in variable 'latest_date'
    latest_date = df_processed.index[-1]
    #finding the oneyear before date from the latest date
    one_year_before = latest_date - timedelta(days=365)
    start_date = st.date_input("Start Date", one_year_before)
    end_date = st.date_input("End Date", latest_date)

    windowed_df = df_to_windowed_df(df_processed, start_date,end_date, n=3)
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test = splitting(dates, X, y)
    model = create_lstm_model()
    model = train_model(model, X_train, y_train, X_val, y_val)  # Add this line
    plottingeverything(model, X_train, y_train, X_val, y_val, X_test, y_test)
    future_prices = futureprediction(df_processed)
    future_dates = futuredates(df_processed)
    final_prediction(df_processed, future_dates, future_prices)









