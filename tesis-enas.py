import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, ConvLSTM2D, BatchNormalization, Input, TimeDistributed, Lambda
from keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, Adamax
from tensorflow.keras.regularizers import L1L2

data_list = pd.read_csv("max_values_per_TT_sumatra_fix_15.csv")
data_matrix = pd.read_csv('matrix_data_CNN_sumatra_fix_15.csv')
data_matrix = np.log(data_matrix)

# Define the 9 zone ranges
zone_ranges = {
    'zone1': {'lat': range(2, 6), 'long': range(95, 100)},
    'zone2': {'lat': range(2, 6), 'long': range(100, 105)},
    'zone3': {'lat': range(2, 6), 'long': range(105, 110)},
    'zone4': {'lat': range(-2, 2), 'long': range(95, 100)},
    'zone5': {'lat': range(-2, 2), 'long': range(100, 105)},
    'zone6': {'lat': range(-2, 2), 'long': range(105, 110)},
    'zone7': {'lat': range(-6, -2), 'long': range(95, 100)},
    'zone8': {'lat': range(-6, -2), 'long': range(100, 105)},
    'zone9': {'lat': range(-6, -2), 'long': range(105, 110)}
}

# Function to determine the zone based on latitude and longitude
def get_zone(lat, long):
    for zone, ranges in zone_ranges.items():
        if lat in ranges['lat'] and long in ranges['long']:
            return zone
    return None

# Process each row to find the maximum value's zone
for index, row in data_matrix.iterrows():
    grid_values = row[:-9]  # Exclude the last 9 zone columns
    max_val = grid_values.max()

    # Reset zone flags for this row
    for zone in zone_ranges.keys():
        data_matrix.at[index, zone] = 0

    # Check if all values are the same (no distinct maximum)
    if grid_values.nunique() == 1:
        continue

    # Find columns with the maximum value and set the zone flag
    for col, val in grid_values.items():
        if val == max_val:
            lat_long = col.replace('Grid', '').split('_')
            lat = int(lat_long[0])
            long = int(lat_long[1])
            zone = get_zone(lat, long)
            if zone:
                data_matrix.at[index, zone] = 1

# Preparing the LSTM dataset
def create_dataset(data, time_steps):
    X, y = [], []
    # Ensure that we do not go out of bounds
    for i in range(len(data) - time_steps):
        # Extract a sequence of 'time_steps' days
        X.append(data[i:(i + time_steps)])
        # Add the next day's intensity value as the target
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Define your time steps and split the data
time_steps = 20
X, y = create_dataset(data_list[["log_MaxValue"][0]], time_steps)

# Define the split point for the training and test sets
split_idx = int(len(X) * 0.75)  # 75% for training

# Sequentially split the data
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# CNN and Branch ConvLSTM
def create_sequences(features, targets, time_steps):
    X, y = [], []
    for i in range(len(features) - time_steps):
        # Reshape each day's data into a 42x15 matrix and then stack 30 of these matrices
        sequence = np.array([features[i+j].reshape(12, 15) for j in range(time_steps)])
        # Now, sequence is of shape (30, 42, 15), we need to transpose it to (42, 15, 30)
        X.append(sequence.transpose((1, 2, 0)))  # Reshape to (42, 15, 30)
        y.append(targets[i + time_steps])
    return np.array(X), np.array(y)

# Assuming 'data' is your DataFrame
features = data_matrix.iloc[:, :180].values
targets = data_matrix.iloc[:, 180:].values

# Create sequences of 30-day matrices
time_steps = 1
X, y = create_sequences(features, targets, time_steps)

# Define the split point for the training and test sets
split_idx = int(len(X) * 0.75)

# Sequentially split the data
train_features, test_features = X[:split_idx], X[split_idx:]
train_targets, test_targets = y[:split_idx], y[split_idx:]

# 10x Running Model LSTM
# Initialize lists to store the final loss values for each run
losses = []
val_losses = []
r2 = []
best_epochs = []

for i in range(10):
    print(f"Training run {i+1}/10")
    # Create a new instance of the model for each run to start fresh

    model = Sequential()
    #model.add(LSTM(100, activation='tanh', input_shape=(time_steps, 1), recurrent_initializer='random_normal'))
    model.add(LSTM(50, activation='tanh', input_shape=(time_steps, 1)))
    model.add(Dense(1))

    #optimizer = Nadam()
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) #callbacks=[es]

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), verbose=0, callbacks=[es])

    # Record the final loss and validation loss
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_epoch = np.argmin(history.history['val_loss']) + 1
    losses.append(final_train_loss)
    val_losses.append(final_val_loss)
    best_epochs.append(best_epoch)
    predictions_model = model.predict(X_test).flatten()
    r2_model = r2_score(y_test, predictions_model)
    r2.append(r2_model)

    # Print the results
    print(f"Training Loss: {final_train_loss}")
    print(f"Validation Loss: {final_val_loss}")
    print(f"Best Epoch: {best_epoch}")
    print(f"R^2: {r2_model}")

# Calculate the average and standard deviation of loss and val_loss
avg_loss = np.mean(losses)
std_loss = np.std(losses)
avg_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)
avg_r_2 = np.mean(r2)
std_r_2 = np.std(r2)
avg_best_epoch = np.mean(best_epochs)  # Average best epoch
std_best_epoch = np.std(best_epochs)  # Std deviation of best epochs

# Print the results
print(f"Average Training Loss: {avg_loss}, Standard Deviation: {std_loss}")
print(f"Average Validation Loss: {avg_val_loss}, Standard Deviation: {std_val_loss}")
print(f"Average R^2: {avg_r_2}, Standard Deviation: {std_r_2}")
print(f"Average Best Epoch: {avg_best_epoch}, Standard Deviation: {std_best_epoch}")

# 10x Running Model CNN
# Initialize lists to store the final loss values for each run
losses = []
val_losses = []
accs = []
best_epochs = []

for i in range(10):
    print(f"Training run {i+1}/10")
    # Create a new instance of the model for each run to start fresh
    model = Sequential([
      Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(12, 15, time_steps)),
      MaxPooling2D((2, 2)),
      Dropout(0.2),
      Conv2D(128, (3, 3), activation='relu', padding='same'),
      MaxPooling2D((2, 2)),
      Dropout(0.2),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.2),
      Dense(128, activation='relu'),
      Dropout(0.2),
      Dense(128, activation='relu'),
      Dropout(0.2),
      Dense(9, activation='softmax')
    ])

    optimizer = Nadam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True) #callbacks=[es]

    # Train the model
    history = model.fit(train_features, train_targets, epochs=100, batch_size=128, validation_data=(test_features, test_targets), verbose=0, callbacks=[es])

    # Record the final loss and validation loss
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_acc = history.history['val_accuracy'][-1]
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    losses.append(final_train_loss)
    val_losses.append(final_val_loss)
    accs.append(final_acc)
    best_epochs.append(best_epoch)

    # Print the results
    print(f"Training Loss: {final_train_loss}")
    print(f"Validation Loss: {final_val_loss}")
    print(f"Accuracy: {final_acc}")
    print(f"Best Epoch: {best_epoch}")

# Calculate the average and standard deviation of loss and val_loss
avg_loss = np.mean(losses)
std_loss = np.std(losses)
avg_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)
avg_acc = np.mean(accs)
std_acc = np.std(accs)
avg_best_epoch = np.mean(best_epochs)  # Average best epoch
std_best_epoch = np.std(best_epochs)  # Std deviation of best epochs

# Print the results
print(f"Average Training Loss: {avg_loss}, Standard Deviation: {std_loss}")
print(f"Average Validation Loss: {avg_val_loss}, Standard Deviation: {std_val_loss}")
print(f"Average Accuracy: {avg_acc}, Standard Deviation: {std_acc}")
print(f"Average Best Epoch: {avg_best_epoch}, Standard Deviation: {std_best_epoch}")
