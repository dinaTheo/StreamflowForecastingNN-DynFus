import time as pytime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.layers import  LSTM, Dense, Input, Concatenate, MultiHeadAttention, RepeatVector, TimeDistributed, Multiply
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
from hydroeval import evaluator, nse, kge


folder_with_data = 'Data'
river_flow = pd.read_csv(folder_with_data + '/river-flow.csv')
river_flow['dateTime'] = pd.to_datetime(river_flow['dateTime'], errors='raise')
river_flow['dateTime'] = river_flow['dateTime'].dt.date
river_flow = river_flow.groupby('dateTime').mean().reset_index()

climate_variables = pd.read_csv(folder_with_data + '/atmospheric-variables.csv')
climate_variables['dateTime'] = pd.to_datetime(climate_variables['dateTime'], errors='raise')
climate_variables['dateTime'] = climate_variables['dateTime'].dt.date
climate_variables = climate_variables.interpolate(method='linear', limit_direction='both', axis=0)

river_flow = river_flow.drop(columns={'Unnamed: 0'})
weather_data = pd.merge(climate_variables, river_flow, on=['dateTime'], how='inner')

weather_data = weather_data.iloc[:,[0,3,1,4,5,6,7,8,9,10,11,12,13]]
time_steps = 30 
horizon = 7 

numerical_columns = weather_data.select_dtypes(include=['float64']).columns
datetime_columns = weather_data.select_dtypes(include=['datetime64']).columns
scaler = MinMaxScaler(feature_range=(0, 1))
weather_data[numerical_columns]= pd.DataFrame(scaler.fit_transform(weather_data[numerical_columns]), columns = numerical_columns)

Y =  weather_data.iloc[:, 7:]  
X1 = weather_data.iloc[:, [1]]  
X2 = weather_data.iloc[:, [2]] 
X3 = weather_data.iloc[:, [3]] 
X4 = weather_data.iloc[:, [4]] 
X5 = weather_data.iloc[:, [5]] 
X6 = weather_data.iloc[:, [6]] 
time = weather_data.iloc[:, [0]] 

def create_sequences(X1, X2, X3, X4, X5, X6,
                     Y, time, time_steps, horizon):
    
    X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq = [], [], [], [], [], []
    
    Y1_past_seq, Y2_past_seq, Y3_past_seq, Y4_past_seq, Y5_past_seq, Y6_past_seq = [], [], [], [], [], []
    
    Y_seq = []
    
    time_seq = []

    for i in range(len(Y) - time_steps - horizon + 1):
        X1_seq.append(X1[i:i+time_steps])
        X2_seq.append(X2[i:i+time_steps])
        X3_seq.append(X3[i:i+time_steps])
        X4_seq.append(X4[i:i+time_steps])
        X5_seq.append(X5[i:i+time_steps])
        X6_seq.append(X6[i:i+time_steps])

        Y1_past_seq.append(Y.iloc[i:i+time_steps, 0]) 
        Y2_past_seq.append(Y.iloc[i:i+time_steps, 1])  
        Y3_past_seq.append(Y.iloc[i:i+time_steps, 2]) 
        Y4_past_seq.append(Y.iloc[i:i+time_steps, 3])  
        Y5_past_seq.append(Y.iloc[i:i+time_steps, 4]) 
        Y6_past_seq.append(Y.iloc[i:i+time_steps, 5]) 

        Y_seq.append(Y.iloc[i+time_steps:i+time_steps+horizon, :])  

        time_seq.append(time[i:i+time_steps])

    X1_seq = np.array(X1_seq)
    X2_seq = np.array(X2_seq)
    X3_seq = np.array(X3_seq)
    X4_seq = np.array(X4_seq)
    X5_seq = np.array(X5_seq)
    X6_seq = np.array(X6_seq)
    
    Y1_past_seq = np.array(Y1_past_seq)
    Y2_past_seq = np.array(Y2_past_seq)
    Y3_past_seq = np.array(Y3_past_seq)
    Y4_past_seq = np.array(Y4_past_seq)
    Y5_past_seq = np.array(Y5_past_seq)
    Y6_past_seq = np.array(Y6_past_seq)
    
    Y_seq = np.array(Y_seq)
    
    time_seq = np.array(time_seq)

    return (X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq,
            Y1_past_seq, Y2_past_seq, Y3_past_seq, Y4_past_seq, Y5_past_seq, Y6_past_seq,
            Y_seq, time_seq)

# Assuming time_steps and horizon are defined
X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y1_past_seq, Y2_past_seq, Y3_past_seq, Y4_past_seq, Y5_past_seq, Y6_past_seq, Y_seq, time_seq = create_sequences(
    X1, X2, X3, X4, X5, X6, Y, time, time_steps, horizon
)

modality_1 = Y1_past_seq
modality_2 = Y2_past_seq
modality_3 = Y3_past_seq
modality_4 = Y4_past_seq
modality_5 = Y5_past_seq
modality_6 = Y6_past_seq

X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, modality_1_train,modality_1_test,modality_2_train,modality_2_test,modality_3_train, modality_3_test,modality_4_train,modality_4_test,modality_5_train,modality_5_test,modality_6_train,modality_6_test,y_train, y_test  = train_test_split(X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, modality_1,modality_2,modality_3,modality_4,modality_5,modality_6, Y_seq, test_size=0.2, shuffle=False)
train_time, test_time = train_test_split(time_seq, test_size=0.2, shuffle=False)

# Model parameters
TIME_STEPS = 30 
HORIZON = 7  
N_FEATURES = 1
INPUTS = [X1_train,X2_train,X3_train,X4_train,X5_train,X6_train, modality_1_train,modality_2_train,modality_3_train,modality_4_train,modality_5_train, modality_6_train]
N_MODALS = 12
N_CNN_FILTERS = 64
CNN_FILTER_SIZE = 3
LSTM_UNITS = 64
ATTENTION_HEADS = 4

def dynamic_fusion_cell(inputs, num_modalities=12, horizon=7, stations=6):
    concatenated = Concatenate()(inputs) # stacked
    weights = Dense(num_modalities, activation='softmax')(concatenated)  
    weighted = Multiply()([concatenated, weights])  # weighted
     
    added = tf.reduce_sum(concatenated, axis=2)
    added = tf.expand_dims(added, axis=-1)
    print(added.shape)
    added = Dense(num_modalities, activation='relu')(added) # added
    print(concatenated.shape)
    print(added.shape)
    print(weighted.shape)
    all_fus_rep = tf.stack([concatenated, weighted, added], axis=0)  
    all_fus_rep = tf.transpose(all_fus_rep, perm=[1, 2, 3, 0])            
    print(all_fus_rep.shape)
    fin_weights = Dense(3, activation='softmax')(all_fus_rep)
    fin_weighted = Multiply()([all_fus_rep, fin_weights ])
    final = tf.reduce_sum(fin_weighted, axis=3)
    print(final.shape)
    final = Dense(stations)(final)
    return final

# LSTM Encoder-Decoder
def lstm_encoder_decoder(input_shape):
    inputs = Input(shape=(input_shape))
    x = LSTM(64, activation='relu')(inputs)
    x = RepeatVector(7)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    x = TimeDistributed(Dense(1))(x)
    model = Model(inputs, x)
    return model
# Multi-Head Attention
def attention_block(lstm_output):
    output = MultiHeadAttention(num_heads=ATTENTION_HEADS, key_dim=LSTM_UNITS)(lstm_output, lstm_output, lstm_output)
    return output
# Complete model
def build_model(time_steps, n_features, n_modals, horizon, inputs):
    inputs = [Input(shape=(time_steps, n_features)) for _ in range(n_modals)]
    lstm_outputs = [lstm_encoder_decoder((time_steps,n_features))(mod_input) for mod_input in inputs]
    attention_outputs = [attention_block(lstm_out) for lstm_out in lstm_outputs]
    fusion_output = dynamic_fusion_cell(attention_outputs) 
    model = Model(inputs=inputs, outputs=fusion_output)
    model.compile(optimizer='adam' , loss='mean_squared_error')
    return model
def ensemble_predict(models, x): 
    predictions = np.array([model.predict(x) for model in models])  
    avg_predictions = np.mean(predictions, axis=0)  
    return avg_predictions

n_runs = 10
durations = []
durations_predictions = []
models = []
predictionstosave = []

for i in range(n_runs):
    model = build_model(TIME_STEPS, N_FEATURES, N_MODALS, HORIZON, INPUTS)  
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    start_time = pytime.time()  
    history = model.fit([X1_train, X2_train,X3_train,X4_train,X5_train,X6_train, modality_1_train,modality_2_train,modality_3_train,modality_4_train,modality_5_train, modality_6_train], y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    end_time = pytime.time() 
    duration = end_time - start_time

    durations.append(duration)
    models.append(model)

    # print(f"Training time: {duration:.2f} seconds")
    best_epoch = np.argmin(history.history['val_loss']) + 1
    # print(f"The best epoch is {best_epoch}")

    test_loss = model.evaluate([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, modality_1_test, modality_2_test, modality_3_test, modality_4_test, modality_5_test, modality_6_test], y_test)
    # print(f'Test Loss: {test_loss}')
    start_time2 = pytime.time()
    prediction = model.predict([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, modality_1_test, modality_2_test, modality_3_test, modality_4_test, modality_5_test, modality_6_test])
    end_time2  = pytime.time()
    duration2 = end_time2 - start_time2
    # print(prediction.shape)  
    # print(f"Prediction time: {duration2:.2f} seconds")
    durations_predictions.append(duration2)
    model.save(f'Results/encdec-mul-dyn1-fusion-{i}.h5')
    predictionstosave.append(prediction)

np.save(f'Results/full-predictions-encdec-mul-dyn1-fusion.npy', predictionstosave)

scaler_modified = MinMaxScaler(feature_range=(0, 1))
scaler_modified.min_ = scaler.min_[6:]
scaler_modified.scale_ = scaler.scale_[6:]
scaler_modified.data_min_ = scaler.data_min_[6:]
scaler_modified.data_max_ = scaler.data_max_[6:]
scaler_modified.data_range_ = scaler.data_range_[6:]
scaler_modified.feature_range = scaler.feature_range

predictions = ensemble_predict(models, [X1_test, X2_test,X3_test,X4_test,X5_test,X6_test,modality_1_test,modality_2_test,modality_3_test,modality_4_test,modality_5_test, modality_6_test])
np.save(f'Results/predictions-encdec-mul-dyn1-fusion.npy', predictions)

samples, horizon, stations = y_test.shape
y_test2 = y_test.reshape(-1, stations)  
y_test_unscaled = scaler_modified.inverse_transform(y_test2)

rmse_per_run  = []
mse_per_run  = []
mae_per_run  = []
mae_high_per_run = []
mape_per_run = []
nse_per_run  = []
kge_per_run  = []

for preds in predictionstosave:
    samples, horizon, stations = preds.shape
    preds2 = preds.reshape(-1, stations)
    preds_unscaled = scaler_modified.inverse_transform(preds2)

    df_actual = pd.DataFrame(y_test_unscaled.flatten(), columns=['Flow'])
    df_pred = pd.DataFrame(preds_unscaled.flatten(), columns=['Flow'])   

    threshold_high_actual = df_actual['Flow'].quantile(0.95)
    threshold_high_pred = df_actual['Flow'].quantile(0.95)

    extreme_highs_actual = df_actual[df_actual['Flow'] >= threshold_high_actual]
    extreme_highs_pred   = df_pred[df_actual['Flow'] >= threshold_high_actual]

    rmse = mean_squared_error(y_test_unscaled.flatten(), preds_unscaled.flatten(), squared=False)
    mse = mean_squared_error(y_test_unscaled.flatten(), preds_unscaled.flatten(),  squared=True)
    mae = mean_absolute_error(y_test_unscaled.flatten(), preds_unscaled.flatten())
    mae_high = mean_absolute_error(extreme_highs_actual, extreme_highs_pred)
    mape = mean_absolute_percentage_error(y_test_unscaled.flatten(), preds_unscaled.flatten())
    nse_value = evaluator(nse, np.array(preds_unscaled.flatten()), np.array(y_test_unscaled.flatten()))
    kge_value = evaluator(kge, np.array(preds_unscaled.flatten()), np.array(y_test_unscaled.flatten()))
    
    rmse_per_run.append(round(rmse, 2))
    mse_per_run.append(round(mse, 2))
    mae_per_run.append(round(mae, 2))
    mae_high_per_run.append(round(mae_high, 2))
    mape_per_run.append(round(mape, 2))
    nse_per_run.append(round(nse_value[0], 2))
    kge_per_run.append(round(kge_value[0,0], 2))

durations_and_metrics_df = pd.DataFrame({
  'Run': [f'{i+1}' for i in range(n_runs)],
    'duration': durations,
    'prediction-duration': durations_predictions,
    'RMSE': rmse_per_run,
    'MSE':  mse_per_run,
    'MAE':  mae_per_run,
    'MAE_HIGH':  mae_high_per_run,
    'MAPE': mape_per_run,
    'NSE':  nse_per_run,
    'KGE':  kge_per_run,            
})

durations_and_metrics_df.to_csv('Results/durations-and-metrics-encdec-mul-dyn1-fusion.csv')
