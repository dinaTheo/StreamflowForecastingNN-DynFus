import time as pytime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Attention, MultiHeadAttention, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
from hydroeval import evaluator, nse, kge


folder_with_data = 'Data/Final-data/'
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
Y1 =  weather_data.iloc[:,7] 
Y2 =  weather_data.iloc[:,8] 
Y3 =  weather_data.iloc[:,9] 
Y4 =  weather_data.iloc[:,10] 
Y5 =  weather_data.iloc[:,11] 
Y6 =  weather_data.iloc[:,12] 
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
    
    Y_past_seq = []
    
    Y_seq = []
    
    time_seq = []

    for i in range(len(Y) - time_steps - horizon + 1):
        X1_seq.append(X1[i:i+time_steps])
        X2_seq.append(X2[i:i+time_steps])
        X3_seq.append(X3[i:i+time_steps])
        X4_seq.append(X4[i:i+time_steps])
        X5_seq.append(X5[i:i+time_steps])
        X6_seq.append(X6[i:i+time_steps])
        
        Y_past_seq.append(Y.iloc[i:i+time_steps]) 

        Y_seq.append(Y.iloc[i+time_steps:i+time_steps+horizon])  

        time_seq.append(time[i:i+time_steps])

    X1_seq = np.array(X1_seq)
    X2_seq = np.array(X2_seq)
    X3_seq = np.array(X3_seq)
    X4_seq = np.array(X4_seq)
    X5_seq = np.array(X5_seq)
    X6_seq = np.array(X6_seq)
    
    Y_past_seq = np.array(Y_past_seq)
    
    Y_seq = np.array(Y_seq)
    
    time_seq = np.array(time_seq)

    return (X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, 
            Y_past_seq,
            Y_seq, time_seq)

X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y1_past_seq, Y1_seq, time_seq = create_sequences(
    X1, X2, X3, X4, X5, X6, Y1, time, time_steps, horizon
)
X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y2_past_seq, Y2_seq, time_seq = create_sequences(
    X1, X2, X3, X4, X5, X6, Y2, time, time_steps, horizon
)
X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y3_past_seq, Y3_seq, time_seq = create_sequences(
    X1, X2, X3, X4, X5, X6, Y3, time, time_steps, horizon
)
X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y4_past_seq, Y4_seq, time_seq = create_sequences(
    X1, X2, X3, X4, X5, X6, Y4, time, time_steps, horizon
)
X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y5_past_seq, Y5_seq, time_seq = create_sequences(
    X1, X2, X3, X4, X5, X6, Y5, time, time_steps, horizon
)
X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y6_past_seq, Y6_seq, time_seq = create_sequences(
    X1, X2, X3, X4, X5, X6, Y6, time, time_steps, horizon
)

modality_1 = Y1_past_seq
modality_2 = Y2_past_seq
modality_3 = Y3_past_seq
modality_4 = Y4_past_seq
modality_5 = Y5_past_seq
modality_6 = Y6_past_seq

# Split into training and testing sets
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, y1_past_train, y1_past_test,  y1_train, y1_test  = train_test_split(X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y1_past_seq, Y1_seq, test_size=0.2, shuffle=False)
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, y2_past_train, y2_past_test,  y2_train, y2_test  = train_test_split(X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y2_past_seq, Y2_seq, test_size=0.2, shuffle=False)
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, y3_past_train, y3_past_test,  y3_train, y3_test  = train_test_split(X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y3_past_seq, Y3_seq, test_size=0.2, shuffle=False)
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, y4_past_train, y4_past_test,  y4_train, y4_test  = train_test_split(X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y4_past_seq, Y4_seq, test_size=0.2, shuffle=False)
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, y5_past_train, y5_past_test,  y5_train, y5_test  = train_test_split(X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y5_past_seq, Y5_seq, test_size=0.2, shuffle=False)
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, y6_past_train, y6_past_test,  y6_train, y6_test  = train_test_split(X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, Y6_past_seq, Y6_seq, test_size=0.2, shuffle=False)

train_time, test_time = train_test_split(time_seq, test_size=0.2, shuffle=False)

# Model parameters
TIME_STEPS = 30 
HORIZON = 7  
N_FEATURES = 1
N_MODALS = 7
N_CNN_FILTERS = 64
CNN_FILTER_SIZE = 3
LSTM_UNITS = 64
ATTENTION_HEADS = 4

# Multimodal Fusion
def weighted_fusion(modalities, weights):
    modalities = tf.stack(modalities, axis=1)
    weights = tf.transpose(weights, perm=[0, 2, 1,3]) 
    modalities_expanded = tf.expand_dims(modalities, axis=-2)
    weighted_sum = tf.reduce_sum(modalities_expanded * weights[:, :, tf.newaxis, :], axis=1)  
    return weighted_sum
def gating_network(inputs, num_modalities):
    x = Dense(64, activation='relu')(inputs)
    logits = Dense(num_modalities)(x) 
    gate_output = tf.nn.softmax(logits) 
    return gate_output
def dynamic_fusion_cell(inputs, num_modalities=7, horizon=7, stations=1):
    addition_result = tf.reduce_sum(inputs, axis=0)
    addition_result = tf.tile(addition_result, [1, 1, num_modalities]) 

    concat_result = tf.concat(inputs, axis=-1)  

    gate_weights = gating_network(concat_result, num_modalities=7) 
    gate_weights_2 = tf.expand_dims(gate_weights, axis=-1)  
    weighted_result = weighted_fusion(inputs, gate_weights_2)

    gate_weights_expanded = tf.expand_dims(gate_weights, axis=2) 
    addition_result_expanded = tf.expand_dims(addition_result, axis=2)  
    concat_result_expanded = tf.expand_dims(concat_result, axis=2) 
    gate_weights_expanded = tf.tile(gate_weights_expanded, [1, 1, horizon, 1]) 
    addition_result_expanded = tf.tile(addition_result_expanded, [1, 1, horizon, 1])  
    concat_result_expanded = tf.tile(concat_result_expanded, [1, 1, horizon, 1])  
    weighted_result_expanded = tf.tile(weighted_result, [1, 1, 1, num_modalities]) 
    fusion_terms = gate_weights_expanded * (addition_result_expanded + concat_result_expanded + weighted_result_expanded)
    fused_output = tf.reduce_sum(fusion_terms, axis=2) 
    final_output = tf.keras.layers.Dense(stations)(fused_output)
    return final_output
# def dynamic_fusion_cell(inputs, num_modalities=8, horizon=7, stations=1):
#     addition_result = tf.reduce_sum(inputs, axis=0)
#     addition_result = tf.tile(addition_result, [1, 1, num_modalities]) 
#     concat_result = tf.concat(inputs, axis=-1)  
#     gate_weights = gating_network(concat_result, num_modalities) 
#     gate_weights_2 = tf.expand_dims(gate_weights, axis=-1)  
#     weighted_result = weighted_fusion(inputs, gate_weights_2)

#     addition_result_expanded = tf.expand_dims(addition_result, axis=2)  
#     concat_result_expanded = tf.expand_dims(concat_result, axis=2) 
#     addition_result_expanded = tf.tile(addition_result_expanded, [1, 1, horizon, 1])  
#     concat_result_expanded = tf.tile(concat_result_expanded, [1, 1, horizon, 1])  
#     weighted_result_expanded = tf.tile(weighted_result, [1, 1, 1, num_modalities]) 
#     learned_weights = Dense(3, activation='softmax')(
#         tf.concat([addition_result_expanded , concat_result_expanded , weighted_result_expanded ], axis=-1)
#     )

#     addition_weight, concat_weight, weighted_result_weight = tf.unstack(learned_weights, axis=-1)
#     addition_weight = tf.expand_dims(addition_weight, axis=-1)  
#     concat_weight = tf.expand_dims(concat_weight, axis=-1)  
#     weighted_result_weight = tf.expand_dims(weighted_result_weight, axis=-1)  

#     addition_result_weighted = addition_weight * addition_result_expanded 
#     concat_result_weighted = concat_weight * concat_result_expanded 
#     weighted_result_weighted = weighted_result_weight * weighted_result_expanded 
#     fusion = addition_result_weighted + concat_result_weighted + weighted_result_weighted

#     fused_output = tf.reduce_sum(fusion, axis=1)  
#     final_output = Dense(stations)(fused_output)  
    
#     return final_output
# LSTM Encoder-Decoder
def lstm_encoder_decoder(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, activation='relu')(inputs)
    x = RepeatVector(7)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    x = TimeDistributed(Dense(1))(x)
    model = Model(inputs, x)
    return model
# Multi-Head Attention
def attention_block(lstm_output):
    output = MultiHeadAttention(num_heads=ATTENTION_HEADS, key_dim=LSTM_UNITS // ATTENTION_HEADS)(lstm_output, lstm_output, lstm_output)
    return output
# Multimodal Fusion
def fusion(modalities, horizon):
    concatenated = Concatenate()(modalities)
    attention = Attention()([concatenated, concatenated])  
    output = Dense(1)(attention) 
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
def ensemble_predict(predictions): 
    predictions = np.array(predictions)
    avg_predictions = np.mean(predictions, axis=0)
    return avg_predictions

n_runs = 10                  
n_stations = 6              
models_per_run = []          
predictions_per_run = []     
durations_per_run = []   
durations_for_preds_per_run = []   
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
y_train_sets = [y1_train, y2_train, y3_train, y4_train, y5_train, y6_train]
y_test_sets = [y1_test, y2_test, y3_test, y4_test, y5_test, y6_test]
y_past_train_sets = [y1_past_train, y2_past_train, y3_past_train, y4_past_train, y5_past_train, y6_past_train]
y_past_test_sets = [y1_past_test, y2_past_test, y3_past_test, y4_past_test, y5_past_test, y6_past_test]

for run in range(n_runs):
    run_models = []             
    run_predictions = []

    for i, (y_train, y_past_train) in enumerate(zip(y_train_sets, y_past_train_sets)):
        INPUTS = [X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, y_past_train]
        model = build_model(TIME_STEPS, N_FEATURES, N_MODALS, HORIZON, INPUTS)
        model.summary()
        print(a)
        run_models.append(model)
    
    start_time = pytime.time() 

    for model, y_train, y_past_train in zip(run_models, y_train_sets, y_past_train_sets):
        history = model.fit([X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, y_past_train], 
                            y_train, epochs=50, batch_size=32, validation_split=0.2, 
                            callbacks=[early_stopping])
    
    end_time = pytime.time()
    duration = end_time - start_time
    durations_per_run.append(duration)

    for model, y_test, y_past_test in zip(run_models, y_test_sets, y_past_test_sets):
        test_loss = model.evaluate([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, y_past_test], y_test)
        # print(f"Test Loss for station {i + 1}, Run {run + 1}: {test_loss}")

    start_time2 = pytime.time() 

    for model, y_past_test in zip(run_models, y_past_test_sets):
        predictions = model.predict([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, y_past_test])
        run_predictions.append(predictions)

    end_time2 = pytime.time()
    duration2 = end_time2 - start_time2
    durations_for_preds_per_run.append(duration2)

    predictions_per_run.append(run_predictions)
    for i, model in enumerate(run_models):
        model.save(f'Results/Saved-from-run-models/encdec-un-dyn1-fusion-run-{run}-station-{i}.h5')

final_ensemble_predictions = []
for station_idx in range(n_stations):
    station_predictions_across_runs = [predictions_per_run[run][station_idx] for run in range(n_runs)]
    station_ensemble = ensemble_predict(station_predictions_across_runs)
    final_ensemble_predictions.append(station_ensemble)

np.save('Results/Saved-from-run-models/predictions-per-run-encdec-dyn1-fusion.npy', final_ensemble_predictions)
np.save('Results/Saved-from-run-models/full-predictions-per-run-encdec-dyn1-fusion.npy', predictions_per_run)
np.save('Results/Saved-from-run-models/durations-encdec-dyn1-fusion.npy', durations_per_run)
np.save('Results/Saved-from-run-models/durations-for-predictions-encdec-dyn1-fusion.npy', durations_for_preds_per_run)

scaler_modified = MinMaxScaler(feature_range=(0, 1))
scaler_modified.feature_range = scaler.feature_range

ys_test = []
for j, y_test in enumerate([y1_test, y2_test, y3_test, y4_test, y5_test, y6_test]): 
    scaler_modified.min_ =  scaler.min_[6+j]
    scaler_modified.scale_ =  scaler.scale_[6+j]
    scaler_modified.data_min_ =  scaler.data_min_[6+j]
    scaler_modified.data_max_ =  scaler.data_max_[6+j]
    scaler_modified.data_range_ =  scaler.data_range_[6+j]

    y_test = scaler_modified.inverse_transform(y_test)
    ys_test.append(y_test)

predictions_per_run = np.array(predictions_per_run)
predictions_per_run2 =[]

for i in range(predictions_per_run.shape[0]):  
    batch_inverse = [] 
    for j in range(predictions_per_run.shape[1]): 
        data_slice = predictions_per_run[i, j, :, :, 0]
        scaler_modified.min_ = scaler.min_[6+j]
        scaler_modified.scale_ = scaler.scale_[6+j]
        scaler_modified.data_min_ = scaler.data_min_[6+j]
        scaler_modified.data_max_ = scaler.data_max_[6+j]
        scaler_modified.data_range_ = scaler.data_range_[6+j]

        data_inverse_transformed = scaler_modified.inverse_transform(data_slice)
        
        batch_inverse.append(data_inverse_transformed)
    
    predictions_per_run2.append(batch_inverse)

predictions_per_run2 = np.array(predictions_per_run2)

y_test_combined = np.stack(ys_test, axis=1) 
y_test_combined = y_test_combined.transpose(0,2,1)

rmse_per_run  = []
mse_per_run  = []
mae_per_run = []
mae_high_per_run  = []
mape_per_run = []
nse_per_run = []
kge_per_run = []

for run_idx, prediction_per_run in enumerate(predictions_per_run2): 
    preds_unscaled = prediction_per_run.transpose(1,2,0).reshape(-1,6).flatten()
    y_test_unscaled = y_test_combined.reshape(-1,6).flatten()

    df_actual = pd.DataFrame(y_test_unscaled, columns=['Flow'])
    df_pred = pd.DataFrame(preds_unscaled, columns=['Flow'])   

    threshold_high_actual = df_actual['Flow'].quantile(0.95)
    threshold_high_pred = df_actual['Flow'].quantile(0.95)

    extreme_highs_actual = df_actual[df_actual['Flow'] >= threshold_high_actual]
    extreme_highs_pred   = df_pred[df_actual['Flow'] >= threshold_high_actual]

    rmse = mean_squared_error(y_test_unscaled, preds_unscaled, squared=False)
    mse = mean_squared_error(y_test_unscaled, preds_unscaled, squared=True)
    mae = mean_absolute_error(y_test_unscaled, preds_unscaled)
    mae_high = mean_absolute_error(extreme_highs_actual, extreme_highs_pred)
    mape = mean_absolute_percentage_error(y_test_unscaled, preds_unscaled)
    nse_value = evaluator(nse, preds_unscaled, y_test_unscaled)
    kge_value = evaluator(kge, preds_unscaled, y_test_unscaled)

    rmse_per_run.append(round(rmse, 2))
    mse_per_run.append(round(mse, 2))
    mae_per_run.append(round(mae, 2))
    mae_high_per_run.append(round(mae_high, 2))
    mape_per_run.append(round(mape, 2))
    nse_per_run.append(round(nse_value[0], 2))  
    kge_per_run.append(round(kge_value[0, 0], 2))

durations_and_metrics_df = pd.DataFrame({
    'Run': [f'{i+1}' for i in range(len(mae_per_run))],
    'duration': durations_for_preds_per_run, 
    'prediction-duration': durations_for_preds_per_run,
    'RMSE': rmse_per_run,
    'MSE':  mse_per_run,
    'MAE': mae_per_run,
    'MAE_HIGH':  mae_high_per_run,
    'MAPE': mape_per_run,
    'NSE': nse_per_run,
    'KGE': kge_per_run,
})

durations_and_metrics_df.to_csv('Results/Saved-from-run-models/durations-and-metrics-encdec-un-dyn1-fusion.csv')



