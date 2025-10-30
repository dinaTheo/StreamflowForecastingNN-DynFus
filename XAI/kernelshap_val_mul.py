import tensorflow as tf
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import seaborn as sns

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


inputs_combined = np.concatenate(
    [X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, 
    modality_1_train.reshape(-1, 30, 1), modality_2_train.reshape(-1, 30, 1),
    modality_3_train.reshape(-1, 30, 1), modality_4_train.reshape(-1, 30, 1),
    modality_5_train.reshape(-1, 30, 1), modality_6_train.reshape(-1, 30, 1)],
    axis=-1
)
inputs_test_list = np.concatenate([
    X1_test, X2_test, X3_test, X4_test, X5_test, X6_test,
    modality_1_test.reshape(-1, 30, 1), modality_2_test.reshape(-1, 30, 1),
    modality_3_test.reshape(-1, 30, 1), modality_4_test.reshape(-1, 30, 1),
    modality_5_test.reshape(-1, 30, 1), modality_6_test.reshape(-1, 30, 1)
], axis=-1)
full_feature_names = [f"{name}_timestep_{t}" for t in range(1, 31) for name in numerical_columns]

for model_name in ["encdec-mul-dyn1-fusion", "encdec-mul-dyn-fusion"]:
    model = load_model(f'Results/Saved-from-run-models/{model_name}.h5')

    def model_predict_wrapper(inputs_test_list):
        num_samples = inputs_test_list.shape[0]
        x_reshaped = inputs_test_list.reshape(num_samples, 30, 12)

        input_list = [x_reshaped[:, :, i] for i in range(12)]
        preds = model.predict(input_list)
        preds = (preds.mean(axis=1)).mean(axis=1)
        return preds

    X_train_flat = inputs_combined.transpose(0, 1, 2).reshape((inputs_combined.shape[0], -1))
    X_test_flat = inputs_test_list.transpose(0, 1, 2).reshape((inputs_test_list.shape[0], -1))  
    
    # background = X_train_flat[:200]
    # test_input = X_test_flat[:10] 
    # np.random.seed(42)

    random_indices = np.random.choice(X_test_flat.shape[0], size=4, replace=False)

    test_input = X_test_flat[random_indices]

    background = shap.kmeans(X_train_flat, 200) 

    explainer = shap.KernelExplainer(model_predict_wrapper, background)
    shap_values = explainer.shap_values(test_input)
    
    X_summary = pd.DataFrame(test_input)

    shap_array = np.array(shap_values) 
    shap_array = shap_array.reshape(-1, 30, 12)
    feature_contributions = shap_array.sum(axis=1)  
    mean_feature_contributions = feature_contributions.mean(axis=0)  

    plt.bar(numerical_columns, np.abs(mean_feature_contributions))
    plt.xlabel("Feature", fontsize=20)
    plt.ylabel("SHAP mean value", fontsize=20)
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'Results/Saved-from-run-models/shap-importance-per-feature-{model_name}.png', dpi=300)
    plt.close()
    plt.clf()

    timestep_contributions = shap_array.sum(axis=2)  
    mean_timestep_contributions = timestep_contributions.mean(axis=0)  

    plt.plot(range(-29, 1), np.abs(mean_timestep_contributions))
    plt.xlabel("Timestep", fontsize=20)
    plt.ylabel("SHAP mean value", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'Results/Saved-from-run-models/shap-importance-per-timestep-{model_name}.png', dpi=300)
    plt.close()
    plt.clf()

