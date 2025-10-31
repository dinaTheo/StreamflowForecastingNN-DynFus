# StreamflowForecastingNN-DynFus

## Introduction
This is the repository with the code for the LSTM-based dynamic feature fusion models for streamflow forecasting.

The project follows this structure:

```.
├── Data
│   ├── Final-data
│   ├── era-land-load.py
│   ├── era-pack.py
│   └── river-flow-defra-get.py
├── Models
│   │── XAI
│   ├── encdec_dyn1_fus_mul.py
│   ├── encdec_dyn1_fus_un.py
│   ├── encdec_dyn_fus_mul.py
│   ├── encdec_dyn_fus_un.py
│   ├── requirements.txt
│   └── .Dockerfile
│   
├── Results
│   ├── Saved-from-run-models
│   ├── mul-vs-un-results-analysis.ipynb
│   ├── multivariate-results-analysis.ipynb
│   └── univariate-results-analysis.ipynb
├── README.md
├── LICENSE
```

## Stage 1: Data loading and preparing

The **Data** folder contains the `.py` scripts that load and prepare the ERA5-Land Data and the river flow measurements from DEFRA/NRFA APIs. The final data derived from these scripts are to be saved on the **Data/Final-data/** folder. This data have already been provided.

## Stage 2: Data preprocessing and model development

The data coming from the previous stage saved on **Data/Final-data/** is to be used as inputs to the models. All the models can be found on the **Models/** folder (`.py` scripts). When the models are run, they are outputting their corresponding predictions on the **Results/Saved-from-run-models** folder. 

## Stage 3: Results

On the **Results** folder, the `.ipynb` files use the saved predictions from the models (on **Results/Saved-from-run-models/**) to produce any visuals/graphs/table info.

## Stage 4: Explainability

Finally, on the **Models/XAI/** folder, the `.py` script for the KernelSHAP explainability method is being provided.

