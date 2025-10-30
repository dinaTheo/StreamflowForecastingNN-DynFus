# StreamflowForecastingNN-DynFus

This is the repository with the code for an LSTM-based dynamic feature fusion model for streamflow forecasting.

The structure of the project is the following:

.
├── Data
│   ├── Final-data
│   ├── era-land-load.py
│   ├── era-pack.py
│   └── river-flow-defra-get.py
├── Models
│   ├── encdec_dyn1_fus_mul.py
│   ├── encdec_dyn1_fus_un.py
│   ├── encdec_dyn_fus_mul.py
│   └── encdec_dyn_fus_un.py
├── Results
│   ├── Saved-from-run-models
│   ├── mul-vs-un-results-analysis.ipynb
│   ├── multivariate-results-analysis.ipynb
│   └── univariate-results-analysis.ipynb
│── XAI
│  └── kernelshap_val_mul.py
├── README.md
├── LICENSE

The `Data` folder contains the .py scripts that load and prepare the ERA5-Land Data and the river flow measurements from DEFRA/NRFA. 

The final data from these scripts, if run, are to be saved on the `Data/Final-data/`folder. This data have been already provided, therefore there is no need to run the load scripts if one is in a rush.

This data is used as inputs to the models, found in the `Models/` folder. When the models (the .py scripts) in this folder are run, they are outputtinng the predictions on the `Results/Saved-from-run-models` folder. 

Also, on the same `Results` folder, there are .ipynb files using the saved predictions from the models (`Results/Saved-from-run-models/`) to produce any visuals/graphs/table info.

Finally, on the `XAI/` folder, the .py script for the KernelSHAP explainability method is being provided.

