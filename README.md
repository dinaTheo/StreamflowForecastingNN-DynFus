# StreamflowForecastingNN-DynFus

This is the repository with the LSTM-based **dynamic feature fusion models** for streamflow forecasting using ERA5-Land climatic and river flow measurement data.

## Project
```
.
├── Data
│   ├── Final-data/
│   ├── era-land-load.py
│   ├── era-pack.py
│   └── river-flow-defra-get.py
│
├── Models
│   ├── XAI/
│   ├── encdec_dyn1_fus_mul.py
│   ├── encdec_dyn1_fus_un.py
│   ├── encdec_dyn_fus_mul.py
│   ├── encdec_dyn_fus_un.py
│   ├── requirements.txt
│   └── .Dockerfile
│
├── Results
│   ├── Saved-from-run-models/
│   ├── mul-vs-un-results-analysis.ipynb
│   ├── multivariate-results-analysis.ipynb
│   └── univariate-results-analysis.ipynb
│
├── README.md
└── LICENSE
```


## Stage 1: Data Loading & preparation

The `Data/` folder contains scripts to:

- Load ERA5-Land data
- Retrieve river flow measurements from DEFRA/NRFA APIs
The processed data is saved in `Data/Final-data/`. These files are already included, but you can regenerate them by running the scripts (after installing the required Python libraries).

## Stage 2: Preprocessing & Model development

The models use data from `Data/Final-data/` as input. All model scripts are located in the `Models/` folder.

### Running models

To run a model using Docker:

1. Clone the repository and open it in your IDE.
2. Navigate to the `Models/` folder:
   ```bash
   cd Models
   ```
3. Build the Docker image:
   ```bash
   docker build -t <IMAGE_NAME> .
   ```
4. Run the container interactively:
   ```bash
   docker run --gpus device=0 \
     -v "$(pwd)/../Data":/app/../Data \
     -v "$(pwd)/../Results":/app/../Results \
     -it <IMAGE_NAME> bash
   ```
5. Inside the container, run a model:
   ```bash
   python3 encdec_dyn1_fus_mul.py
   ```

The model outputs are saved in `Results/Saved-from-run-models/`.

## Stage 3: Results
The `Results/` folder contains the Jupyter notebooks that analyse model predictions. These notebooks use the saved outputs from `Results/Saved-from-run-models/` to generate visualizations, graphs, and tables.

## Stage 4: Explainability

The `Models/XAI/` folder includes a script for **KernelSHAP** explainability. To run it:

```bash
python3 XAI/kernelshap_val_mul.py
```


