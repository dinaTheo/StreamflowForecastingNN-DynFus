import os
import xarray as xr
import zipfile
import pandas as pd
import tempfile

folder = 'Data/era-land/'
final_folder = 'Data'
files = sorted([f for f in os.listdir(folder) if f.endswith('.nc') or f.endswith('.zip')])
df_list = []

for file in files:
    file_path = os.path.join(folder, file)

    if zipfile.is_zipfile(file_path):
        print(f"\nExtracting file: {file}")
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            instant_file = None
            accum_file = None
            print(temp_dir)
            for f in os.listdir(temp_dir):
                print(f)
                if 'instant' in f and f.endswith('.nc'):
                    instant_file = os.path.join(temp_dir, f)
                elif 'accum' in f and f.endswith('.nc'):
                    accum_file = os.path.join(temp_dir, f)
            if not instant_file and not accum_file:
                print(f"No usable .nc files found: {file}")
                continue

            try:
                ds_instant = xr.open_dataset(instant_file, engine='netcdf4', decode_times=True) if instant_file else None
            except:
                ds_instant = xr.open_dataset(instant_file, engine='scipy', decode_times=True) if instant_file else None

            try:
                ds_accum = xr.open_dataset(accum_file, engine='netcdf4', decode_times=True) if accum_file else None
            except:
                ds_accum = xr.open_dataset(accum_file, engine='scipy', decode_times=True) if accum_file else None

            if ds_instant and ds_accum:
                merged_ds = xr.merge([ds_instant, ds_accum], compat='override')
            else:
                merged_ds = ds_instant or ds_accum

            df = merged_ds.to_dataframe().reset_index()

            for col in df.select_dtypes(include=['float']).columns:
                df[col] = df[col].astype('float64')

            if 'valid_time' in df.columns:
                df['dateTime'] = pd.to_datetime(df['valid_time'], errors='raise')
                df['dateTime'] = df['dateTime'].dt.date
                df = df.drop(columns={'number', 'expver', 'valid_time'})
                df = df.groupby(['longitude','latitude','dateTime']).mean().reset_index()
                df = df.drop(columns={'longitude', 'latitude'})
                df = df.groupby(['dateTime']).mean().reset_index()
            df_list.append(df)
if df_list:
    full_df = pd.concat(df_list, ignore_index=True)
    print(full_df.head())

    output_path = os.path.join(final_folder, 'atmospheric-variables.csv')
    full_df.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"\nSaved merged data to: {output_path}")
else:
    print("\nNo data merged. Check file structure or file contents.")
