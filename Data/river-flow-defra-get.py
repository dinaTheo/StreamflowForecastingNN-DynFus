import requests
import os
import pandas as pd
from datetime import datetime, timedelta
import io
import numpy as np

def get_daily_data(station):
    url = f"https://nrfaapps.ceh.ac.uk/nrfa/ws/time-series?format=nrfa-csv&data-type=gdf&station={station}"

    response = requests.get(url)
    response.raise_for_status()

    lines = response.text.splitlines()
    data_start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("data,last"):
            data_start_index = i + 1
            break
    if data_start_index is None:
        raise ValueError("Data section not found in the API response.")

    data_lines = "\n".join(lines[data_start_index:])
    df = pd.read_csv(io.StringIO(data_lines), header=None, names=["date", "value", "quality"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")

def call_api_range(url, start_date, end_date, filename):
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')

    full_url = f"{url}?mineq-date={start}&maxeq-date={end}"
    try:
        response = requests.get(full_url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), parse_dates=["dateTime"])
        return df
    except Exception as e:
        print(f"Failed to fetch data for {start} to {end}: {e}")
        return pd.DataFrame()

nfra_ids = [54057,55002,54095,54001,54032,54005]
guuids=[]
for id in nfra_ids:
    url = f"https://environment.data.gov.uk/hydrology/id/stations?nrfaStationID={id}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Output only the global unique station ids
        for station in data['items']:
            guuids.append(station['stationGuid'])
            
ids = dict(zip(nfra_ids, guuids))
flow_csv_filename = f"Data/{id}-river-flow.csv"

for id, guid in ids.items():
    base_url = f"https://environment.data.gov.uk/hydrology/id/measures/{guid}-flow-i-900-m3s-qualified/readings.csv"
    start_date = datetime(1985, 1, 1, 0, 0, 0)
    end_date = datetime(2024, 10, 19, 23, 45, 0)
    chunk_years = 2

    full_data = pd.DataFrame()

    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=365 * chunk_years), end_date)
        print(f"Fetching {id} from {current_start.date()} to {current_end.date()}")

        chunk_df = call_api_range(base_url, current_start, current_end, id)
        full_data = pd.concat([full_data, chunk_df], ignore_index=True)
        current_start = current_end + timedelta(days=1)

    flow_csv_filename = f"Data/{id}-river-flow.csv"
    full_data = full_data[['dateTime', 'value']]
    full_data['date'] = full_data['dateTime']
    full_data = full_data.drop(columns="date")
    full_data['dateTime'] = pd.to_datetime(full_data['dateTime'])

    full_data.set_index('dateTime', inplace=True)
    full_index = pd.date_range(start=start_date, end=end_date, freq='15T')
    full_data = full_data.reindex(full_index)
    full_data.index.name = 'dateTime'

    full_data['value'] = pd.to_numeric(full_data['value'], errors='coerce')
    if id==55002:
        full_data['value'] = full_data['value'].interpolate(method='time', limit_direction='both')    
    hourly_data = full_data.resample('H').mean()
    hourly_data.reset_index(inplace=True)
    if id != 55002:
        daily_data = get_daily_data(id)
        # Make sure daily_data is indexed by date only (no time)
        daily_data.index = daily_data.index.normalize()

        def fill_from_daily(ts):
            date = pd.Timestamp(ts.date())
            value = daily_data.loc[date, 'value'] if date in daily_data.index else np.nan
            return value
        missing_mask = hourly_data['value'].isna()
        hourly_data.loc[missing_mask, 'value'] = hourly_data.loc[missing_mask, 'dateTime'].map(fill_from_daily)
          
    hourly_data = hourly_data.round(3)
    if hourly_data['value'].isnull().sum()>0:
        hourly_data['dateTime'] = pd.to_datetime(hourly_data['dateTime'])  
        hourly_data = hourly_data.set_index('dateTime')                    
        hourly_data['value'] = hourly_data['value'].interpolate(method='time')  
        hourly_data.reset_index(inplace=True)
    else:
        hourly_data['value'] = hourly_data['value']

    print(hourly_data.isna().sum())
    hourly_data.to_csv(flow_csv_filename, index=False)
    print(f"Saved river flow data for {id} to {flow_csv_filename}")

folder_with_data ='Data'
river_flow_combined = []
river_flow_csvs = [f for f in os.listdir(folder_with_data) if f.endswith('river-flow.csv')]
for file in river_flow_csvs:
    file_path = os.path.join(folder_with_data, file)
    river_flow_per_station = pd.read_csv(file_path, parse_dates=['dateTime'])
    river_flow_per_station['id'] = file.split('-')[0]
    river_flow_combined.append(river_flow_per_station)

concatenated_df = pd.concat(river_flow_combined, ignore_index=True)
concatenated_df = concatenated_df.rename(columns={'value':'river-flow'}) 

pivoted_df = concatenated_df.pivot(index="dateTime", columns="id", values="river-flow")
pivoted_df.reset_index(inplace=True)
pivoted_df.to_csv(folder_with_data + '/river-flow.csv', index=True)
