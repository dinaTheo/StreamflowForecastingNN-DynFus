import cdsapi

years = list(range(1985, 2024))  # up to 2024
variables = [
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "potential_evaporation",
    "soil_type",
    "volumetric_soil_water_layer_1",
    "leaf_area_index_high_vegetation"
]

months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]
hours = [f"{h:02d}:00" for h in range(24)]

client = cdsapi.Client(    
    url='https://cds.climate.copernicus.eu/api',
    key='f76583b1-6843-4e6d-8753-d0793135bd43')

for year in years:
    for month in months:
        print(f"Downloading data for {year}")
        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": str(year),
            "month": month,
            "day": days,
            "time": hours,
            "data_format": "netcdf",
            "area": [53.02, -3.82, 51.11, -0.95]  # North, West, South, East
        }

        filename = f"era-land/{year}-{month}-atmospheric-variables.nc"
        client.retrieve("reanalysis-era5-land", request, target=filename)
        print(f"Saved data for year {year} and month {month}: {filename}")