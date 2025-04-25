# This file follows from `hetero_2_6_-1_MA.py` and precedes the file `hetero_2_6_1_visualize_MA_sa_dt0sa.ipynb`.
# This file do the similar thing as `hetero_2_3_analyse_the_annual_ts.py` but for the annual-reducted-MA data instead of the original data (annual-reducted data).


import h5py
import numpy as np
from tqdm import tqdm
import cupy as cp


if __name__ == '__main__':
    
    
    # Read the file
    filename = r"C:\SUSTech\datasets_of_graduation_project\hetero_outputs\data_2mtemp_annual_reduction_MA.h5"
    with h5py.File(filename, 'r') as f:
        alphas = f['alphas'][:]
        years = f['years'][:]
        latitudes = f['latitudes'][:]
        longitudes = f['longitudes'][:]
        data_2mtemp_annual_reduction_ma = f['data_2mtemp_annual_reduction_ma'][:]
        window_size = f['window_size'][()]
        
        ALPHAS = alphas
        YEARS = years
        LATS = latitudes
        LONS = longitudes
        TS_2MTEMP_ANNUAL_REDUCTION_MA = data_2mtemp_annual_reduction_ma
        WINDOW_SIZE = window_size
    
    # dimensions of data_2mtemp_annual: (alpha, years, latitudes, longitudes)
    print("WINDOW_SIZE:", WINDOW_SIZE, ";", "type(WINDOW_SIZE):", type(WINDOW_SIZE))
    print("The shape of TS_2MTEMP_ANNUAL_REDUCTION_MA[alpha, year, lat, lon] is: \nALPHAS: ", ALPHAS.shape, "\nYEARS: ", YEARS.shape, "\nLATS: ", LATS.shape, "\nLONS: ", LONS.shape)
    
    
    
    # Calculate the overall linear fit
    print("Start computing the overall (80 years) linear fit for all the latlons...")
    TS_2MTEMP_ANNUAL_REDUCTION_MA_gpu = cp.array(TS_2MTEMP_ANNUAL_REDUCTION_MA)
    overall_slopes_gpu = cp.zeros((len(ALPHAS), len(LATS), len(LONS)), dtype=cp.float32)
    print("YEARS: \n", YEARS)
    YEARS_gpu = cp.array(YEARS)
    
    for (lat_idx, lon_idx) in tqdm(cp.ndindex(len(LATS), len(LONS)), desc="Processing LatLons"):
        for alpha_idx in range(len(ALPHAS)):
            overall_slopes_gpu[alpha_idx, lat_idx, lon_idx], _ = cp.polyfit(YEARS_gpu, TS_2MTEMP_ANNUAL_REDUCTION_MA_gpu[alpha_idx, :, lat_idx, lon_idx], 1)
    
    overall_slopes = overall_slopes_gpu.get()
    
    print("Finish computing the overall linear fit.")
    
    # Save the data
    print("Start saving the overall slopes...")
    filename = r"C:\SUSTech\datasets_of_graduation_project\hetero_outputs\overall_slopes_MA.h5"
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('alphas', data=ALPHAS)
        f.create_dataset('latitudes', data=LATS)
        f.create_dataset('longitudes', data=LONS)
        f.create_dataset('overall_slopes_ma', data=overall_slopes)
        f.create_dataset('window_size', data=WINDOW_SIZE)
    
    print(f"Data saved to {filename}")
    
    
    # Calculate the decadely linear fit
    print("Start computing the decadely (every 10 years) linear fit for all the latlons...")
    DECADES = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020] # 7 decades: 1950s, 1960s, ..., 2010s
    DECADES_gpu = cp.array(DECADES)
    DECADES_indices = [0, 10, 20, 30, 40, 50, 60, 70] # The indices of the decades in the YEARS
    decadely_slopes_gpu = cp.zeros((len(ALPHAS), len(DECADES)-1, len(LATS), len(LONS)), dtype=cp.float32)
    
    for (lat_idx, lon_idx) in tqdm(cp.ndindex(len(LATS), len(LONS)), desc="Processing LatLons"):
        for alpha_idx in range(len(ALPHAS)):
            for decade_idx in range(len(DECADES_indices) - 1):
                start_idx = DECADES_indices[decade_idx]
                end_idx = DECADES_indices[decade_idx + 1]
                decadely_slopes_gpu[alpha_idx, decade_idx, lat_idx, lon_idx], _ = cp.polyfit(YEARS_gpu[start_idx:end_idx], TS_2MTEMP_ANNUAL_REDUCTION_MA_gpu[alpha_idx, start_idx:end_idx, lat_idx, lon_idx], 1)
                
    decadely_slopes = decadely_slopes_gpu.get()
    print("Finish computing the decadely linear fit.")
    
    
    # Save the data
    print("Start saving the decadely slopes...")
    filename = r"C:\SUSTech\datasets_of_graduation_project\hetero_outputs\decadely_slopes_MA.h5"
    with h5py.File(filename, 'w') as f:
        f.create_dataset('alphas', data=ALPHAS)
        f.create_dataset('decades', data=DECADES)
        f.create_dataset('latitudes', data=LATS)
        f.create_dataset('longitudes', data=LONS)
        f.create_dataset('decadely_slopes_ma', data=decadely_slopes)
        f.create_dataset('window_size', data=WINDOW_SIZE)
        
    print(f"Data saved to {filename}")








