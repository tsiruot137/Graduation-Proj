# This file applies the latlon-wise moving average (MA) and obtains a same-size data as the original one: `data_2mtemp_annual_reduction.h5`.
# This file follows from the file `hetero_2_2_annual_reduction.py` and precedes the file `hetero_2_6_analyse_the_annual_ts_MA.py`.

from turtle import end_fill
import h5py
import numpy as np
from tqdm import tqdm
import cupy as cp

WINDOW_SIZE = 10 # The window size for moving average

if __name__ == '__main__':
    
    
    # Read the file
    filename = r"C:\SUSTech\datasets_of_graduation_project\hetero_outputs\data_2mtemp_annual_reduction.h5"
    with h5py.File(filename, 'r') as f:
        alphas = f['alphas'][:]
        years = f['years'][:]
        latitudes = f['latitudes'][:]
        longitudes = f['longitudes'][:]
        data_2mtemp_annual_reduction = f['data_2mtemp_annual_reduction'][:]
        
        years_int = years.astype(np.int32) # Convert years from string to int32
        
        ALPHAS = alphas
        YEARS = years_int
        LATS = latitudes
        LONS = longitudes
        TS_2MTEMP_ANNUAL_REDUCTION = data_2mtemp_annual_reduction
        
    # dimensions of data_2mtemp_annual: (alpha, years, latitudes, longitudes)
    print("The shape of TS_2MTEMP_ANNUAL_REDUCTION[alpha, year, lat, lon] is: \nALPHAS: ", ALPHAS.shape, "\nYEARS: ", YEARS.shape, "\nLATS: ", LATS.shape, "\nLONS: ", LONS.shape)
    print("The datatype of ALPHAS:", type(ALPHAS[0]))
    print("The datatype of YEARS:", type(YEARS[0]))
    
    
    # Apply the latlon-wise MA
    print("Start computing the latlon-wise moving average...")
    TS_2MTEMP_ANNUAL_REDUCTION_gpu = cp.array(TS_2MTEMP_ANNUAL_REDUCTION)
    TS_2MTEMP_ANNUAL_REDUCTION_MA_gpu = cp.zeros((len(ALPHAS), len(YEARS), len(LATS), len(LONS)), dtype=cp.float32)
    
    LEN_MA = YEARS.shape[0] - WINDOW_SIZE + 1 # The length of the time series produced by MA. In "valid" mode the MA produces a shorter time series than the original one.
    
    for (lat_idx, lon_idx) in tqdm(cp.ndindex(len(LATS), len(LONS)), desc="Processing LatLons"):
        for alpha_idx in range(len(ALPHAS)):
            TS_2MTEMP_ANNUAL_REDUCTION_MA_gpu[alpha_idx, :LEN_MA, lat_idx, lon_idx] = cp.convolve(TS_2MTEMP_ANNUAL_REDUCTION_gpu[alpha_idx, :, lat_idx, lon_idx], cp.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
            
    TS_2MTEMP_ANNUAL_REDUCTION_MA_gpu = TS_2MTEMP_ANNUAL_REDUCTION_MA_gpu[:, :LEN_MA, :, :] # Keep only the valid part of MA
    print("The shape of TS_2MTEMP_ANNUAL_REDUCTION_MA:", TS_2MTEMP_ANNUAL_REDUCTION_MA_gpu.shape)
    TS_2MTEMP_ANNUAL_REDUCTION_MA = TS_2MTEMP_ANNUAL_REDUCTION_MA_gpu.get()
    print("Finish computing the latlon-wise MA.")
    
    
    # Save the data
    print("Start saving the latlon-wise moving average...")
    filename = r"C:\SUSTech\datasets_of_graduation_project\hetero_outputs\data_2mtemp_annual_reduction_MA.h5"
    
    YEARS_MA = YEARS[int(WINDOW_SIZE/2) : int(WINDOW_SIZE/2) + LEN_MA] # The years corresponding to the MA series.
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('alphas', data=ALPHAS)
        f.create_dataset('years', data=YEARS_MA)
        f.create_dataset('latitudes', data=LATS)
        f.create_dataset('longitudes', data=LONS)
        f.create_dataset('data_2mtemp_annual_reduction_ma', data=TS_2MTEMP_ANNUAL_REDUCTION_MA)
        f.create_dataset('window_size', data=WINDOW_SIZE) # Save the window size for future reference
    
    print(f"Data saved to {filename}")