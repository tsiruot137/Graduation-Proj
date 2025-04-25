# The code of this file may fail to run in Jupyter Notebook, so we place it in this .py file.
# This file follows from `hetero_2_1_cupy__data_rearrangement.py` and calculates the annual alpha-th percentile of 2m temperature.


import h5py
import numpy as np
from tqdm import tqdm
import cupy as cp

alpha_list = [90, 75, 50, 25, 10]

if __name__ == '__main__':
    
    # Read the file
    filename = r"C:\SUSTech\datasets_of_graduation_project\hetero_outputs\data_2mtemp.h5"
    with h5py.File(filename, 'r') as f:
        timestamps = f['timestamps'][:]
        latitudes = f['latitudes'][:]
        longitudes = f['longitudes'][:]
        data_2mtemp = f['data_2mtemp'][:]
        
        DATA_2MTEMP = {"timestamps": timestamps, 
                    "latitudes": latitudes, 
                    "longitudes": longitudes, 
                    "data_2mtemp": data_2mtemp}
        
    # dimensions of data_2mtemp: (alpha, timestamp, latitude, longitude)
    
    
    TIMESTAMPS = DATA_2MTEMP["timestamps"]
    LATS = DATA_2MTEMP["latitudes"]
    LONS = DATA_2MTEMP["longitudes"]
    TS_2MTEMP = DATA_2MTEMP["data_2mtemp"]
    
    print("The shape of \nTIMESTAMPS: %s\nLATS: %s\nLONS: %s\nTS_2MTEMP: %s" % (TIMESTAMPS.shape, LATS.shape, LONS.shape, TS_2MTEMP.shape))
    
    
    # Obtain the indices of distinct years
    print(TIMESTAMPS[0], "->", TIMESTAMPS[0].decode())
    TIMESTAMPS_str = np.array([ts.decode() for ts in TIMESTAMPS])
    years = np.array([int(tstamp[:4]) for tstamp in TIMESTAMPS_str])
    _, first_indices = np.unique(years, return_index=True)
    print(first_indices)
    
    
    # Calculate the percentiles of 2m temperature


    TS_2MTEMP_gpu = cp.array(TS_2MTEMP)
    TS_2MTEMP_ANNUAL_REDUCTION_gpu = cp.zeros((len(alpha_list), len(first_indices), len(LATS), len(LONS)), dtype=cp.float32)

    for alpha_idx, alpha in enumerate(alpha_list):
        print("Start computing the annual %dth percentile of 2m temperature..." % alpha)
        for year_idx in tqdm(range(len(first_indices)), desc="Processing Years"):
            for lat_idx in range(len(LATS)):
                for lon_idx in range(len(LONS)):
                    start = first_indices[year_idx]
                    end = first_indices[year_idx + 1] if year_idx + 1 < len(first_indices) else TS_2MTEMP_gpu.shape[0]
                    TS_2MTEMP_ANNUAL_REDUCTION_gpu[alpha_idx, year_idx, lat_idx, lon_idx] = cp.percentile(TS_2MTEMP_gpu[start:end, lat_idx, lon_idx], alpha)
        print("Finish computing the annual %dth percentile of 2m temperature." % alpha)
        
        
    # Sort out what we have
    ALPHAS = np.array(alpha_list)
    YEARS = years[first_indices]
    TS_2MTEMP_ANNUAL_REDUCTION = TS_2MTEMP_ANNUAL_REDUCTION_gpu.get()
    
    DATA_2MTEMP_ANNUAL_REDUCTION = {
        "alphas": ALPHAS,
        "years": YEARS,
        "latitudes": LATS,
        "longitudes": LONS,
        "data_2mtemp_annual_reduction": TS_2MTEMP_ANNUAL_REDUCTION
    }
    
    
    # Save the data
    print("Saving the data of annual reduction data of 2m temperature...")
    filename = r"C:\SUSTech\datasets_of_graduation_project\hetero_outputs\data_2mtemp_annual_reduction.h5"
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('alphas', data=DATA_2MTEMP_ANNUAL_REDUCTION['alphas'])
        f.create_dataset('years', data=DATA_2MTEMP_ANNUAL_REDUCTION['years'])
        f.create_dataset('latitudes', data=DATA_2MTEMP_ANNUAL_REDUCTION['latitudes'])
        f.create_dataset('longitudes', data=DATA_2MTEMP_ANNUAL_REDUCTION['longitudes'])
        f.create_dataset('data_2mtemp_annual_reduction', data=DATA_2MTEMP_ANNUAL_REDUCTION['data_2mtemp_annual_reduction'])
    
    print(f"Data saved to {filename}")


    










        
    
    
    