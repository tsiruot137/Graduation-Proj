# The code of this file may fail to run in Jupyter Notebook, so we place it in this .py file.
# This file follows from `big_1_1_cupy.py` and calculates the annual alpha-th percentile of 2m temperature.


import h5py
import numpy as np
from tqdm import tqdm
import cupy as cp

alpha = 10 # alpha% percentile, alpha = [90, 75, 50, 25, 10]


if __name__ == '__main__':
    
    
    # Read the file
    filename = r"C:\SUSTech\datasets_of_graduation_project\big_outputs\data_2mtemp.h5"
    with h5py.File(filename, 'r') as f:
        timestamps = f['timestamps'][:]
        latitudes = f['latitudes'][:]
        longitudes = f['longitudes'][:]
        data_2mtemp = f['data_2mtemp'][:]
        
        DATA_2MTEMP = {"timestamps": timestamps, 
                    "latitudes": latitudes, 
                    "longitudes": longitudes, 
                    "data_2mtemp": data_2mtemp}
        
    # dimensions of data_2mtemp: (timestamps, latitudes, longitudes)
    
    
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
    TS_2MTEMP_annualALPHAth_gpu = cp.zeros((len(first_indices), len(LATS), len(LONS)), dtype=cp.float32)

    print("Start computing the annual %dth percentile of 2m temperature..." % alpha)
    for year_idx in tqdm(range(len(first_indices)), desc="Processing Years"):
        for lat_idx in range(len(LATS)):
            for lon_idx in range(len(LONS)):
                start = first_indices[year_idx]
                end = first_indices[year_idx + 1] if year_idx + 1 < len(first_indices) else TS_2MTEMP_gpu.shape[0]
                TS_2MTEMP_annualALPHAth_gpu[year_idx, lat_idx, lon_idx] = cp.percentile(TS_2MTEMP_gpu[start:end, lat_idx, lon_idx], alpha)

    TS_2MTEMP_annualALPHAth = TS_2MTEMP_annualALPHAth_gpu.get()
    
    
    # Save the data
    print("Saving the data of annual 90th percentile of 2m temperature...")
    DATA_2MTEMP_ANNUALalphaTH = {
        "years": years[first_indices],
        "latitudes": LATS,
        "longitudes": LONS,
        "data_2mtemp_annual" + str(alpha) + "th": TS_2MTEMP_annualALPHAth
    }
    
    filename = r"C:\SUSTech\datasets_of_graduation_project\big_outputs\hetero_of_temp\data_2mtemp_annual" + str(alpha) + "th.h5"
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('years', data=DATA_2MTEMP_ANNUALalphaTH['years'])
        f.create_dataset('latitudes', data=DATA_2MTEMP_ANNUALalphaTH['latitudes'])
        f.create_dataset('longitudes', data=DATA_2MTEMP_ANNUALalphaTH['longitudes'])
        f.create_dataset('data_2mtemp_annual' + str(alpha) + 'th', data=DATA_2MTEMP_ANNUALalphaTH['data_2mtemp_annual' + str(alpha) + 'th'])
    
    print(f"Data saved to {filename}")


    










        
    
    
    