# This file follows from `big_1_2_annual_reduction.py` and applies latlon-wise linear fit. 


import h5py
import numpy as np
from tqdm import tqdm
import cupy as cp

alpha = 10 # alpha% percentile, alpha = [90, 75, 50, 25, 10]

if __name__ == '__main__':
    
    
    # Read the file
    filename = r"C:\SUSTech\datasets_of_graduation_project\big_outputs\hetero_of_temp\data_2mtemp_annual" + str(alpha) + "th.h5"
    with h5py.File(filename, 'r') as f:
        years = f['years'][:]
        latitudes = f['latitudes'][:]
        longitudes = f['longitudes'][:]
        data_2mtemp_annualALPHAth = f['data_2mtemp_annual' + str(alpha) + 'th'][:]
        
        DATA_2MTEMP_ANNUALalphath = {"years": years, 
                    "latitudes": latitudes, 
                    "longitudes": longitudes, 
                    "data_2mtemp_annual" + str(alpha) + "th": data_2mtemp_annualALPHAth}
    
    # dimensions of data_2mtemp_annual: (years, latitudes, longitudes)
    
    
    YEARS = DATA_2MTEMP_ANNUALalphath["years"]
    LATS = DATA_2MTEMP_ANNUALalphath["latitudes"]
    LONS = DATA_2MTEMP_ANNUALalphath["longitudes"]
    TS_2MTEMP_ANNUALalphath = DATA_2MTEMP_ANNUALalphath["data_2mtemp_annual" + str(alpha) + "th"]
    
    print("The shape of \nYEARS: %s\nLATS: %s\nLONS: %s\nTS_2MTEMP_ANNUAL%sth: %s" % (YEARS.shape, LATS.shape, LONS.shape, alpha, TS_2MTEMP_ANNUALalphath.shape))
    
    
    # Calculate the linear fit
    print("Start computing the linear fit for all the latlons...")
    TS_2MTEMP_ANNUALalphath_gpu = cp.array(TS_2MTEMP_ANNUALalphath)
    slopes_gpu = cp.zeros((len(LATS), len(LONS)), dtype=cp.float32)
    YEARS_gpu = cp.array(YEARS)
    
    for (lat_idx, lon_idx) in tqdm(cp.ndindex(len(LATS), len(LONS)), desc="Processing LatLons"):
        slopes_gpu[lat_idx, lon_idx], _ = cp.polyfit(YEARS_gpu, TS_2MTEMP_ANNUALalphath_gpu[:, lat_idx, lon_idx], 1)
    
    slopes = slopes_gpu.get()
    
    print("Finish computing the linear fit.")
    
    
    # Save the data
    print("Start saving the data...")
    filename = r"C:\SUSTech\datasets_of_graduation_project\big_outputs\hetero_of_temp\slopes_of_2mtemp_annual" + str(alpha) + "th.h5"
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('latitudes', data=LATS)
        f.create_dataset('longitudes', data=LONS)
        f.create_dataset('slopes', data=slopes)
    
    print(f"Data saved to {filename}")