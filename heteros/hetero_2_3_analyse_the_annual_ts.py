# This file follows from `hetero_2_2_annual_reduction.py` and applies latlon-wise linear fit. 


from turtle import end_fill
import h5py
import numpy as np
from tqdm import tqdm
import cupy as cp


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
        DATA_2MTEMP_ANNUAL_REDUCTION = {
            "alphas": alphas,
            "years": years_int,
            "latitudes": latitudes,
            "longitudes": longitudes,
            "data_2mtemp_annual_reduction": data_2mtemp_annual_reduction
        }
    
    # dimensions of data_2mtemp_annual: (alpha, years, latitudes, longitudes)
    
    
    ALPHAS = DATA_2MTEMP_ANNUAL_REDUCTION["alphas"]
    YEARS = DATA_2MTEMP_ANNUAL_REDUCTION["years"]
    LATS = DATA_2MTEMP_ANNUAL_REDUCTION["latitudes"]
    LONS = DATA_2MTEMP_ANNUAL_REDUCTION["longitudes"]
    TS_2MTEMP_ANNUAL_REDUCTION = DATA_2MTEMP_ANNUAL_REDUCTION["data_2mtemp_annual_reduction"]
    
    print("The shape of TS_2MTEMP_ANNUAL_REDUCTION[alpha, year, lat, lon] is: \nALPHAS: ", ALPHAS.shape, "\nYEARS: ", YEARS.shape, "\nLATS: ", LATS.shape, "\nLONS: ", LONS.shape)
    
    
    
    # Calculate the overall linear fit
    print("Start computing the overall (80 years) linear fit for all the latlons...")
    TS_2MTEMP_ANNUAL_REDUCTION_gpu = cp.array(TS_2MTEMP_ANNUAL_REDUCTION)
    overall_slopes_gpu = cp.zeros((len(ALPHAS), len(LATS), len(LONS)), dtype=cp.float32)
    print("YEARS: \n", YEARS)
    YEARS_gpu = cp.array(YEARS)
    
    for (lat_idx, lon_idx) in tqdm(cp.ndindex(len(LATS), len(LONS)), desc="Processing LatLons"):
        for alpha_idx in range(len(ALPHAS)):
            overall_slopes_gpu[alpha_idx, lat_idx, lon_idx], _ = cp.polyfit(YEARS_gpu, TS_2MTEMP_ANNUAL_REDUCTION_gpu[alpha_idx, :, lat_idx, lon_idx], 1)
    
    overall_slopes = overall_slopes_gpu.get()
    
    print("Finish computing the overall linear fit.")
    
    # Save the data
    print("Start saving the overall slopes...")
    filename = r"C:\SUSTech\datasets_of_graduation_project\hetero_outputs\overall_slopes.h5"
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('alphas', data=ALPHAS)
        f.create_dataset('latitudes', data=LATS)
        f.create_dataset('longitudes', data=LONS)
        f.create_dataset('overall_slopes', data=overall_slopes)
    
    print(f"Data saved to {filename}")
    
    
    
    # Calculate the decadely linear fit
    print("Start computing the decadely (every 10 years) linear fit for all the latlons...")
    DECADES = [1945, 1955, 1965, 1975, 1985, 1995, 2005, 2015, 2025] # 8 decades in YEARS: 1945-1954, 1955-1964, ...
    DECADES_gpu = cp.array(DECADES)
    DECADES_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80] # The indices of the decades in the YEARS
    decadely_slopes_gpu = cp.zeros((len(ALPHAS), len(DECADES)-1, len(LATS), len(LONS)), dtype=cp.float32)
    
    for (lat_idx, lon_idx) in tqdm(cp.ndindex(len(LATS), len(LONS)), desc="Processing LatLons"):
        for alpha_idx in range(len(ALPHAS)):
            for decade_idx in range(len(DECADES_indices) - 1):
                start_idx = DECADES_indices[decade_idx]
                end_idx = DECADES_indices[decade_idx + 1]
                decadely_slopes_gpu[alpha_idx, decade_idx, lat_idx, lon_idx], _ = cp.polyfit(YEARS_gpu[start_idx:end_idx], TS_2MTEMP_ANNUAL_REDUCTION_gpu[alpha_idx, start_idx:end_idx, lat_idx, lon_idx], 1)
                
    decadely_slopes = decadely_slopes_gpu.get()
    print("Finish computing the decadely linear fit.")
    
    
    # Save the data
    print("Start saving the decadely slopes...")
    filename = r"C:\SUSTech\datasets_of_graduation_project\hetero_outputs\decadely_slopes.h5"
    with h5py.File(filename, 'w') as f:
        f.create_dataset('alphas', data=ALPHAS)
        f.create_dataset('decades', data=DECADES)
        f.create_dataset('latitudes', data=LATS)
        f.create_dataset('longitudes', data=LONS)
        f.create_dataset('decadely_slopes', data=decadely_slopes)
        
    print(f"Data saved to {filename}")

                