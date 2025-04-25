import pygrib
import numpy as np
import cupy as cp  # import CuPy
from tqdm import tqdm
import h5py

# the resolution of the sampled map
DELTA_LAT = 0.5
DELTA_LON = 2.5
N_DISTINCT_LATS = int((90 - 60) / DELTA_LAT)
N_DISTINCT_LONS = int((180 - (-180)) / DELTA_LON)

if __name__ == '__main__':
    
    
    # read the grib file
    print("Start reading the grib file...")
    msgs = pygrib.open(r"C:\SUSTech\datasets_of_graduation_project\0422_n60_to_n90.grib")
    print("Complete reading the grib file.")
    
    
    # compute the lat-lon indices
    print("The resolution of the sampled map: (delta_lat * delta_lon) = (%.2f, %.2f)" % (DELTA_LAT, DELTA_LON))
    print("Computing the lat-lon indices...")
    msg = msgs[1]
    DISTINCT_LATS_of_MAP = cp.array(msg["distinctLatitudes"])
    DISTINCT_LONS_of_MAP = cp.array(msg["distinctLongitudes"])
    if "msg" in locals(): del msg
    
    lat_indices = cp.linspace(0, len(DISTINCT_LATS_of_MAP) - 1, N_DISTINCT_LATS, dtype=int)
    lon_indices = cp.linspace(0, len(DISTINCT_LONS_of_MAP) - 1, N_DISTINCT_LONS, dtype=int)
    sampled_lats = DISTINCT_LATS_of_MAP[lat_indices]
    sampled_lons = DISTINCT_LONS_of_MAP[lon_indices]
    print("Finish computing the lat-lon indices.")
    
    
    # compute the length of the time series
    print("Computing the length of the time series...")
    msgs.rewind()
    TS_LENGTH = sum(1 for _ in msgs)
    print("The length of the time series: %d" % TS_LENGTH)
    
    TS_2MTEMP_SHAPE = (TS_LENGTH, N_DISTINCT_LATS, N_DISTINCT_LONS)
    ts_2mtemp_gpu = cp.zeros(TS_2MTEMP_SHAPE, dtype=cp.float32) # place ts_2mtemp on GPU
    print("The shape of the time series of 2m temperature: (%d, %d, %d)" % TS_2MTEMP_SHAPE)
    
    
    # compute the ts_2mtemp
    msgs.rewind()
    print("Start computing ts_2mtemp...")
    timestamps = []
    for msg_idx, msg in tqdm(enumerate(msgs), total=TS_LENGTH, desc="Processing GRIB messages", unit="msg"):
        timestamps.append(f"{msg['year']:04d}_{msg['month']:02d}_{msg['day']:02d}_{msg['hour']:02d}")
        values_gpu = cp.asarray(msg["values"])
        ts_2mtemp_gpu[msg_idx] = values_gpu[cp.ix_(lat_indices, lon_indices)]

    print("Finish computing ts_2mtemp.")
    
    
    # save the data
    DATA_2MTEMP = {
        "timestamps": timestamps, 
        "latitudes": sampled_lats.get(),
        "longitudes": sampled_lons.get(),
        "data_2mtemp": ts_2mtemp_gpu.get()
    }
    
    filename = r"C:\SUSTech\datasets_of_graduation_project\hetero_outputs\data_2mtemp.h5"

    with h5py.File(filename, 'w') as f:
        # save timestamps, latitudes, longitudes and temperature data
        f.create_dataset('timestamps', data=DATA_2MTEMP['timestamps'])
        f.create_dataset('latitudes', data=DATA_2MTEMP['latitudes'])
        f.create_dataset('longitudes', data=DATA_2MTEMP['longitudes'])
        f.create_dataset('data_2mtemp', data=DATA_2MTEMP['data_2mtemp'])

    print(f"Data saved to {filename}")
    
    
        
        
    
    
    
    
    