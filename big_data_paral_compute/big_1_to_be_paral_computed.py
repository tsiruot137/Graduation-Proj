# The code of this file is waiting to be optimized as a parallel computing task, which is completed in the file `big_1_1_cupy.py`.



# import numpy as np

# # 假设有一个 10x10 的数组
# arr = np.arange(100).reshape(10, 10)
# print(arr)

# # 在行和列方向上均匀抽取 5 个点
# num_samples = 5
# row_indices = np.linspace(0, arr.shape[0] - 1, num_samples, dtype=int)
# col_indices = np.linspace(0, arr.shape[1] - 1, num_samples, dtype=int)

# # 获取抽取的子数组
# sampled_array = arr[np.ix_(row_indices, col_indices)]
# print(sampled_array)



import pygrib
import numpy as np

# the resolution of the entire map (in degrees)
DELTA_LAT = 0.5
DELTA_LON = 2.5
N_DISTINCT_LATS = int((90 - 66.5) / DELTA_LAT)
N_DISTINCT_LONS = int((180 - (-180)) / DELTA_LON)

if __name__ == '__main__':
    
    print("Start reading the grib file...")
    msgs = pygrib.open(r"C:\SUSTech\datasets_of_graduation_project\0220.grib")
    print("Complete reading the grib file.")
    
    print("The resolution of the sampled map: (delta_lat * delta_lon) = (%.2f, %.2f)" % (DELTA_LAT, DELTA_LON))
    print("Computing the lat-lon indices...")
    DISTINCT_LATS_of_MAP = msgs[1]["distinctLatitudes"]
    DISTINCT_LONS_of_MAP = msgs[1]["distinctLongitudes"]
    lat_indices = np.linspace(0, len(DISTINCT_LATS_of_MAP) - 1, N_DISTINCT_LATS, dtype=int)
    lon_indices = np.linspace(0, len(DISTINCT_LONS_of_MAP) - 1, N_DISTINCT_LONS, dtype=int)
    sampled_lats = DISTINCT_LATS_of_MAP[lat_indices]
    sampled_lons = DISTINCT_LONS_of_MAP[lon_indices]
    print("Finish computing the lat-lon indices.")
    
    print("Computing the length of the time series...")
    msgs.rewind()
    TS_LENGTH = 0
    for msg in msgs:
        TS_LENGTH += 1
    print("The length of the time series: %d" % TS_LENGTH)
    
    TS_2MTEMP_SHAPE = (TS_LENGTH, N_DISTINCT_LATS, N_DISTINCT_LONS)
    ts_2mtemp = np.zeros(TS_2MTEMP_SHAPE)
    print("The shape of the time series of 2m temperature: (%d, %d, %d)" % TS_2MTEMP_SHAPE)
    msgs.rewind()
    print("Start computing ts_2mtemp...")
    timestamps = []
    for msg_idx, msg in enumerate(msgs):
        timestamps.append(f"{msg['year']:04d}_{msg['month']:02d}_{msg['day']:02d}_{msg['hour']:02d}")
        ts_2mtemp[msg_idx] = msg["values"][np.ix_(lat_indices, lon_indices)]
    # Attach the axis of lats, lons and timestamps to the 3D numpy array ts_2mtemp. To access the data, first access the corresponding indices of lats, lons and timestamps from the corresponding axis.
    DATA_2MTEMP = {"timestamps": timestamps, 
                   "latitudes": sampled_lats, 
                   "longitudes": sampled_lons, 
                   "data_2mtemp": ts_2mtemp}
    print("Finish computing the time series of 2m temperature.")
    
    
        
        
    
    
    
    
    