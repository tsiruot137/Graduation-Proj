import numpy as np
import multiprocessing as mp
import pygrib  # need to open the grib file within each process
from functools import partial



# compute for a single (lat, lon), (processes are independent)
def process_latlon(latlon, latlon_to_latlonIdx, latlonIdx_to_vicinityIdx):
    lat, lon = latlon
    
    # need to open the grib file within each process
    msgs = pygrib.open(r"C:\SUSTech\datasets_of_graduation_project\0220.grib")

    lat1_idx, lat2_idx = latlonIdx_to_vicinityIdx[latlon_to_latlonIdx[(lat, lon)]][0]
    lon1_idx, lon2_idx = latlonIdx_to_vicinityIdx[latlon_to_latlonIdx[(lat, lon)]][1]
    
    values = np.array([np.mean(msg["values"][lat1_idx:lat2_idx, lon1_idx:lon2_idx]) for msg in msgs])
    msgs.close()
    return (lat, lon), values



# parallel computing
def parallel_compute(latlon_to_latlonIdx, latlonIdx_to_vicinityIdx):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # partial function with fixed arguments
        func = partial(process_latlon, latlon_to_latlonIdx=latlon_to_latlonIdx, latlonIdx_to_vicinityIdx=latlonIdx_to_vicinityIdx)
        results = pool.map(func, latlon_to_latlonIdx.keys())

    return dict(results)






# the main function is necessary for multiprocessing
if __name__ == '__main__':

    # read the grib file
    print("Start reading the grib file...")
    msgs = pygrib.open(r"C:\SUSTech\datasets_of_graduation_project\0220.grib")
    print("Complete reading the grib file.")
    
    
    
    LAT_RANGE = (90, 66.5) # the entire map is from 90 degree to 66.5 degree, closed interval
    LON_RANGE = (-180, 179.5)
    LAT_RESOL = 5 # the grid resolution of latitudes; two distinct lats are differred by 5 degrees
    LON_RESOL = 15
    LAT_VICINITY_R = 1 # Consider the average of the vicinity of each grid point, to reduce the noise. The radius of the vicinity is 1 degree here. This value should not exceed LAT_RESOL
    LON_VICINITY_R = 1
    DISTINCT_LATS = [90 - i * LAT_RESOL 
        for i in range(int((abs(LAT_RANGE[1] - LAT_RANGE[0]) // LAT_RESOL) + 1))] # round down when sampling points
    DISTINCT_LONS = [-180 + i * LON_RESOL 
        for i in range(int((abs(LON_RANGE[1] - LON_RANGE[0]) // LON_RESOL) + 1))]
    
    
    
    print("Start computing the bidicts...")
    import numpy as np
    from bidict import bidict
    msg = msgs[1]

    # (lat, lon) -> (lat_idx, lon_idx)
    latlon_to_latlonIdx = bidict()
    for lat in DISTINCT_LATS:
        for lon in DISTINCT_LONS:
            latlon_to_latlonIdx[(lat, lon)] = (np.where(msg["distinctLatitudes"] == lat)[0][0], 
                                        np.where(msg["distinctLongitudes"] == lon)[0][0])

    # (lat_idx, lon_idx) -> ((lat1_idx, lat2_idx), (lon1_idx, lon2_idx)), 
    # with lat1_idx <= lat2_idx, lon1_idx <= lon2_idx
    latlonIdx_to_vicinityIdx = bidict()
    for latlonIdx in latlon_to_latlonIdx.values():
        lat_idx, lon_idx = latlonIdx
        lat1_idx = int(max(0, lat_idx - LAT_VICINITY_R // 0.25))
        lat2_idx = int(min(len(msg["distinctLatitudes"]), lat_idx + LAT_VICINITY_R // 0.25 + 1))
        lon1_idx = int(max(0, lon_idx - LON_VICINITY_R // 0.25))
        lon2_idx = int(min(len(msg["distinctLongitudes"]), lon_idx + LON_VICINITY_R // 0.25 + 1))
        latlonIdx_to_vicinityIdx[latlonIdx] = ((lat1_idx, lat2_idx), (lon1_idx, lon2_idx))
    
    print("Complete computing the bidicts.")



    # parallel computing
    print("Start parallel computing...")
    ts_2mTemp = parallel_compute(latlon_to_latlonIdx, latlonIdx_to_vicinityIdx)
    print("Complete the parallel computing.")



    # save the ts_2mTemp as a h5 file
    print("Saving the ts_2mTemp...")
    import h5py
    path = r"outputs\ts_2mTemp_(5_30_1_1_para).h5"

    with h5py.File(path, "w") as f: # (LAT_RESOL, LON_RESOL, LAT_VICINITY_R, LON_VICINITY_R) = (5, 30, 1, 1)
        for key, value in ts_2mTemp.items():
            key_str = f"{key[0]}_{key[1]}"  # convert the keys ((lat, lon) tuples) to strings
            f.create_dataset(key_str, data=value)
            
    print("Complete the saving.")