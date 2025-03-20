import numpy as np
import multiprocessing as mp
import pygrib  # need to open the grib file within each process
from functools import partial



# compute for a single (lat, lon), (processes are independent)
def process_latlon(latlon, latlon_to_latlonIdx):
    lat, lon = latlon
    
    # need to open the grib file within each process
    msgs = pygrib.open(r"C:\SUSTech\datasets_of_graduation_project\0220.grib")

    lat_idx, lon_idx = latlon_to_latlonIdx[(lat, lon)]
    
    values = np.array([msg["values"][lat_idx, lon_idx] for msg in msgs])
    msgs.close()
    return (lat, lon), values



# parallel computing
def parallel_compute(latlon_to_latlonIdx):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        print(f"Start parallel computing with {mp.cpu_count()} processes...")
        # partial function with fixed arguments
        func = partial(process_latlon, latlon_to_latlonIdx=latlon_to_latlonIdx)
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
    LON_RESOL = 30
    
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
    
    print("Complete computing the bidicts.")



    # parallel computing
    print("Start parallel computing...")
    ts_2mTemp = parallel_compute(latlon_to_latlonIdx)
    print("Complete the parallel computing.")



    # save the ts_2mTemp as a h5 file
    print("Saving the ts_2mTemp...")
    import h5py
    path = r"outputs\ts_2mTemp_(5_30_noVicinity_para).h5"

    with h5py.File(path, "w") as f: # (LAT_RESOL, LON_RESOL) = (5, 30)
        for key, value in ts_2mTemp.items():
            key_str = f"{key[0]}_{key[1]}"  # convert the keys ((lat, lon) tuples) to strings
            f.create_dataset(key_str, data=value)
            
    print("Complete the saving.")