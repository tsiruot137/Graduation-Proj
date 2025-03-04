import numpy as np
import multiprocessing as mp
import pygrib  # 需要在每个进程内部打开 GRIB 文件
from functools import partial

# 计算单个点的函数（子进程独立执行）
def process_latlon(latlon, latlon_to_latlonIdx, latlonIdx_to_vicinityIdx):
    lat, lon = latlon
    msgs = pygrib.open(r"C:\SUSTech\datasets_of_graduation_project\0220.grib")

    lat1_idx, lat2_idx = latlonIdx_to_vicinityIdx[latlon_to_latlonIdx[(lat, lon)]][0]
    lon1_idx, lon2_idx = latlonIdx_to_vicinityIdx[latlon_to_latlonIdx[(lat, lon)]][1]
    
    values = np.array([np.mean(msg["values"][lat1_idx:lat2_idx, lon1_idx:lon2_idx]) for msg in msgs])
    
    msgs.close()
    return (lat, lon), values

# 并行计算
def parallel_compute(latlon_to_latlonIdx, latlonIdx_to_vicinityIdx):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # 用 partial 绑定额外参数
        func = partial(process_latlon, latlon_to_latlonIdx=latlon_to_latlonIdx, latlonIdx_to_vicinityIdx=latlonIdx_to_vicinityIdx)
        results = pool.map(func, latlon_to_latlonIdx.keys())

    return dict(results)

# 主程序部分
if __name__ == '__main__':



    print("Start reading the grib file...")
    msgs = pygrib.open(r"C:\SUSTech\datasets_of_graduation_project\0220.grib")
    LAT_RANGE = (90, 66.5) # the entire map is from 90 degree to 66.5 degree, closed interval
    LON_RANGE = (-180, 179.5)

    LAT_RESOL = 5 # the grid resolution of latitudes; two distinct lats are differred by 5 degrees
    LON_RESOL = 30
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
        lat1_idx = int(max(0, 
                    lat_idx - LAT_VICINITY_R // 0.25))
        lat2_idx = int(min(len(msg["distinctLatitudes"]), 
                    lat_idx + LAT_VICINITY_R // 0.25 + 1))
        lon1_idx = int(max(0, 
                    lon_idx - LON_VICINITY_R // 0.25))
        lon2_idx = int(min(len(msg["distinctLongitudes"]), 
                    lon_idx + LON_VICINITY_R // 0.25 + 1))
        latlonIdx_to_vicinityIdx[latlonIdx] = ((lat1_idx, lat2_idx), (lon1_idx, lon2_idx))
    
    



    # 执行并行计算
    print("Start parallel computing...")
    ts_2mTemp = parallel_compute(latlon_to_latlonIdx, latlonIdx_to_vicinityIdx)
    print("Complete the parallel computing.")






    # 保存结果
    print("Saving the ts_2mTemp...")
    import h5py
    path = r"outputs\ts_2mTemp_(5_30_1_1_para).h5"

    with h5py.File(path, "w") as f: # (LAT_RESOL, LON_RESOL, LAT_VICINITY_R, LON_VICINITY_R) = (5, 30, 1, 1)
        for key, value in ts_2mTemp.items():
            key_str = f"{key[0]}_{key[1]}"  # convert the keys ((lat, lon) tuples) to strings
            f.create_dataset(key_str, data=value)
            
    print("Complete the saving.")





    print("Plotting the results...")
    
    # read the data back
    with h5py.File(path, "r") as f:
        ts_2mTemp_read = {}
        for key in f.keys():
            key_parts = key.split('_')
            key_tuple = (int(key_parts[0]), int(key_parts[1]))
            ts_2mTemp_read[key_tuple] = np.array(f[key])

    # check if the data is saved correctly and read correctly
    print(ts_2mTemp_read.keys() == ts_2mTemp.keys())
    print(all([np.array_equal(ts_2mTemp_read[key], ts_2mTemp[key]) for key in ts_2mTemp.keys()]))

    N = 8*15*12 # 8 hours per day, 15 days per month, 12 months per year
    N_years = 80 # 1945, 1942, ..., 2024, 80 years in total

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(DISTINCT_LONS), len(DISTINCT_LATS), figsize=(15, 40))
    for i, lon in enumerate(DISTINCT_LONS):
        for j, lat in enumerate(DISTINCT_LATS):
            # maximize the temp over each year 
            ts_annualMax_2mTemp = np.array([ts_2mTemp[(lat, lon)][y*N:(y+1)*N].max() 
                                                for y in range(0, N_years-1)])
            # get the 90th percentile of the target temp over each year 
            ts_annual90Percentile_2mTemp = np.array([np.percentile(ts_2mTemp[(lat, lon)][y*N:(y+1)*N], 90) 
                                                for y in range(0, N_years-1)])
            ax[i, j].plot(ts_annualMax_2mTemp, label="Max")
            ax[i, j].plot(ts_annual90Percentile_2mTemp, label="90th percentile")
            # ax[i, j].legend()
            ax[i, j].set_title(f"({lat}, {lon})")
            
    plt.show()
    fig.savefig(r"outputs\ts_2mTemp_(5_30_1_1_para).png")









    






        
        
        
        
