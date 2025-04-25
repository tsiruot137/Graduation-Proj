import pygrib
import numpy as np
import cupy as cp  # 引入 CuPy、
from tqdm import tqdm
import h5py

# 地图分辨率
DELTA_LAT = 0.5
DELTA_LON = 2.5
N_DISTINCT_LATS = int((90 - 66.5) / DELTA_LAT)
N_DISTINCT_LONS = int((180 - (-180)) / DELTA_LON)

if __name__ == '__main__':
    
    
    # 读取 grib 文件
    print("Start reading the grib file...")
    msgs = pygrib.open(r"C:\SUSTech\datasets_of_graduation_project\0220.grib")
    print("Complete reading the grib file.")
    
    
    # 计算地图的纬度和经度索引
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
    
    
    # 计算时间序列长度
    print("Computing the length of the time series...")
    msgs.rewind()
    TS_LENGTH = sum(1 for _ in msgs)
    print("The length of the time series: %d" % TS_LENGTH)
    
    TS_2MTEMP_SHAPE = (TS_LENGTH, N_DISTINCT_LATS, N_DISTINCT_LONS)
    ts_2mtemp_gpu = cp.zeros(TS_2MTEMP_SHAPE, dtype=cp.float32) # 将 ts_2mtemp 放到 GPU 上
    print("The shape of the time series of 2m temperature: (%d, %d, %d)" % TS_2MTEMP_SHAPE)
    
    
    # 计算 2m 温度的时间序列
    msgs.rewind()
    print("Start computing ts_2mtemp...")
    timestamps = []
    for msg_idx, msg in tqdm(enumerate(msgs), total=TS_LENGTH, desc="Processing GRIB messages", unit="msg"):
        timestamps.append(f"{msg['year']:04d}_{msg['month']:02d}_{msg['day']:02d}_{msg['hour']:02d}")
        values_gpu = cp.asarray(msg["values"])
        ts_2mtemp_gpu[msg_idx] = values_gpu[cp.ix_(lat_indices, lon_indices)]

    print("Finish computing the time series of 2m temperature.")
    
    
    # 数据存储
    DATA_2MTEMP = {
        "timestamps": timestamps, 
        "latitudes": sampled_lats.get(),
        "longitudes": sampled_lons.get(),
        "data_2mtemp": ts_2mtemp_gpu.get()
    }
    
    filename = r"C:\SUSTech\datasets_of_graduation_project\big_outputs\data_2mtemp.h5"

    with h5py.File(filename, 'w') as f:
        # 保存时间戳、纬度、经度和温度数据
        f.create_dataset('timestamps', data=DATA_2MTEMP['timestamps'])
        f.create_dataset('latitudes', data=DATA_2MTEMP['latitudes'])
        f.create_dataset('longitudes', data=DATA_2MTEMP['longitudes'])
        f.create_dataset('data_2mtemp', data=DATA_2MTEMP['data_2mtemp'])

    print(f"Data saved to {filename}")
    
    
        
        
    
    
    
    
    