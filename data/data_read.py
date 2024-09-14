import pandas as pd
import numpy as np
import os
def read_water_quality(data_file_path,target_channel):
    start_time = '2023-03-21 01:00'
    end_time = "2024-03-20 00:00"
    data = pd.DataFrame(index=pd.date_range(start=start_time, end=end_time, freq=pd.Timedelta('15min')))

    for root,dirs,files in os.walk(data_file_path):
        for filename in files:
            df = pd.read_csv(os.path.join(root, filename),sep="\t",header=None,index_col=2)[[4]]
            df.index = pd.to_datetime(df.index)
            df.columns=[filename.split(".")[0]]
            data = data.join(df)

    data = data[~data.index.duplicated()].fillna(method='ffill')
    return np.expand_dims(data.values, axis=-1),data.index

def read_h5(data_file_path,target_channel):
    df = pd.read_hdf(data_file_path)
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., target_channel]
    return data,None

def read_npz(data_file_path,target_channel):
    data = np.load(data_file_path)["data"]
    data = data[..., target_channel]
    return data,None

def read_grid(data_file_path):
    grid_file_path = data_file_path
    geo_file_path = grid_file_path.replace("grid","geo")
    data_col = ["inflow", "outflow"]
    geofile = pd.read_csv(geo_file_path)
    geo_ids = list(geofile['geo_id'])
    gridfile = pd.read_csv(grid_file_path)
    if data_col != '':
        data_col.insert(0, 'time')
        data_col.insert(1, 'row_id')
        data_col.insert(2, 'column_id')
        gridfile = gridfile[data_col]

    timesolts = gridfile['time'][:int(gridfile.shape[0] / len(geo_ids))]
    timesolts = timesolts.map(lambda x: x.replace('T', ' ').replace('Z', ''))
    timesolts = timesolts.astype("datetime64")

    df = gridfile[gridfile.columns[-2:]]
    len_time = len(timesolts)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i + len_time].values)
    data = np.array(data, dtype=np.float)
    data = data.swapaxes(0, 1)
    return data,timesolts

def read_grid_inflow(data_file_path,target_channel):
    data,timeslots = read_grid(data_file_path)
    return data[...,[0]], timeslots

def read_grid_outflow(data_file_path,target_channel):
    data,timeslots = read_grid(data_file_path)
    return data[...,[1]], timeslots



