import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from fastdtw import fastdtw
from tslearn.clustering import TimeSeriesKMeans, KShape
from datetime import datetime
from utils.serialization import load_pkl
from config.cfg_general import CFG_GENERAL
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def preprocess(dataset_name):
    ds_args = {
        "PEMS04":307,
        "PEMS08":170,
        "PEMS-BAY":325,
        "METR-LA":207
    }
    num_nodes = ds_args[dataset_name]
    points_per_hour = 12
    points_per_day = 288
    output_dim = 1


    if dataset_name in ["PEMS04","PEMS08"]:
        df = np.load(f"datasets/raw_data/{dataset_name}/{dataset_name}.npz")['data']
        df = df[...,[0]]
        adj_mx = load_pkl(f"datasets/raw_data/{dataset_name}/adj_{dataset_name}.pkl")
    else:
        df = pd.read_hdf(f"datasets/raw_data/{dataset_name}/{dataset_name}.h5")
        df = np.expand_dims(df.values, axis=-1)
        adj_mx = load_pkl(f"datasets/raw_data/{dataset_name}/adj_{dataset_name}.pkl")[2]

    num_samples = df.shape[0] - (CFG_GENERAL.DATASET.HISTORY_SEQ_LEN + CFG_GENERAL.DATASET.FUTURE_SEQ_LEN) + 1
    train_num = round(num_samples * CFG_GENERAL.DATASET.TRAIN_RATIO)
    data_train = df[:train_num+CFG_GENERAL.DATASET.HISTORY_SEQ_LEN-1]
    mean, std = data_train[..., 0].mean(), data_train[..., 0].std()
    data_norm = (df - mean)/std

    x_list = []
    for t in range(CFG_GENERAL.DATASET.HISTORY_SEQ_LEN, num_samples + CFG_GENERAL.DATASET.HISTORY_SEQ_LEN):
        x_list.append(data_norm[t-CFG_GENERAL.DATASET.HISTORY_SEQ_LEN:t])

    x_train = np.stack(x_list,axis=0)[:train_num]

    print(f"{datetime.now()}: {dataset_name}开始运行 {adj_mx.shape} {df.shape} {x_train.shape}")


    sd_mx = None
    type_short_path = 'hop'
    sh_path = '{}_sh_mx.npy'.format(dataset_name)
    if not os.path.exists(sh_path):
        sh_mx = adj_mx.copy()
        if type_short_path == 'hop':
            sh_mx[sh_mx > 0] = 1
            sh_mx[sh_mx == 0] = 511
            for i in range(num_nodes):
                sh_mx[i, i] = 0
            for k in range(num_nodes):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j], 511)
            np.save(sh_path, sh_mx)

    sh_mx = np.load(sh_path)
    print(f"{datetime.now()}: {dataset_name}生成shmx")

    cache_path = f'{dataset_name}_dtw.npy'

    if not os.path.exists(cache_path):
        data_mean = np.mean(
            [df[24 * points_per_hour * i: 24 * points_per_hour * (i + 1)]
                for i in range(df.shape[0] // (24 * points_per_hour))], axis=0)
        dtw_distance = np.zeros((num_nodes, num_nodes))
        for i in tqdm(range(num_nodes)):
            for j in range(i, num_nodes):
                dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
        for i in range(num_nodes):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(cache_path, dtw_distance)
    dtw_matrix = np.load(cache_path)
    print(f"{datetime.now()}: {dataset_name}生成dtw")


    cluster_method = 'kshape'
    dataset = dataset_name
    cand_key_days = 21
    s_attn_size = 3
    n_cluster = 16
    cluster_max_iter = 5

    pattern_key_file = os.path.join(
        './', 'pattern_keys_{}_{}_{}_{}_{}_{}'.format(
            cluster_method, dataset, cand_key_days, s_attn_size, n_cluster, cluster_max_iter))

    if not os.path.exists(pattern_key_file + '.npy'):
        cand_key_time_steps = cand_key_days * points_per_day
        pattern_cand_keys = x_train[:cand_key_time_steps, :s_attn_size, :, :output_dim].swapaxes(1, 2).reshape(-1, s_attn_size, output_dim)
        if cluster_method == "kshape":
            km = KShape(n_clusters=n_cluster, max_iter=cluster_max_iter).fit(pattern_cand_keys)
        else:
            km = TimeSeriesKMeans(n_clusters=n_cluster, metric="softdtw", max_iter=cluster_max_iter).fit(pattern_cand_keys)
        pattern_keys = km.cluster_centers_
        np.save(pattern_key_file, pattern_keys)
    else:
        pattern_keys = np.load(pattern_key_file + ".npy")
    print(f"{datetime.now()}: {dataset_name}生成pattern keys")


for data in ["PEMS-BAY","METR-LA","PEMS08"]:
    preprocess(data)