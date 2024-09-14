import torch
import numpy as np
import scipy.sparse as sp
import math
from easydict import EasyDict
from utils.metrics import masked_mae
from utils.serialization import load_pkl,load_adj
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from baselines.STID import STID
from baselines.STDCN import STDCN
from baselines.D2STGNN import D2STGNN
from baselines.DGCRN import DGCRN
from baselines.DCRNN import DCRNN
from baselines.GWNet import GraphWaveNet
from baselines.STGCN import STGCN
from baselines.STGODE import STGODE
from baselines.STGODE.generate_matrices import generate_dtw_spa_matrix
from baselines.STNorm import STNorm
from baselines.STWave import STWave
from baselines.MTGNN import MTGNN

CFG_MODEL = EasyDict()




# ======== model: D2STGNN =============== #
def D2STGNN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"D2STGNN_{ds_name}"
    MODEL.ARCH = D2STGNN
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "doubletransition")
    MODEL.PARAM = {
        "num_feat": 1,
        "num_hidden": 32,
        "dropout": 0.1,
        "seq_length": output_len,
        "k_t": 3,
        "k_s": 2,
        "gap": 3,
        "num_nodes": num_nodes,
        "adjs": [torch.tensor(adj) for adj in adj_mx],
        "num_layers": 5,
        "num_modalities": 2,
        "node_hidden": 10,
        "time_emb_dim": 10,
        "time_in_day_size": steps_per_day,
        "day_in_week_size": 7,
    }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.D2STGNN = D2STGNN_ARGS


# ======== model: DCRNN =============== #
def DCRNN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"DCRNN{ds_name}"
    MODEL.ARCH = DCRNN
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "doubletransition")
    MODEL.PARAM = {
        "cl_decay_steps": 2000,
        "horizon": output_len,
        "input_dim": 2,
        "max_diffusion_step": 2,
        "num_nodes": num_nodes,
        "num_rnn_layers": 2,
        "output_dim": 1,
        "rnn_units": 64,
        "seq_len": input_len,
        "adj_mx": [torch.tensor(i) for i in adj_mx],
        "use_curriculum_learning": True
    }
    MODEL.FORWARD_FEATURES = [0,1]
    return MODEL
CFG_MODEL.DCRNN = DCRNN_ARGS

# ======== model: DGCRN =============== #
def DGCRN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"DGCRN_{ds_name}"
    MODEL.ARCH = DGCRN
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "doubletransition")
    MODEL.PARAM = {
        "gcn_depth": 2,
        "num_nodes": num_nodes,
        "predefined_A": [torch.Tensor(_) for _ in adj_mx],
        "dropout": 0.3,
        "subgraph_size": 20,
        "node_dim": 40,
        "middle_dim": 2,
        "seq_length": input_len,
        "in_dim": 2,
        "list_weight": [0.05, 0.95, 0.95],
        "tanhalpha": 3,
        "cl_decay_steps": 4000,
        "rnn_size": 64,
        "hyperGNN_dim": 16
    }
    MODEL.FORWARD_FEATURES = [0,1]
    return MODEL
CFG_MODEL.DGCRN = DGCRN_ARGS




# ======== model: GraphWaveNet =============== #
def GraphWaveNet_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"GraphWaveNet_{ds_name}"
    MODEL.ARCH = GraphWaveNet
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "doubletransition")
    MODEL.PARAM = {
        "num_nodes": num_nodes,
        "supports": [torch.tensor(i) for i in adj_mx],
        "dropout": 0.3,
        "gcn_bool": True,
        "addaptadj": True,
        "aptinit": None,
        "in_dim": 2,
        "out_dim": output_len,
        "residual_channels": 16,
        "dilation_channels": 16,
        "skip_channels": 64,
        "end_channels": 128,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 2
    }
    MODEL.FORWARD_FEATURES = [0, 1]
    return MODEL
CFG_MODEL.GraphWaveNet = GraphWaveNet_ARGS



# ======== model: MTGNN =============== #
def MTGNN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    buildA_true = True
    if buildA_true: # self-learned adjacency matrix
        adj_mx = None
    else:           # use predefined adjacency matrix
        _, adj_mx = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "doubletransition")
        adj_mx = torch.tensor(adj_mx)-torch.eye(num_nodes)

    MODEL = EasyDict()
    MODEL.NAME = f"MTGNN_{ds_name}"
    MODEL.ARCH = MTGNN
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "gcn_true"  : True,
        "buildA_true": buildA_true,
        "gcn_depth": 2,
        "num_nodes": num_nodes,
        "predefined_A":adj_mx,
        "dropout":0.3,
        "subgraph_size":20,
        "node_dim":40,
        "dilation_exponential":1,
        "conv_channels":32,
        "residual_channels":32,
        "skip_channels":64,
        "end_channels":128,
        "seq_length":input_len,
        "in_dim":2,
        "out_dim":output_len,
        "layers":3,
        "propalpha":0.05,
        "tanhalpha":3,
        "layer_norm_affline":True
    }
    MODEL.FORWARD_FEATURES = [0,1]
    return MODEL
CFG_MODEL.MTGNN = MTGNN_ARGS




# ======== model: STGCN =============== #
def STGCN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STGCN_{ds_name}"
    MODEL.ARCH = STGCN
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name + "/adj_mx.pkl", "normlap")
    adj_mx = torch.Tensor(adj_mx[0])
    MODEL.PARAM = {
        "Ks" : 3, 
        "Kt" : 3,
        "blocks" : [[1], [64, 16, 64], [64, 16, 64], [128, 128], [output_len]],
        "T" : input_len,
        "n_vertex" : num_nodes,
        "act_func" : "glu",
        "graph_conv_type" : "cheb_graph_conv",
        "gso" : adj_mx,
        "bias": True,
        "droprate" : 0.5
    }
    MODEL.FORWARD_FEATURES = [0]
    return MODEL
CFG_MODEL.STGCN = STGCN_ARGS




# ======== model: STGODE =============== #
def STGODE_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STGODE_{ds_name}"
    MODEL.ARCH = STGODE
    MODEL.LOSS = masked_mae
    A_se_wave, A_sp_wave = generate_dtw_spa_matrix(ds_name, input_len, output_len)
    MODEL.PARAM = {
        "num_nodes": num_nodes,
        "num_features": 3,
        "num_timesteps_input": input_len,
        "num_timesteps_output": output_len,
        "A_sp_hat" : A_sp_wave,
        "A_se_hat" : A_se_wave
    }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.STGODE = STGODE_ARGS



# ======== model: STID =============== #
def STID_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STID_{ds_name}"
    MODEL.ARCH = STID
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "num_nodes":num_nodes,
        "input_len": input_len,
        "input_dim": 1,
        "embed_dim": 16,
        "output_len": output_len,
        "num_layer": 3,
        "if_node": True,
        "node_dim": 16,
        "if_T_i_D": True,
        "if_D_i_W": True,
        "temp_dim_tid": 16,
        "temp_dim_diw": 16,
        "time_of_day_size": steps_per_day,
        "day_of_week_size": 7,
        "type":"FC",                # [FC,CONV1D,CONV2D]
        "concat":True
    }
    return MODEL
CFG_MODEL.STID = STID_ARGS


from baselines.PDFormer import PDFormer
import pickle
# ======== model: PDFormer =============== #
def PDFormer_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"PDFormer_{ds_name}"
    MODEL.ARCH = PDFormer
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        'input_window': input_len,
        'output_window': output_len,
        'add_time_in_day': True,
        'add_day_in_week': True,
        'step_size': 2776,
        'far_mask_delta': 7,
        'geo_num_heads': 4,
        'sem_num_heads': 2,
        't_num_heads': 2,
        'type_ln': 'pre',
        'huber_delta': 2,
        'embed_dim': 64,
        'skip_dim': 128,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'drop': 0,
        'attn_drop': 0,
        'drop_path': 0.3,
        's_attn_size': 3,
        't_attn_size': 1,
        'enc_depth': 4,
        'type_short_path': 'hop',
        'task_level': 0,
        'use_curriculum_learning': True,
        'quan_delta': 0.25,
        'dtw_delta': 5,
        'lape_dim': 8,
        'output_dim': 1,
        'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')}
    
    pattern_key_file = f'datasets/{ds_name}/pattern_mx.npy'
    cache_path = f'datasets/{ds_name}/dtw_mx.npy'


    pattern_keys = np.load(pattern_key_file)
    dtw_matrix = np.load(cache_path)
    
    if ds_name in ["PEMS-BAY","METR-LA"]:
        adj_mx = load_pkl(f"datasets/{ds_name}/adj_mx.pkl")[2]
    else:
        adj_mx = load_pkl(f"datasets/{ds_name}/adj_mx.pkl")


    sd_mx = None
    sh_mx = np.load(f'datasets/{ds_name}/sh_mx.npy')

    output_dim = 1
    feature_dim = 3
    ext_dim = feature_dim - output_dim

    MODEL.PARAM.update({"adj_mx": adj_mx, "sd_mx": sd_mx, "sh_mx": sh_mx,
                    "ext_dim": ext_dim, "num_nodes": num_nodes, "feature_dim": feature_dim,
                    "output_dim": output_dim, "dtw_matrix": dtw_matrix, "pattern_keys": pattern_keys})

    return MODEL
CFG_MODEL.PDFormer = PDFormer_ARGS


# ======== model: STDCN =============== #
def STDCN_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STDCN_{ds_name}"
    MODEL.ARCH = STDCN
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "num_nodes":num_nodes,
        "d_model": 64,
        "input_len": input_len,
        "output_len": output_len,
        "num_layer": 6,
        "if_node": True,
        "if_T_i_D": True,
        "if_D_i_W": True,
        "time_of_day_size": steps_per_day,
        "day_of_week_size": 7,
    }
    return MODEL
CFG_MODEL.STDCN = STDCN_ARGS




# ======== model: STNorm =============== #
def STNorm_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    MODEL = EasyDict()
    MODEL.NAME = f"STNorm_{ds_name}"
    MODEL.ARCH = STNorm
    MODEL.LOSS = masked_mae
    MODEL.PARAM = {
        "num_nodes" : num_nodes,
        "tnorm_bool": True,
        "snorm_bool": True,
        "in_dim"    : 2,
        "out_dim"   : output_len,
        "channels"  : 32,
        "kernel_size": 2,
        "blocks"    : 4,
        "layers"    : 2,
    }
    MODEL.FORWARD_FEATURES = [0,1]
    return MODEL
CFG_MODEL.STNorm = STNorm_ARGS




# ======== model: STWave =============== #
def STWave_ARGS(ds_name,num_nodes, input_len, output_len, rescale, steps_per_day):
    def laplacian(W):
        """Return the Laplacian of the weight matrix."""
        # Degree matrix.
        d = W.sum(axis=0)
        # Laplacian matrix.
        d = 1 / np.sqrt(d)
        D = sp.diags(d, 0)
        I = sp.identity(d.size, dtype=W.dtype)
        L = I - D * W * D
        return L

    def largest_k_lamb(L, k):
        lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
        return (lamb, U)

    def get_eigv(adj,k):
        L = laplacian(adj)
        eig = largest_k_lamb(L,k)
        return eig

    def loadGraph(adj_mx, hs, ls):
        graphwave = get_eigv(adj_mx+np.eye(adj_mx.shape[0]), hs)
        sampled_nodes_number = int(np.around(math.log(adj_mx.shape[0]))+2)*ls
        graph = csr_matrix(adj_mx)
        dist_matrix = dijkstra(csgraph=graph)
        dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
        adj_gat = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]
        return adj_gat, graphwave

    MODEL = EasyDict()
    MODEL.NAME = f"STWave_{ds_name}"
    MODEL.ARCH = STWave
    MODEL.LOSS = masked_mae
    adj_mx, _ = load_adj("datasets/" + ds_name +  "/adj_mx.pkl", "original")
    adjgat, gwv = loadGraph(_, 128, 1)
    MODEL.PARAM = {
        "input_dim": 1,
        "hidden_size": 128,
        "layers": 2,
        "seq_len": input_len,
        "horizon": output_len,
        "log_samples": 1,
        "adj_gat": adjgat,
        "graphwave": gwv,
        "time_in_day_size": steps_per_day,
        "day_in_week_size": 7,
        "wave_type": "coif1",
        "wave_levels": 2,
    }
    MODEL.FORWARD_FEATURES = [0,1,2]
    return MODEL
CFG_MODEL.STWave = STWave_ARGS
