## Introduction

STDCN is designed for the Spatial Temporal Forecasting (STF) task. 



## Baselines
SIDLinear surveys a diverse range of Time Series Forecasting methods by different backbone models and the table below provides an overview of these methods along with their references.

Method        |Type | Category                | Link
--------------|-----|-------------------------|----
STGCN         |STF  |Prior-Graph-based,2018   |https://arxiv.org/abs/1709.04875
DCRNN         |STF  |Prior-Graph-based,2018   |https://arxiv.org/abs/1707.01926
GraphWaveNet  |STF  |Prior-Graph-based,2019   |https://arxiv.org/abs/1906.00121
DGCRN         |STF  |Prior-Graph-based,2023   |https://arxiv.org/abs/2104.14917
D2STGNN       |STF  |Prior-Graph-based,2022   |https://arxiv.org/abs/2206.09112
AGCRN         |STF  |Latent-Graph-based,2020  |https://arxiv.org/abs/2007.02842
MTGNN         |STF  |Latent-Graph-based,2020  |https://arxiv.org/abs/2005.11650
StemGNN       |STF  |Latent-Graph-based,2020  |https://arxiv.org/abs/2103.07719
GTS           |STF  |Latent-Graph-based       |https://arxiv.org/abs/2101.06861
STNorm        |STF  |Non-Graph-based          |https://dl.acm.org/doi/10.1145/3447548.3467330
STID          |STF  |Non-Graph-based          |https://arxiv.org/abs/2208.05233
STGODE        |STF  |todo                     |https://arxiv.org/abs/2106.12931
STWave        |STF  |todo                     |https://ieeexplore.ieee.org/document/10184591



## Datasets  
SIDLinear selects a diverse range of real-world datasets from various domains, ensuring a wide coverage of scenarios for STF task.

data                | Variates  | Timesteps | Granularity
--------------------|-----------|-----------|------------
METR-LA             |207        |1722       |34272 
PEMS-BAY            |325        |2694       |52116 
PEMS08              |170        |548        |17856 
PEMS04              |307        |680        |16992 

### Datasets Downloading

The raw data can be downloaded at this [Baidu Yun](https://pan.baidu.com/s/12i-U0C5zcRpd_izYtdQF7w)(password: t6ps), and should be unzipped to datasets/raw_data/.


## Running
```shell
python runner/evaluation.py  --out_dir /root/autodl-tmp/checkpoints --epoch 50
```


***
> todo:
>1. generate experiment report               -> done
>2. generate ranking graph and polar graph   -> done
>3. generate latex table code                -> done
>3. add domain adaptation experiment         -> done 
>4. evaluate on different horizons           -> done
>5. visualize training loss by tensorboard  -> done
>6. add tool for visualizing dataset         -> done
>7. add more metrics, such as dtw,dilate     -> done
>8. add experiment on the effects of different loss function    -> done
>9. upload datasets and checkpoints to Badiduyun for reproduce - > done
