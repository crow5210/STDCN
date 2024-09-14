# STDCN: Spatial Temporal Decomposed Convective Networks

## Introduction
This is a PyTorch implementation of STDCN which is designed for the Spatial Temporal Forecasting (STF) task. 

> \* Equal Contributions.


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pdformer-propagation-delay-aware-dynamic-long/traffic-prediction-on-pemsd4)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd4?p=pdformer-propagation-delay-aware-dynamic-long) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pdformer-propagation-delay-aware-dynamic-long/traffic-prediction-on-pemsd7)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7?p=pdformer-propagation-delay-aware-dynamic-long) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pdformer-propagation-delay-aware-dynamic-long/traffic-prediction-on-pemsd8)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd8?p=pdformer-propagation-delay-aware-dynamic-long) 


## Requirements
Our code is based on Python version 3.11.5 and PyTorch version 2.0.0. Please make sure you have installed Python and PyTorch correctly. 


## Datasets  
SIDLinear selects a diverse range of real-world datasets from various domains, ensuring a wide coverage of scenarios for STF task.

data                | Variates      | Timesteps | Granularity
--------------------|---------------|-----------|------------
PEMS08              |170            |548        |17856 
PEMS04              |307            |680        |16992 
PEMS07              |883            |866        |28224 
CHIBike             |270 (15x18)    |1966        |4416 
T-Drive             |1024 (32x32)   |7812        |3600

### Datasets Downloading

The raw data can be downloaded at this [Baidu Yun](https://pan.baidu.com/s/1trrqN3lUZG7l-7H6FsMofw)(password: xljf), and should be unzipped to datasets/raw_data/.

> Note that PDFormer would calculate a DTW matrix and a traffic pattern set for each dataset, which is time-consuming. 

## Train & evaluate
You can train and evaluate **STDCN** through the following commands. 
```shell
python runner/evaluation.py  --out_dir /root/autodl-tmp/checkpoints --epoch 200
```
**Note**: By default the result recorded in the experiment log is the average of the first n steps. If you need to get the results of each step separately, please modify this parameter **CFG_GENERAL.TEST.HORIZON= [0,1,2,3,4,5,6,7,8,9,10,11]**

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
