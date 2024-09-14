import torch
from torch import nn
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class DataEmbedding(nn.Module):
    def __init__(self, d_model, num_nodes=None, time_of_day_size=None, day_of_week_size=None, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.num_nodes = num_nodes

        self.value_emb = nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=(1,1), bias=True)
        if self.num_nodes is not None:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, d_model))
            nn.init.xavier_uniform_(self.node_emb)
        if self.time_of_day_size is not None:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, d_model))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.day_of_week_size is not None:
            self.day_in_week_emb = nn.Parameter( torch.empty(self.day_of_week_size, d_model))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        
    def forward(self, history):

        bs,in_len,_,_ = history.shape
        x = self.value_emb(history[...,[0]].permute(0,3,1,2)).permute(0,2,3,1)

        if self.time_of_day_size is not None:
            x += self.time_in_day_emb[(history[:,:,0,1]*self.time_of_day_size).long()].unsqueeze(2).expand(-1,-1,self.num_nodes,-1)

        if self.day_of_week_size is not None:
            x += self.day_in_week_emb[(history[:,:,0,2]*self.day_of_week_size).long()].unsqueeze(2).expand(-1,-1,self.num_nodes,-1)

        if self.num_nodes is not None:
            x += self.node_emb.unsqueeze(0).unsqueeze(1).expand(bs,in_len,-1,-1)



        return x.permute(0,3,1,2)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(3, 1), bias=True, padding='same')
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 1), bias=True, padding='same')
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden

# class STDCN(nn.Module):
#     def __init__(self, **model_args):
#         super().__init__()
#         # attributes

#         self.d_model = model_args["d_model"]
#         self.num_nodes = model_args["num_nodes"]
#         self.input_len = model_args["input_len"]
#         self.output_len = model_args["output_len"]
#         self.num_layer = model_args["num_layer"]
#         self.time_of_day_size = model_args["time_of_day_size"]
#         self.day_of_week_size = model_args["day_of_week_size"]

#         self.if_time_in_day = model_args["if_T_i_D"]
#         self.if_day_in_week = model_args["if_D_i_W"]
#         self.if_spatial = model_args["if_node"]
        
#         self.emb = DataEmbedding(self.d_model,self.num_nodes,self.time_of_day_size,self.day_of_week_size)        
#         self.block = nn.ModuleList()
#         for i in range(self.num_layer):
#             self.block.append(MultiLayerPerceptron(self.d_model, self.d_model))
#         self.end_conv1 = nn.Conv2d(in_channels=self.d_model, out_channels=1, kernel_size=(1,1), bias=True)
#         self.end_conv2 = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1,1), bias=True)
        


#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         x = self.emb(history_data)
#         for i in range(self.num_layer):
#             x = self.block[i](x)
        
#         out1 = self.end_conv1(x)
#         prediction = self.end_conv2(out1.permute(0,2,3,1))
#         return prediction

class STDCN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes

        self.d_model = model_args["d_model"]
        self.num_nodes = model_args["num_nodes"]
        self.input_len = model_args["input_len"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        
        self.decom = series_decomp(1)
        self.emb_trend = DataEmbedding(self.d_model,self.num_nodes,self.time_of_day_size,self.day_of_week_size)        
        self.emb_season = DataEmbedding(self.d_model,self.num_nodes,self.time_of_day_size,self.day_of_week_size)   
        self.block_trend = nn.ModuleList()
        self.block_season = nn.ModuleList()
        for i in range(self.num_layer):
            self.block_trend.append(MultiLayerPerceptron(self.d_model, self.d_model))
            self.block_season.append(MultiLayerPerceptron(self.d_model, self.d_model))
        self.end_conv1 = nn.Conv2d(in_channels=self.d_model, out_channels=1, kernel_size=(1,1), bias=True)
        self.end_conv2 = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1,1), bias=True)
        


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        season, trend = self.decom(history_data[...,0])
        season_emb = self.emb_season(torch.cat([season.unsqueeze(-1), history_data[...,1:]], axis=3))
        trend_emb = self.emb_trend(torch.cat([trend.unsqueeze(-1), history_data[...,1:]], axis=3))
        for i in range(self.num_layer):
            season_emb = self.block_season[i](season_emb)
            trend_emb = self.block_trend[i](trend_emb)

        
        out1 = self.end_conv1(season_emb + trend_emb)
        prediction = self.end_conv2(out1.permute(0,2,3,1))
        return prediction

# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         x = x.permute(0,3,1,2)
#         front = x[:,:,:,[0]].repeat(1,1,1,(self.kernel_size - 1) // 2)
#         end = x[:,:,:,[-1]].repeat(1,1,1,(self.kernel_size - 1) // 2)
#         x = torch.cat([front,x,end], dim=3)


#         upper = x[:,:,[0],:].repeat(1,1,(self.kernel_size - 1) // 2,1)
#         lower = x[:,:,[-1],:].repeat(1,1,(self.kernel_size - 1) // 2,1)
#         x = torch.cat([upper,x,lower],dim=2)

#         x = self.avg(x)
#         x = x.permute(0, 2, 3, 1)
#         return x
    
# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean
    
# class DataEmbedding(nn.Module):
#     def __init__(self, d_model, num_nodes=None, time_of_day_size=None, day_of_week_size=None, dropout=0.1):
#         super(DataEmbedding, self).__init__()
#         self.time_of_day_size = time_of_day_size
#         self.day_of_week_size = day_of_week_size
#         self.num_nodes = num_nodes

#         self.value_emb = nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=(1,1), bias=True)
#         if self.num_nodes is not None:
#             self.node_emb = nn.Parameter(torch.empty(self.num_nodes, d_model))
#             nn.init.xavier_uniform_(self.node_emb)
#         if self.time_of_day_size is not None:
#             self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, d_model))
#             nn.init.xavier_uniform_(self.time_in_day_emb)
#         if self.day_of_week_size is not None:
#             self.day_in_week_emb = nn.Parameter( torch.empty(self.day_of_week_size, d_model))
#             nn.init.xavier_uniform_(self.day_in_week_emb)
        
#     def forward(self, history):
#         bs,in_len,_,_ = history.shape
#         x = self.value_emb(history[...,[0]].permute(0,3,1,2)).permute(0,2,3,1)

#         if self.time_of_day_size is not None:
#             x += self.time_in_day_emb[(history[:,:,0,1]*self.time_of_day_size).long()].unsqueeze(2).expand(-1,-1,self.num_nodes,-1)

#         if self.day_of_week_size is not None:
#             x += self.day_in_week_emb[(history[:,:,0,2]*self.day_of_week_size).long()].unsqueeze(2).expand(-1,-1,self.num_nodes,-1)

#         if self.num_nodes is not None:
#             x += self.node_emb.unsqueeze(0).unsqueeze(1).expand(bs,in_len,-1,-1)

#         return x.permute(0,3,1,2)

# class MultiLayerPerceptron(nn.Module):
#     def __init__(self, input_dim, hidden_dim) -> None:
#         super().__init__()
#         self.fc1 = nn.Conv2d(
#             in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(3, 1), bias=True, padding='same')
#         self.fc2 = nn.Conv2d(
#             in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 1), bias=True, padding='same')
#         self.act = nn.ReLU()
#         self.drop = nn.Dropout(p=0.15)

#     def forward(self, input_data: torch.Tensor) -> torch.Tensor:
#         hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
#         hidden = hidden + input_data                           # residual
#         return hidden

# class STDCN(nn.Module):
#     def __init__(self, **model_args):
#         super().__init__()
#         # attributes

#         self.d_model = model_args["d_model"]
#         self.num_nodes = model_args["num_nodes"]
#         self.input_len = model_args["input_len"]
#         self.output_len = model_args["output_len"]
#         self.num_layer = model_args["num_layer"]
#         self.time_of_day_size = model_args["time_of_day_size"]
#         self.day_of_week_size = model_args["day_of_week_size"]

#         self.if_time_in_day = model_args["if_T_i_D"]
#         self.if_day_in_week = model_args["if_D_i_W"]
#         self.if_spatial = model_args["if_node"]
        
#         self.decom = series_decomp(5)
#         self.emb = DataEmbedding(self.d_model,self.num_nodes,self.time_of_day_size,self.day_of_week_size)        
 
#         self.block = nn.ModuleList()
#         for i in range(self.num_layer):
#             self.block.append(MultiLayerPerceptron(self.d_model, self.d_model))
#         self.end_conv1 = nn.Conv2d(in_channels=self.d_model, out_channels=1, kernel_size=(1,1), bias=True)
#         self.end_conv2 = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1,1), bias=True)
        


#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         x = self.emb(history_data)
#         residual,trend = self.decom(x)
#         for i in range(self.num_layer):
#             hidden = self.block[i](trend)
#             senson, trend = self.decom(hidden)
#             residual += senson

        
#         out1 = self.end_conv1(residual+trend)
#         prediction = self.end_conv2(out1.permute(0,2,3,1))
#         return prediction