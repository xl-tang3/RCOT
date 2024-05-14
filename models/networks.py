import os,sys
sys.path.append("../")
import torch
import torch.nn as nn
from torch.nn import Conv1d,Conv2d
from knn_cuda import KNN
from models.pointnet2_utils import grouping_operation
import torch.nn.functional as F
from torch.autograd import Variable

class get_edge_feature(nn.Module):
    """construct edge feature for each point
    Args:
        tensor: input a point cloud tensor,batch_size,num_dims,num_points
        k: int
    Returns:
        edge features: (batch_size,num_dims,num_points,k)
    """
    def __init__(self,k=16):
        super(get_edge_feature,self).__init__()
        self.KNN=KNN(k=k+1,transpose_mode=False)
        self.k=k
    def forward(self,point_cloud):
        dist,idx=self.KNN(point_cloud,point_cloud)
        '''
        idx is batch_size,k,n_points
        point_cloud is batch_size,n_dims,n_points
        point_cloud_neightbors is batch_size,n_dims,k,n_points
        '''
        idx=idx[:,1:,:]
        point_cloud_neighbors=grouping_operation(point_cloud,idx.contiguous().int())
        point_cloud_central=point_cloud.unsqueeze(2).repeat(1,1,self.k,1)
        #print(point_cloud_central.shape,point_cloud_neighbors.shape)
        edge_feature=torch.cat([point_cloud_central,point_cloud_neighbors-point_cloud_central],dim=1)

        return edge_feature,idx



        return dist,idx

class denseconv(nn.Module):
    def __init__(self,growth_rate=64,k=16,in_channels=6,isTrain=True):
        super(denseconv,self).__init__()
        self.edge_feature_model=get_edge_feature(k=k)
        '''
        input to conv1 is batch_size,2xn_dims,k,n_points
        '''
        self.conv1=nn.Sequential(
            Conv2d(in_channels=in_channels,out_channels=growth_rate,kernel_size=[1,1]),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            Conv2d(in_channels=growth_rate+in_channels,out_channels=growth_rate,kernel_size=[1,1]),
            nn.ReLU()
        )
        self.conv3=nn.Sequential(
            Conv2d(in_channels=2*growth_rate+in_channels,out_channels=growth_rate,kernel_size=[1,1]),
        )
    def forward(self,input):
        '''
        y should be batch_size,in_channel,k,n_points
        '''
        y,idx=self.edge_feature_model(input)
        inter_result=torch.cat([self.conv1(y),y],dim=1) #concat on feature dimension
        inter_result=torch.cat([self.conv2(inter_result),inter_result],dim=1)
        inter_result=torch.cat([self.conv3(inter_result),inter_result],dim=1)
        final_result=torch.max(inter_result,dim=2)[0] #pool the k channel
        return final_result,idx


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction,self).__init__()
        self.growth_rate=24
        self.dense_n=3
        self.knn=16
        self.input_channel=3
        comp=self.growth_rate*2
        '''
        make sure to permute the input, the feature dimension is in the second one.
        input of conv1 is batch_size,num_dims,num_points
        '''
        self.conv1=nn.Sequential(
            Conv1d(in_channels=self.input_channel,out_channels=24,kernel_size=1,padding=0),
            nn.ReLU()
        )
        self.denseconv1=denseconv(in_channels=24*2,growth_rate=self.growth_rate)#return batch_size,(3*24+48)=120,num_points
        self.conv2=nn.Sequential(
            Conv1d(in_channels=144,out_channels=comp,kernel_size=1),
            nn.ReLU()
        )
        self.denseconv2=denseconv(in_channels=comp*2,growth_rate=self.growth_rate)
        self.conv3=nn.Sequential(
            Conv1d(in_channels=312,out_channels=comp,kernel_size=1),
            nn.ReLU()
        )
        self.denseconv3=denseconv(in_channels=comp*2,growth_rate=self.growth_rate)
        self.conv4=nn.Sequential(
            Conv1d(in_channels=480,out_channels=comp,kernel_size=1),
            nn.ReLU()
        )
        self.denseconv4=denseconv(in_channels=comp*2,growth_rate=self.growth_rate)
    def forward(self,input):
        l0_features=self.conv1(input) #b,24,n
        #print(l0_features.shape)
        l1_features,l1_index=self.denseconv1(l0_features) #b,24*2+24*3=120,n
        l1_features=torch.cat([l1_features,l0_features],dim=1) #b,120+24=144,n

        l2_features=self.conv2(l1_features) #b,48,n
        l2_features,l2_index=self.denseconv2(l2_features) #b,48*2+24*3=168,n
        l2_features=torch.cat([l2_features,l1_features],dim=1)#b,168+144=312,n

        l3_features=self.conv3(l2_features)#b,48,n
        l3_features,l3_index=self.denseconv3(l3_features)#b,48*2+24*3=168,n
        l3_features=torch.cat([l3_features,l2_features],dim=1)#b,168+312=480,n

        l4_features=self.conv4(l3_features)#b,48,n
        l4_features,l4_index=self.denseconv4(l4_features)
        l4_features=torch.cat([l4_features,l3_features],dim=1)#b,168+480=648,n

        return l4_features

class Generator(nn.Module):
    def __init__(self,params=None):
        super(Generator,self).__init__()
        self.feature_extractor=feature_extraction()
        #self.up_ratio=params['up_ratio']
        #self.num_points=params['patch_num_point']
        #self.out_num_point=int(self.num_points*self.up_ratio)
        self.up_projection_unit=up_projection_unit()

        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=64,kernel_size=1),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=3,kernel_size=1)
        )
    def forward(self,input):
        features=self.feature_extractor(input) #b,648,n


        H=self.up_projection_unit(features) #b,128,4*n

        coord=self.conv1(H)
        coord=self.conv2(coord)
        return coord

class Generator_recon(nn.Module):
    def __init__(self,params):
        super(Generator_recon,self).__init__()
        self.feature_extractor=feature_extraction()
        self.up_ratio=params['up_ratio']
        self.num_points=params['patch_num_point']

        self.conv0=nn.Sequential(
            nn.Conv1d(in_channels=648,out_channels=128,kernel_size=1),
            nn.ReLU()
        )

        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=64,kernel_size=1),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=3,kernel_size=1)
        )
    def forward(self,input):
        features=self.feature_extractor(input) #b,648,n
        coord=self.conv0(features)
        coord=self.conv1(coord)
        coord=self.conv2(coord)
        return coord

class attention_unit(nn.Module):
    def __init__(self,in_channels=130):
        super(attention_unit,self).__init__()
        self.convF=nn.Sequential(
            Conv1d(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1),
            nn.ReLU()
        )
        self.convG = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels// 4, kernel_size=1),
            nn.ReLU()
        )
        self.convH = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReLU()
        )
        self.gamma=nn.Parameter(torch.tensor(torch.zeros([1]))).cuda()
    def forward(self,inputs):
        f=self.convF(inputs)
        g=self.convG(inputs)#b,32,n
        h=self.convH(inputs)
        s=torch.matmul(g.permute(0,2,1),f)#b,n,n
        beta=F.softmax(s,dim=2)#b,n,n

        o=torch.matmul(h,beta)#b,130,n

        x=self.gamma*o+inputs

        return x



class up_block(nn.Module):
    def __init__(self,up_ratio=4,in_channels=130):
        super(up_block,self).__init__()
        self.up_ratio=up_ratio
        self.conv1=nn.Sequential(
            Conv1d(in_channels=in_channels,out_channels=256,kernel_size=1),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            Conv1d(in_channels=256,out_channels=128,kernel_size=1),
            nn.ReLU()
        )
        self.grid=torch.tensor(self.gen_grid(up_ratio)).cuda()
        self.attention_unit=attention_unit(in_channels=in_channels)
    def forward(self,inputs):
        net=inputs #b,128,n
        grid=self.grid.clone()
        grid=grid.unsqueeze(0).repeat(net.shape[0],1,net.shape[2])#b,4,2*n
        grid=grid.view([net.shape[0],-1,2])#b,4*n,2

        net=net.permute(0,2,1)#b,n,128
        net=net.repeat(1,self.up_ratio,1)#b,4n,128
        net = torch.cat([net, grid], dim=2)  # b,n*4,130

        net=net.permute(0,2,1)#b,130,n*4

        net=self.attention_unit(net)

        net=self.conv1(net)
        net=self.conv2(net)

        return net


    def gen_grid(self,up_ratio):
        import math
        sqrted=int(math.sqrt(up_ratio))+1
        for i in range(1,sqrted+1).__reversed__():
            if (up_ratio%i)==0:
                num_x=i
                num_y=up_ratio//i
                break
        grid_x=torch.linspace(-0.2,0.2,num_x)
        grid_y=torch.linspace(-0.2,0.2,num_y)

        x,y=torch.meshgrid([grid_x,grid_y])
        grid=torch.stack([x,y],dim=-1)#2,2,2
        grid=grid.view([-1,2])#4,2
        return grid

class down_block(nn.Module):
    def __init__(self,up_ratio=4,in_channels=128):
        super(down_block,self).__init__()
        self.conv1=nn.Sequential(
            Conv2d(in_channels=in_channels,out_channels=256,kernel_size=[up_ratio,1],padding=0),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            Conv1d(in_channels=256,out_channels=128,kernel_size=1),
            nn.ReLU()
        )
        self.up_ratio=up_ratio
    def forward(self,inputs):
        net=inputs#b,128,n*4
        #net = torch.cat(
        #    [net[:, :, 0:1024].unsqueeze(2), net[:, :, 1024:2048].unsqueeze(2), net[:, :, 2048:3072].unsqueeze(2),
        #     net[:, :, 3072:4096].unsqueeze(2)], dim=2)
        net=net.view([inputs.shape[0],inputs.shape[1],self.up_ratio,-1])#b,128,4,n
        #net=torch.cat(torch.unbind(net,dim=2),dim=2)
        net=self.conv1(net)#b,256,1,n
        net=net.squeeze(2)
        net=self.conv2(net)
        return net


class up_projection_unit(nn.Module):
    def __init__(self,up_ratio=4):
        super(up_projection_unit,self).__init__()
        self.conv1=nn.Sequential(
            Conv1d(in_channels=648,out_channels=128,kernel_size=1),
            nn.ReLU()
        )
        self.up_block1=up_block(up_ratio=4,in_channels=128+2)
        self.up_block2=up_block(up_ratio=4,in_channels=128+2)
        self.down_block=down_block(up_ratio=4,in_channels=128)
    def forward(self,input):
        L=self.conv1(input)#b,128,n

        H0=self.up_block1(L)#b,128,n*4
        L0=self.down_block(H0)#b,128,n

        #print(H0.shape,L0.shape,L.shape)
        E0=L0-L #b,128,n
        H1=self.up_block2(E0)#b,128,4*n
        H2=H0+H1 #b,128,4*n
        return H2

class mlp_conv(nn.Module):
    def __init__(self,in_channels,layer_dim):
        super(mlp_conv,self).__init__()
        self.conv_list=nn.ModuleList()
        for i,num_out_channel in enumerate(layer_dim[:-1]):
            if i==0:
                sub_module=nn.Sequential(
                    Conv1d(in_channels=in_channels, out_channels=num_out_channel, kernel_size=1),
                    nn.ReLU()
                )
                self.conv_list.append(sub_module)
            else:
                sub_module=nn.Sequential(
                    Conv1d(in_channels=layer_dim[i-1],out_channels=num_out_channel,kernel_size=1),
                    nn.ReLU()
                )
                self.conv_list.append(sub_module)
        self.conv_list.append(
            Conv1d(in_channels=layer_dim[-2],out_channels=layer_dim[-1],kernel_size=1)
        )
    def forward(self,inputs):
        net=inputs
        for module in self.conv_list:
            net=module(net)
        return net

class mlp(nn.Module):
    def __init__(self,in_channels,layer_dim):
        super(mlp,self).__init__()
        self.mlp_list=nn.ModuleList()
        for i,num_outputs in enumerate(layer_dim[:-1]):
            if i==0:
                sub_module=nn.Sequential(
                    nn.Linear(in_channels, num_outputs),
                    nn.ReLU()
                )
                self.mlp_list.append(sub_module)
            else:
                sub_module=nn.Sequential(
                    nn.Linear(layer_dim[i-1],num_outputs),
                    nn.ReLU()
                )
                self.mlp_list.append(sub_module)
        self.mlp_list.append(
            nn.Linear(layer_dim[-2],layer_dim[-1])
        )
    def forward(self,inputs):
        net=inputs
        for sub_module in self.mlp_list:
            net=sub_module(net)
        return net

class Discriminator(nn.Module):
    def __init__(self,params,in_channels):
        super(Discriminator,self).__init__()
        self.params=params
        self.start_number=32
        self.mlp_conv1=mlp_conv(in_channels=in_channels,layer_dim=[self.start_number, self.start_number * 2])
        self.attention_unit=attention_unit(in_channels=self.start_number*4)
        self.mlp_conv2=mlp_conv(in_channels=self.start_number*4,layer_dim=[self.start_number*4,self.start_number*8])
        self.mlp=mlp(in_channels=self.start_number*8,layer_dim=[self.start_number * 8, 1])
    def forward(self,inputs):
        features=self.mlp_conv1(inputs)
        features_global=torch.max(features,dim=2)[0] ##global feature
        features=torch.cat([features,features_global.unsqueeze(2).repeat(1,1,features.shape[2])],dim=1)
        features=self.attention_unit(features)

        features=self.mlp_conv2(features)
        features=torch.max(features,dim=2)[0]

        output=self.mlp(features)

        return output
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


if __name__=="__main__":
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params={
        "up_ratio":4,
        "patch_num_point":100
    }
    generator=Generator(params).cuda()
    point_cloud=torch.rand(4,3,100).cuda()
    output=generator(point_cloud)
    print(output.shape)
    discriminator=Discriminator(params,in_channels=3).cuda()
    dis_output=discriminator(output)
    print(dis_output.shape)