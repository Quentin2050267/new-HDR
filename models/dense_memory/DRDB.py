import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint 

class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, dilation_size, kernel_size=(1, 3, 3)):
        super(make_dilation_dense, self).__init__()
        
        # 修改一下
        padding_size = tuple((dilation_size * (k - 1)) // 2 for k in kernel_size)
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=kernel_size, padding=padding_size,
                              bias=True, dilation=dilation_size, groups=1)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
    

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, ds=(1, 2, 3, 5, 7)):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate, ds[i]))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1x1(out)
        out = out + x
        return out
    


class Feat_extractor(nn.Module):     
    def __init__(self, in_channel=3, nf=144//2):         
        super(Feat_extractor, self).__init__()          
        self.enc1 = nn.Sequential(             
            nn.Conv3d(in_channel, nf, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),             
            nn.LeakyReLU(0.2, inplace=True),             
            #nn.Conv3d(nf, nf, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),             
            #nn.LeakyReLU(0.2, inplace=True),         
            )         
        
        self.enc2 = nn.Sequential(             
            nn.Conv3d(nf, nf * 2, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # [128, h/2, w/2]             
            nn.LeakyReLU(0.2, inplace=True),             
            #nn.Conv3d(nf * 2, nf * 2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),  # [64, h/2, w/2]             
            #nn.LeakyReLU(0.2, inplace=True),         
            )         
        
        self.enc3 = nn.Sequential(             
            nn.Conv3d(nf * 2, nf * 2, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # [128, h/4, w/4]             
            nn.LeakyReLU(0.2, inplace=True),             
            #nn.Conv3d(nf * 2, nf * 2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),  # [128, h/4, w/4]             
            #nn.LeakyReLU(0.2, inplace=True),         
            )          
        
        self.res = nn.Sequential(             
            DRDB(nf * 2, 4, nf // 2),             
            DRDB(nf * 2, 4, nf // 2),             
            )      
        
    
    def forward(self, x):         
        bt, c, _, _, _ = x.size()         # h, w = h//4, w//4         
        mid_feature = []         
        x = self.enc1(x)         
        mid_feature.append(x)  # c=64, h,w          
        x = self.enc2(x)         
        mid_feature.append(x)  # c=128, h/2, w/2          
        x = self.enc3(x)  # 256 h/4, h/4         
        x = checkpoint(self.res, x)           
        return x #, mid_feature 
    
class Reconstruction(nn.Module):     
    def __init__(self, in_channel=3, nf=144//2):         
        super(Reconstruction, self).__init__()          
        self.enc1 = nn.Sequential(             
            nn.Conv3d(in_channel, nf, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),             
            nn.LeakyReLU(0.2, inplace=True),             
            #nn.Conv3d(nf, nf, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),             
            #nn.LeakyReLU(0.2, inplace=True),         
            )         
        
        self.enc2 = nn.Sequential(             
            nn.Conv3d(nf, nf * 2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),  # [144, h, w]             
            nn.LeakyReLU(0.2, inplace=True),             
            #nn.Conv3d(nf * 2, nf * 2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),  # [64, h/2, w/2]             
            #nn.LeakyReLU(0.2, inplace=True),         
            )         
        
        self.enc3 = nn.Sequential(             
            nn.Conv3d(nf * 2, nf * 2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),  # [144, h, w]             
            nn.LeakyReLU(0.2, inplace=True),             
            #nn.Conv3d(nf * 2, nf * 2, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),  # [128, h/4, w/4]             
            #nn.LeakyReLU(0.2, inplace=True),         
            )          
        
        self.res = nn.Sequential(             
            DRDB(nf * 2, 4, nf // 2),             
            DRDB(nf * 2, 4, nf // 2),             
            )      
        
    
    def forward(self, x):         
        bt, c, _, _, _ = x.size()         # h, w = h//4, w//4         
        mid_feature = []         
        x = self.enc1(x)         
        mid_feature.append(x)  # c=64, h,w          
        x = self.enc2(x)         
        mid_feature.append(x)  # c=128, h/2, w/2          
        x = self.enc3(x)  # 256 h/4, h/4         
        x = checkpoint(self.res, x)           
        return x #, mid_feature 


