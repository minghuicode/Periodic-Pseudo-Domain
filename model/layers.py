'''
model utils layers:
    - re-organize
    - Upsample 
    - Residual Block 
    - conv_leaky
    - conv_down
create date: 2022-05-17-13:12
'''
import torch 
import torch.nn as nn 
import torch.nn.functional as F  

__all__ = ['ReOrg', 'ResidualBlock', 'Upsample', 'conv_leaky', 'conv_down']

class ReOrg(nn.Module):
    '''
    re-orgnize the feature map 
    '''
    def __init__(self, grid_size: int=2):
        super().__init__()
        self.grid_size = grid_size 

    def forward(self, x): 
        s = self.grid_size 
        nB, nC, nH, nW = x.shape 
        assert(nH%s==0)
        assert(nW%s==0)  
        h = nH//s 
        w = nW//s 
        c = nC*s*s 
        # resize for height and width 
        x = x.reshape(nB, nC, h, s, w, s)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(nB, h, w, c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

def conv_leaky(c_i: int, c_o: int, kernel_size=3):
    pad = (kernel_size - 1)//2
    return nn.Sequential(
        nn.Conv2d(
            in_channels=c_i,
            out_channels=c_o,
            kernel_size=kernel_size,
            stride=1,
            padding=pad,
            bias=False
        ),
        nn.BatchNorm2d(c_o, momentum=0.9, eps=1e-5),
        nn.LeakyReLU(0.1)
    )

def conv_down(c_i: int, c_o: int):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=c_i,
            out_channels=c_o,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        ),
        nn.BatchNorm2d(c_o, momentum=0.9, eps=1e-5),
        nn.LeakyReLU(0.1)
    )

def conv_dilate(c_i: int, c_o: int):
    return nn.Sequential( 
        nn.Conv2d(
            in_channels=c_i,
            out_channels=c_o,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            bias=False
        ),
        nn.BatchNorm2d(c_o, momentum=0.9, eps=1e-5),
        nn.LeakyReLU(0.1)
    )
    
def conv_twin(c_i: int, c_t: int=None):
    '''
    basic part for residual block
    '''
    if c_t is None:
        c_t = c_i//4
    return nn.Sequential(
        conv_leaky(c_i, c_t, 1),
        conv_leaky(c_t, c_i, 3)
    )

def dilate_twin(c_i: int, c_t: int=None):
    '''
    basic part for residual block
    '''
    if c_t is None:
        c_t = c_i//4
    return nn.Sequential(
        conv_leaky(c_i, c_t, 1),
        conv_dilate(c_t, c_i)
    )

class Upsample(nn.Module):
    '''
    basic part in FPN structure 
    '''
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


def up_twin(c_i: int, c_o: int):
    '''
    basic part in FPN structure 
    '''
    return nn.Sequential(
        conv_leaky(c_i, c_o, 1),
        Upsample(scale_factor=2)
    )
  
class Bottleneck(nn.Module):
    '''
    bottleneck module 
    define as resnet part
    '''
    def __init__(self, c_i: int, c_o: int, c_t: int):
        super().__init__()
        assert(c_i==c_o)
        # if c_i != c_o:
        # downsampling   
        self.conv1 = nn.Conv2d(c_i, c_t, 1) 
        self.bn1 = nn.BatchNorm2d(c_t)
        self.relu = nn.ReLU(inplace=True) 
        self.conv2 = nn.Conv2d(c_t, c_t, 3, 1, 1) 
        self.bn2 = nn.BatchNorm2d(c_t)
        self.conv3 = nn.Conv2d(c_t, c_o, 1)
        self.bn3 = nn.BatchNorm2d(c_o)
        # self.twins = nn.ModuleList([
        #     conv_twin(c_i, c_t),
        #     dilate_twin(c_i, c_t),
        #     conv_twin(c_i, c_t),
        #     dilate_twin(c_i, c_t),

        #     conv_twin(c_i, c_t),
        #     dilate_twin(c_i, c_t),
        # ])
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)  
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)  
        y = self.conv3(y)
        y = self.bn3(y) 
        return self.relu(x+y)    


class ResidualBlock(nn.Module):
    '''
    a residual sequence
    '''
    def __init__(self, c_i: int, c_t: int):
        super().__init__() 
        # self.twins = nn.ModuleList([
        #     conv_twin(c_i, c_t),
        #     dilate_twin(c_i, c_t),
        #     conv_twin(c_i, c_t),
        #     dilate_twin(c_i, c_t),

        #     conv_twin(c_i, c_t),
        #     dilate_twin(c_i, c_t),
        # ])
        self.res = nn.ModuleList([
            Bottleneck(c_i, c_i, c_t),
            Bottleneck(c_i, c_i, c_t),
            Bottleneck(c_i, c_i, c_t)
        ])

    def forward(self, x):
        return x 
        for res in self.res: 
            x = x + res(x) 
        return x  


if __name__ == "__main__":
    conv = conv_dilate(3,3)
    x = torch.randn([3,3,512,512])
    y = conv(x)
    print(x.shape)
    print(y.shape)

