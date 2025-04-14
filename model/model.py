import torch
import torch.nn as nn
from model.backbone_radarResNet import build_backbone
from model.yolo_head import singleLayerHead
import torch.nn.functional as F
from .triplet_attention import *
from complextorch.nn.modules.conv import CVConv2d
from complextorch.nn.modules.layernorm import CVLayerNorm

class RADDet(nn.Module):
    def __init__(self, config_model, config_data, config_train, anchor_boxes):
        super(RADDet, self).__init__()
        assert (isinstance(config_model["input_shape"], tuple) or isinstance(config_model["input_shape"], list))
        self.input_size = list(config_model["input_shape"])
        self.input_channels = self.input_size[-1]
        self.num_class = len(config_data["all_classes"])
        self.anchor_boxes = anchor_boxes
        self.yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
        self.focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]
        self.yolo_feature_size = config_model["yolo_feature_size"]
        self.backbone = build_backbone(config_model)
        self.yolo_head = singleLayerHead(num_anchors=len(self.anchor_boxes),
                                         num_class=self.num_class,
                                         last_channel=int(self.yolo_feature_size[-1]/4),
                                         in_feature_size=self.yolo_feature_size)

    def forward(self, input):
        bk_out = self.backbone(input)
        out = self.yolo_head(bk_out)
        return bk_out, out


class FPNDet(nn.Module):
    def __init__(self, config_model, config_data, config_train, anchor_boxes):
        super(FPNDet, self).__init__()
        assert (isinstance(config_model["input_shape"], tuple) or isinstance(config_model["input_shape"], list))
        self.input_size = list(config_model["input_shape"])
        self.input_channels = self.input_size[-1]
        self.num_class = len(config_data["all_classes"])
        self.anchor_boxes = anchor_boxes
        self.yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
        self.focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]
        self.yolo_feature_size = config_model["yolo_feature_size"]
        self.complx_Angle_FFT = complex_Angle_FFT()
        if config_model["use_triplet_attention"]:
            self.backbone = serial_atten_FPN_BackBone(num_block=[3, 6, 6, 3],channels= [32,40,48,56],mimo_layer=64,block_expansion=4,use_triplet_attention=config_model["use_triplet_attention"])
        else:
            self.backbone = FPN_BackBone(num_block=[3, 6, 6, 3],channels= [32,40,48,56],mimo_layer=64,block_expansion=4)
        self.RA_decoder = RangeAngle_Decoder()
        self.yolo_head = singleLayerHead(num_anchors=len(self.anchor_boxes),
                                         num_class=self.num_class,
                                         last_channel=int(self.yolo_feature_size[-1]/4),
                                         in_feature_size=self.yolo_feature_size)

    def forward(self, input):
        complx_ARD = self.complx_Angle_FFT(input)
        magnitude_ARD = torch.sqrt(complx_ARD.realize **2 + magnitude_ARD.imagize **2)
        real_RAD = magnitude_ARD.permute(0,3,2,1)
        bk_out = self.backbone(real_RAD)
        raout=self.RA_decoder(bk_out)
        out = self.yolo_head(raout)
        return bk_out, out

class complex_Angle_FFT(nn.Module):

    def __init__(self,):
        super(complex_Angle_FFT, self).__init__()
        self.cplx_Conv2D = CVConv2d(8, 256, kernel_size=1)
        self.cplx_Ly_normal = CVLayerNorm(normalized_shape=[256,256,64],elementwise_affine=False)

    def forward(self,x):
        x = self.cplx_Conv2D(x)
        x = self.cplx_Ly_normal(x)

class serial_atten_FPN_BackBone(nn.Module):

    def __init__(self, num_block,channels,mimo_layer,block_expansion,use_bn=True, use_triplet_attention:str=None):
        super(serial_atten_FPN_BackBone, self).__init__()
        self.in_planes = mimo_layer
        self.block_expansion = block_expansion
        self.use_bn = use_bn
        self.conv = conv3x3(self.in_planes, self.in_planes)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Residuall blocks
        self.block1 = self._make_layer(serial_tri_atten_Bottleneck, planes=channels[0], num_blocks=num_block[0], use_triplet_attention=use_triplet_attention)
        self.block2 = self._make_layer(serial_tri_atten_Bottleneck, planes=channels[1], num_blocks=num_block[1], use_triplet_attention=use_triplet_attention)
        self.block3 = self._make_layer(serial_tri_atten_Bottleneck, planes=channels[2], num_blocks=num_block[2], use_triplet_attention=use_triplet_attention)
        self.block4 = self._make_layer(serial_tri_atten_Bottleneck, planes=channels[3], num_blocks=num_block[3], use_triplet_attention=use_triplet_attention)
                                       
    def forward(self, x):

        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Backbone
        features = {}
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        
        features['x0'] = x
        features['x1'] = x1
        features['x2'] = x2
        features['x3'] = x3
        features['x4'] = x4

        return features
            
    def _make_layer(self, block, planes, num_blocks, use_triplet_attention:bool=False):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * self.block_expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * self.block_expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample,expansion=self.block_expansion, use_triplet_attention=use_triplet_attention))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1,expansion=self.block_expansion, use_triplet_attention=use_triplet_attention))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)
    
class FPN_BackBone(nn.Module):

    def __init__(self, num_block,channels,mimo_layer,block_expansion,use_bn=True):
        super(FPN_BackBone, self).__init__()
        self.in_planes = mimo_layer
        self.block_expansion = block_expansion
        self.use_bn = use_bn
        self.conv = conv3x3(self.in_planes, self.in_planes)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Residuall blocks
        self.block1 = self._make_layer(Bottleneck, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(Bottleneck, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(Bottleneck, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(Bottleneck, planes=channels[3], num_blocks=num_block[3])
                                    
    def forward(self, x):

        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Backbone
        features = {}
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        
        features['x0'] = x
        features['x1'] = x1
        features['x2'] = x2
        features['x3'] = x3
        features['x4'] = x4

        return features


    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * self.block_expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * self.block_expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample,expansion=self.block_expansion))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1,expansion=self.block_expansion))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)



def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)



class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None,expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)


        out = F.relu(residual + out)
        return out

class serial_tri_atten_Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None,expansion=4,use_triplet_attention=False):
        super(serial_tri_atten_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        if use_triplet_attention == "RA_Before":
            self.triplet_attention = RA_Before_Serial_TripletAttention(no_spatial=False)
        elif use_triplet_attention == "RA_Behand":
            self.triplet_attention = RA_Behand_Serial_TripletAttention(no_spatial=False)
        elif use_triplet_attention == "RA_Middle":
            self.triplet_attention = RA_Middle_Serial_TripletAttention(no_spatial=False)
        elif use_triplet_attention == "Pallel":
            self.triplet_attention = TripletAttention(no_spatial=False)    
        else:
            self.triplet_attention = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out_cov = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.triplet_attention is not None:
            # out = self.triplet_attention(out)
            # torch.cuda.empty_cache() #清空当前显存
            # torch.cuda.reset_peak_memory_stats() #记录显存使用情况
            out_atten = self.triplet_attention(out_cov)
            # max_memory = torch.cuda.max_memory_allocated() #输出最大显存
            # print(f'Max memory allocated: {max_memory / (1024 ** 3):.2f} GB')
            out = out_atten + out_cov

        out = F.relu(residual + out)
        return out

class RangeAngle_Decoder(nn.Module):
    def __init__(self, ):
        super(RangeAngle_Decoder, self).__init__()
        
        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(224, 224, kernel_size=3, stride=(1,2), padding=1, output_padding=(0,1))
        
        self.conv_block4 = BasicBlock(448,128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))
        self.conv_block3 = BasicBlock(352,256)

        self.L3  = nn.Conv2d(192, 224, kernel_size=3, stride=(2,1),padding=1)
        self.L2  = nn.Conv2d(160, 224, kernel_size=3, stride=(2,1),padding=1)
        
        
    def forward(self,features):

        T4 = features['x4']#.transpose(1, 3)  #224,16,16
        T3 = self.L3(features['x3'])#.transpose(1, 3)   #224,16,32
        T2 = self.L2(features['x2'])#.transpose(1, 3)   #224,32,64
        TT4= self.deconv4(T4)            #224,16,32
        S4 = torch.cat((TT4,T3),axis=1)  #448,16,32
        S4 = self.conv_block4(S4)       #128,16,32
        deconvS4 = self.deconv3(S4)     #128,16,
        S43 = torch.cat((deconvS4,T2),axis=1)    #352,32,64
        out = self.conv_block3(S43)
        
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out