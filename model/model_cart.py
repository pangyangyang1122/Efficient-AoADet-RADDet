import torch
import torch.nn as nn
from einops import rearrange
import util.loader as loader
from model.model import RADDet
from dataset.radar_dataset_3d import RararDataset3D
# from complextorch.nn.modules.conv import CVConv2d
# from complextorch.nn.modules.layernorm import CVLayerNorm
# from complextorch.nn.modules.activation.complex_relu import CPReLU,CReLU
from cplxmodule.nn import CplxConv1d, CplxLinear, CplxDropout
from cplxmodule.nn import CplxModReLU, CplxParameter, CplxModulus, CplxToCplx, CplxAngle
from cplxmodule.nn.modules.casting import TensorToCplx,CplxToTensor
from cplxmodule.nn import RealToCplx, CplxToReal
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_


# class complex_Angle_FFT(nn.Module):
#     "relize angle-FFT use convolution"

#     def __init__(self,):
#         super(complex_Angle_FFT, self).__init__()
#         self.cplx_Conv2D = CVConv2d(8, 256, kernel_size=1)
#         self.cplx_Ly_normal = CVLayerNorm(normalized_shape=[256,256,64],elementwise_affine=False)
#         # self.cplx_relu = CReLU(inplace=False)
#         # self.cplx_relu = CPReLU()

#     def forward(self,x):
#         x = self.cplx_Conv2D(x)    #12,256,256,64
#         x = self.cplx_Ly_normal(x)
#         # x = self.cplx_relu(x)
#         return x
    
class AOA_Fourier_Net(nn.Module):
    def __init__(self):
        super(AOA_Fourier_Net, self).__init__()
        self.aoa_nn = CplxLinear(8, 256, bias = False)
        aoa_weights = np.zeros((256, 8), dtype=np.complex64)
        for j in range(8):
            for h in range(256):
                hh = h + 128
                if hh >= 256:
                    hh = hh - 256
                h_idx = h
                aoa_weights[h_idx][j] = np.exp(-1j * 2 * np.pi * (j *hh / 256))

        aoa_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(aoa_weights)))
        self.aoa_nn.weight = CplxParameter(aoa_weights)
    
    def forward(self, x):
        x = self.aoa_nn(x)
        
        return x
    
class polor_to_cart(nn.Module):
    def __init__(self, ):
        super(polor_to_cart, self).__init__()
        # Top-down layers
        self.deconv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0,1))
        
        # self.conv_block = BasicBlock(128,256)
        
    def forward(self,x4,x3):

        # T4 = features #256，32，32
        TT4= self.deconv(x4) #128.32，64
        
        # S3 = self.conv_block(TT4) # 256,32，64
        
        
        return S3
    
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

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

class RADDetCart(nn.Module):
    def __init__(self, config_model, config_data, config_train, anchor_boxes, device, input_shape, babone_out_size, pos_encode,backbone=None, RA_decoder=None):
        """ make sure the model is buit when initializint the class.
        Only by this, the graph could be built and the trainable_variables
        could be initialized """
        super(RADDetCart, self).__init__()
        assert (isinstance(babone_out_size, tuple) or isinstance(babone_out_size, list))
        self.config_model = config_model
        self.config_data = config_data
        self.config_train = config_train
        self.babone_out_size = babone_out_size
        self.input_size = input_shape
        self.num_class = len(config_data["all_classes"])
        self.anchor_boxes = torch.tensor(anchor_boxes).to(device)
        self.yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
        self.focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]
        self.complx_Angle_FFT = AOA_Fourier_Net()
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.backbone = backbone
        self.pos_encode = pos_encode
        # self.RA_decoder = RA_decoder
        # self.polor_to_cart = polor_to_cart()
        if self.pos_encode:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.input_size[2], self.input_size[0], self.input_size[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            
        dense_feature_size = babone_out_size[0]*babone_out_size[1]
        self.fc1 = nn.Linear(in_features=dense_feature_size, out_features=dense_feature_size*2, bias=True)
        self.fc2 = nn.Linear(in_features=dense_feature_size*2, out_features=dense_feature_size*2, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=self.babone_out_size[-1],
                               out_channels=self.babone_out_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn1 = nn.BatchNorm2d(self.babone_out_size[-1])

        self.conv2 = nn.Conv2d(in_channels=self.babone_out_size[-1],
                               out_channels=self.babone_out_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn2 = nn.BatchNorm2d(self.babone_out_size[-1])

        self.conv3 = nn.Conv2d(in_channels=self.babone_out_size[-1],
                               out_channels=self.babone_out_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn3 = nn.BatchNorm2d(self.babone_out_size[-1])

        self.conv_yolo_head = nn.Conv2d(in_channels=self.babone_out_size[-1],
                                        out_channels=self.babone_out_size[-1]*2,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True
                                        )
        self.bn_yolo_head = nn.BatchNorm2d(self.babone_out_size[-1]*2)

        self.conv_yolo_head_1 = nn.Conv2d(in_channels=self.babone_out_size[-1]*2,
                                          out_channels=len(self.anchor_boxes) * (self.num_class + 5),
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=True
                                          )
        self.conv224_256 = nn.Conv2d(in_channels=224,
                                          out_channels=256,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=True
                                          )
        self.bn224_256 = nn.BatchNorm2d(256)

        # TODO: adding weight initialization code.
    def forward(self, input):
        x_RDA = input.permute(0,2,3,1) #6，256，64，8
        complx_RDA = self.complx_Angle_FFT(x_RDA)        #[B,256,64,256]
        magnitude_RDA = torch.sqrt(complx_RDA.real **2 + complx_RDA.imag **2)
        real_DRA = magnitude_RDA.permute(0,2,1,3)
        flipped_DRA = torch.flip(real_DRA, dims=[-1])   # 对角度维flip
        if self.pos_encode:
            Wh, Ww = flipped_DRA.size(2), flipped_DRA.size(3)
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            flipped_DRA = (flipped_DRA + absolute_pos_embed)
        x = self.backbone(flipped_DRA)       #12,256,16,16
        ###########预测结果不翻倍，即不要RA_decoder用这两句######
        x = x['x4'] 
        x = self.bn224_256(self.conv224_256(x))  
        #####################################
        # x = self.RA_decoder(x) #12,256,32,32   输入翻倍用这个
        
        ####使用反卷积实现坐标转换#####
        # x = self.polor_to_cart(x,x3) 
        #########利用全连接实现坐标转换###############
        x = rearrange(x, "b c h w -> b c (h w)")  #12,256,256
        x = self.relu(self.fc1(x))    #12,256,512
        x = self.relu(self.fc2(x))    #12,256,512
        x = rearrange(x, "b c (h w) -> b c h w", h=self.babone_out_size[0]) #12,256,16,32
        ############################################
        res_x = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))        #12,256,16,32
        x = self.relu(self.bn3(self.conv3(x)))       #12,256,16,32
        x = x + res_x      #12,256,16,32

        x = self.relu(self.bn_yolo_head(self.conv_yolo_head(x))) #12,512,16,32
        x = self.conv_yolo_head_1(x)    #12,66,16,32
        x = rearrange(x, "b c h w -> b h w c") #12,16,32,66
        x = rearrange(x, "b h w (c1 c2) -> b h w c1 c2", c1=len(self.anchor_boxes)) #12,16,32,6,11
        return x, flipped_DRA

    def decodeYolo(self, x):
        output_size = [int(self.config_model["input_shape"][0]), int(2 * self.config_model["input_shape"][0])]  #256,512
        strides = torch.tensor(output_size) / torch.tensor(list(x.shape[1:3]))  #[16,16]
        strides = strides.to(x.device)
        raw_xy, raw_wh, raw_conf, raw_prob = x[..., 0:2], x[..., 2:4], x[..., 4:5], x[..., 5:] #[12,16,32,6,2],[12,16,32,6,2],[12,16,32,6,1],[12,16,32,6,6]
        xx, yy = torch.meshgrid([torch.arange(0, x.shape[1]), torch.arange(0, x.shape[2])])  #[16,32],[16,32] x,y分别表示横纵坐标
        xy_grid = [xx.T, yy.T]
        xy_grid = torch.unsqueeze(torch.stack(xy_grid, dim=-1), dim=-2).to(x.device) #[32,16,1,2]
        xy_grid = torch.unsqueeze(rearrange(xy_grid, "b c h w -> c b h w"), dim=0)  #[1,16,32,1,2]
        xy_grid = torch.tile(xy_grid, (x.shape[0], 1, 1, len(self.anchor_boxes), 1)).to(torch.float32)   #[12,16,32,6,2],torch.tile() 函数将 xy_grid 张量沿着指定的维度进行复制
        scale = self.yolohead_xyz_scales[0] 
        pred_xy = ((torch.sigmoid(raw_xy) * scale) - 0.5 * (scale - 1) + xy_grid) * strides
        ###---------------- clipping values --------------------###
        raw_wh = torch.clamp(raw_wh, 1e-12, 1e12)
        ###-----------------------------------------------------###
        pred_wh = torch.exp(raw_wh) * self.anchor_boxes
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)

        pred_conf = torch.sigmoid(raw_conf)
        pred_prob = torch.sigmoid(raw_prob)
        return torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

    def extractYoloInfo(self, yoloformat_data):
        box = yoloformat_data[..., :4]
        conf = yoloformat_data[..., 4:5]
        category = yoloformat_data[..., 5:]
        return box, conf, category

    def loss(self, pred_raw, pred, gt, raw_boxes):
        raw_box, raw_conf, raw_category = self.extractYoloInfo(pred_raw)
        pred_box, pred_conf, pred_category = self.extractYoloInfo(pred)
        gt_box, gt_conf, gt_category = self.extractYoloInfo(gt)

        box_loss = gt_conf * (torch.square(pred_box[..., :2] - gt_box[..., :2]) +
                              torch.square(torch.sqrt(pred_box[..., 2:]) - torch.sqrt(gt_box[..., 2:])))
        iou = tf_iou2d(torch.unsqueeze(pred_box, dim=-2), raw_boxes[:, None, None, None, :, :])
        max_iou = torch.unsqueeze(torch.max(iou, dim=-1)[0], dim=-1)
        gt_conf_negative = (1.0 - gt_conf) * (max_iou < self.config_train["focal_loss_iou_threshold"]).to(torch.float32)
        conf_focal = torch.pow(gt_conf - pred_conf, 2)
        alpha = 0.01

        conf_loss = conf_focal * (gt_conf * self.bce_criterion(target=gt_conf, input=raw_conf)
                                  + alpha * gt_conf_negative * self.bce_criterion(target=gt_conf, input=raw_conf))
        ### NOTE: category loss function ###
        category_loss = gt_conf * self.bce_criterion(target=gt_category, input=raw_category)

        ### NOTE: combine together ###
        box_loss_all = torch.mean(torch.sum(box_loss, dim=[1, 2, 3, 4]))
        box_loss_all *= 1e-1
        conf_loss_all = torch.mean(torch.sum(conf_loss, dim=[1, 2, 3, 4]))
        
        category_loss_all = torch.mean(torch.sum(category_loss, dim=[1, 2, 3, 4]))
        
        total_loss = box_loss_all + conf_loss_all + category_loss_all
        return total_loss, box_loss_all, conf_loss_all, category_loss_all

class RADDetCart_convtoCart(nn.Module):
    def __init__(self, config_model, config_data, config_train, anchor_boxes, device, input_shape, babone_out_size, pos_encode,backbone=None, RA_decoder=None):
        """ make sure the model is buit when initializint the class.
        Only by this, the graph could be built and the trainable_variables
        could be initialized """
        super(RADDetCart_convtoCart, self).__init__()
        assert (isinstance(babone_out_size, tuple) or isinstance(babone_out_size, list))
        self.config_model = config_model
        self.config_data = config_data
        self.config_train = config_train
        self.babone_out_size = babone_out_size
        self.input_size = input_shape
        self.num_class = len(config_data["all_classes"])
        self.anchor_boxes = torch.tensor(anchor_boxes).to(device)
        self.yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
        self.focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]
        self.complx_Angle_FFT = AOA_Fourier_Net()
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.backbone = backbone
        self.pos_encode = pos_encode
        self.RA_decoder = RA_decoder
        # self.polor_to_cart = polor_to_cart()
        if self.pos_encode:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.input_size[2], self.input_size[0], self.input_size[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            
        # dense_feature_size = babone_out_size[0]*babone_out_size[1]
        # self.fc1 = nn.Linear(in_features=dense_feature_size, out_features=dense_feature_size*2, bias=True)
        # self.fc2 = nn.Linear(in_features=dense_feature_size*2, out_features=dense_feature_size*2, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=self.babone_out_size[-1],
                               out_channels=self.babone_out_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn1 = nn.BatchNorm2d(self.babone_out_size[-1])

        self.conv2 = nn.Conv2d(in_channels=self.babone_out_size[-1],
                               out_channels=self.babone_out_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn2 = nn.BatchNorm2d(self.babone_out_size[-1])

        self.conv3 = nn.Conv2d(in_channels=self.babone_out_size[-1],
                               out_channels=self.babone_out_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn3 = nn.BatchNorm2d(self.babone_out_size[-1])

        self.conv_yolo_head = nn.Conv2d(in_channels=self.babone_out_size[-1],
                                        out_channels=self.babone_out_size[-1]*2,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True
                                        )
        self.bn_yolo_head = nn.BatchNorm2d(self.babone_out_size[-1]*2)

        self.conv_yolo_head_1 = nn.Conv2d(in_channels=self.babone_out_size[-1]*2,
                                          out_channels=len(self.anchor_boxes) * (self.num_class + 5),
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=True
                                          )
        # self.conv224_256 = nn.Conv2d(in_channels=224,
        #                                   out_channels=256,
        #                                   kernel_size=1,
        #                                   stride=1,
        #                                   padding=0,
        #                                   bias=True
        #                                   )
        # self.bn224_256 = nn.BatchNorm2d(256)

        # TODO: adding weight initialization code.
    def forward(self, input):
        x_RDA = input.permute(0,2,3,1) #6，256，64，8
        complx_RDA = self.complx_Angle_FFT(x_RDA)        #[B,256,64,256]
        magnitude_RDA = torch.sqrt(complx_RDA.real **2 + complx_RDA.imag **2)
        real_DRA = magnitude_RDA.permute(0,2,1,3)
        flipped_DRA = torch.flip(real_DRA, dims=[-1])   # 对角度维flip
        if self.pos_encode:
            Wh, Ww = flipped_DRA.size(2), flipped_DRA.size(3)
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            flipped_DRA = (flipped_DRA + absolute_pos_embed)
        x = self.backbone(flipped_DRA)       #12,256,16,16
        
        x = self.RA_decoder(x) #12,256,32,64   输入翻倍用这个,作为坐标变换模块
        
        ####使用反卷积实现坐标转换#####
        # x = self.polor_to_cart(x,x3) 
        #########利用全连接实现坐标转换###############
        # x = rearrange(x, "b c h w -> b c (h w)")  #12,256,256
        # x = self.relu(self.fc1(x))    #12,256,512
        # x = self.relu(self.fc2(x))    #12,256,512
        # x = rearrange(x, "b c (h w) -> b c h w", h=self.babone_out_size[0]) #12,256,16,32
        ############################################
        res_x = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))        #12,256,16,32
        x = self.relu(self.bn3(self.conv3(x)))       #12,256,16,32
        x = x + res_x      #12,256,16,32

        x = self.relu(self.bn_yolo_head(self.conv_yolo_head(x))) #12,512,16,32
        x = self.conv_yolo_head_1(x)    #12,66,16,32
        x = rearrange(x, "b c h w -> b h w c") #12,16,32,66
        x = rearrange(x, "b h w (c1 c2) -> b h w c1 c2", c1=len(self.anchor_boxes)) #12,16,32,6,11
        return x, flipped_DRA

    def decodeYolo(self, x):
        output_size = [int(self.config_model["input_shape"][0]), int(2 * self.config_model["input_shape"][0])]  #256,512
        strides = torch.tensor(output_size) / torch.tensor(list(x.shape[1:3]))  #[16,16]
        strides = strides.to(x.device)
        raw_xy, raw_wh, raw_conf, raw_prob = x[..., 0:2], x[..., 2:4], x[..., 4:5], x[..., 5:] #[12,16,32,6,2],[12,16,32,6,2],[12,16,32,6,1],[12,16,32,6,6]
        xx, yy = torch.meshgrid([torch.arange(0, x.shape[1]), torch.arange(0, x.shape[2])])  #[16,32],[16,32] x,y分别表示横纵坐标
        xy_grid = [xx.T, yy.T]
        xy_grid = torch.unsqueeze(torch.stack(xy_grid, dim=-1), dim=-2).to(x.device) #[32,16,1,2]
        xy_grid = torch.unsqueeze(rearrange(xy_grid, "b c h w -> c b h w"), dim=0)  #[1,16,32,1,2]
        xy_grid = torch.tile(xy_grid, (x.shape[0], 1, 1, len(self.anchor_boxes), 1)).to(torch.float32)   #[12,16,32,6,2],torch.tile() 函数将 xy_grid 张量沿着指定的维度进行复制
        scale = self.yolohead_xyz_scales[0] 
        pred_xy = ((torch.sigmoid(raw_xy) * scale) - 0.5 * (scale - 1) + xy_grid) * strides
        ###---------------- clipping values --------------------###
        raw_wh = torch.clamp(raw_wh, 1e-12, 1e12)
        ###-----------------------------------------------------###
        pred_wh = torch.exp(raw_wh) * self.anchor_boxes
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)

        pred_conf = torch.sigmoid(raw_conf)
        pred_prob = torch.sigmoid(raw_prob)
        return torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

    def extractYoloInfo(self, yoloformat_data):
        box = yoloformat_data[..., :4]
        conf = yoloformat_data[..., 4:5]
        category = yoloformat_data[..., 5:]
        return box, conf, category

    def loss(self, pred_raw, pred, gt, raw_boxes):
        raw_box, raw_conf, raw_category = self.extractYoloInfo(pred_raw)
        pred_box, pred_conf, pred_category = self.extractYoloInfo(pred)
        gt_box, gt_conf, gt_category = self.extractYoloInfo(gt)

        box_loss = gt_conf * (torch.square(pred_box[..., :2] - gt_box[..., :2]) +
                              torch.square(torch.sqrt(pred_box[..., 2:]) - torch.sqrt(gt_box[..., 2:])))
        iou = tf_iou2d(torch.unsqueeze(pred_box, dim=-2), raw_boxes[:, None, None, None, :, :])
        max_iou = torch.unsqueeze(torch.max(iou, dim=-1)[0], dim=-1)
        gt_conf_negative = (1.0 - gt_conf) * (max_iou < self.config_train["focal_loss_iou_threshold"]).to(torch.float32)
        conf_focal = torch.pow(gt_conf - pred_conf, 2)
        alpha = 0.01

        conf_loss = conf_focal * (gt_conf * self.bce_criterion(target=gt_conf, input=raw_conf)
                                  + alpha * gt_conf_negative * self.bce_criterion(target=gt_conf, input=raw_conf))
        ### NOTE: category loss function ###
        category_loss = gt_conf * self.bce_criterion(target=gt_category, input=raw_category)

        ### NOTE: combine together ###
        box_loss_all = torch.mean(torch.sum(box_loss, dim=[1, 2, 3, 4]))
        box_loss_all *= 1e-1
        conf_loss_all = torch.mean(torch.sum(conf_loss, dim=[1, 2, 3, 4]))
        
        category_loss_all = torch.mean(torch.sum(category_loss, dim=[1, 2, 3, 4]))
        
        total_loss = box_loss_all + conf_loss_all + category_loss_all
        return total_loss, box_loss_all, conf_loss_all, category_loss_all

def tf_iou2d(box_xywh_1, box_xywh_2):
    """ Tensorflow version of 3D bounding box IOU calculation
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]"""
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    ### areas of both boxes
    box1_area = box_xywh_1[..., 2] * box_xywh_1[..., 3]
    box2_area = box_xywh_2[..., 2] * box_xywh_2[..., 3]
    ### find the intersection box
    box1_min = box_xywh_1[..., :2] - box_xywh_1[..., 2:] * 0.5
    box1_max = box_xywh_1[..., :2] + box_xywh_1[..., 2:] * 0.5
    box2_min = box_xywh_2[..., :2] - box_xywh_2[..., 2:] * 0.5
    box2_max = box_xywh_2[..., :2] + box_xywh_2[..., 2:] * 0.5

    left_top = torch.maximum(box1_min, box2_min)
    bottom_right = torch.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = torch.maximum(bottom_right - left_top,
                                 torch.zeros(bottom_right.shape, dtype=bottom_right.dtype, device=box_xywh_1.device))
    intersection_area = intersection[..., 0] * intersection[..., 1]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = torch.nan_to_num(torch.div(intersection_area, union_area + 1e-10), 0.0)
    return iou


if __name__ == "__main__":
    input_features = torch.randn(3, 256, 16, 16)
    config = loader.readConfig(config_file_name="/media/ljm/b930b01d-640a-4b09-8c3c-777d88f63e8b/Dujialun/RADDet_Pytorch-main/config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]

    anchors_layer = anchor_boxes = loader.readAnchorBoxes(
        anchor_boxes_file="/media/ljm/b930b01d-640a-4b09-8c3c-777d88f63e8b/Dujialun/RADDet_Pytorch-main/anchors.txt")

    anchors_layer_cart = anchor_boxes_cart = loader.readAnchorBoxes(
        anchor_boxes_file="/media/ljm/b930b01d-640a-4b09-8c3c-777d88f63e8b/Dujialun/RADDet_Pytorch-main/anchors_cartboxes.txt")  # load anchor boxes with order
    num_classes = len(config_data["all_classes"])

    input_size = list(config_model["input_shape"])
    input_channels = input_size[-1]
    num_class = len(config_data["all_classes"])
    yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
    focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn(3, 64, 256, 256)
    input = input.to(device)

    model = RADDet(config_model, config_data, config_train, anchor_boxes_cart)
    model.to(device)
    bk = model.backbone

    cart_model = RADDetCart(config_model=config_model,
                            config_data=config_data,
                            config_train=config_train,
                            anchor_boxes=anchor_boxes_cart,
                            input_shape=config_model["bk_output_size"],
                            device=device)
    cart_model.to(device)

    out = bk(input)
    pred_raw = cart_model(out)
    pred = cart_model.decodeYolo(pred_raw)

    label = torch.randn(3, 16, 32, 6, 11).to(device)
    raw_box = torch.randn(3, 30, 5).to(device)

    cart_model.loss(pred_raw, pred, label, raw_box[..., :4])

    radar_dataset_3d = RararDataset3D(config_data=config_data,
                                      config_train=config_train,
                                      config_model=config_model,
                                      headoutput_shape=[3, 16, 16, 4, 78],
                                      anchors=anchor_boxes,
                                      anchors_cart=anchor_boxes_cart,
                                      cart_shape=[3, 16, 32, 6, 11])
    for d in radar_dataset_3d:
        data, gt_label, raw_boxes = d
        print(data.shape, gt_label.shape, raw_boxes.shape)




class FPNDetCart(nn.Module):
    def __init__(self, config_model, config_data, config_train, anchor_boxes, input_shape, device, backbone=None,ra=None):
        """ make sure the model is buit when initializint the class.
        Only by this, the graph could be built and the trainable_variables
        could be initialized """
        super(FPNDetCart, self).__init__()
        assert (isinstance(input_shape, tuple) or isinstance(input_shape, list))
        self.config_model = config_model
        self.config_data = config_data
        self.config_train = config_train
        self.input_size = input_shape
        self.num_class = len(config_data["all_classes"])
        self.anchor_boxes = torch.tensor(anchor_boxes).to(device)
        self.yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
        self.focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]

        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.backbone = backbone
        self.RA_decoder= ra
        dense_feature_size = input_shape[0]*input_shape[1]
        self.fc1 = nn.Linear(in_features=dense_feature_size, out_features=dense_feature_size*2, bias=True)
        self.fc2 = nn.Linear(in_features=dense_feature_size*2, out_features=dense_feature_size*2, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=self.input_size[-1],
                               out_channels=self.input_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn1 = nn.BatchNorm2d(self.input_size[-1])

        self.conv2 = nn.Conv2d(in_channels=self.input_size[-1],
                               out_channels=self.input_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn2 = nn.BatchNorm2d(self.input_size[-1])

        self.conv3 = nn.Conv2d(in_channels=self.input_size[-1],
                               out_channels=self.input_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn3 = nn.BatchNorm2d(self.input_size[-1])

        self.conv_yolo_head = nn.Conv2d(in_channels=self.input_size[-1],
                                        out_channels=self.input_size[-1]*2,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True
                                        )
        self.bn_yolo_head = nn.BatchNorm2d(self.input_size[-1]*2)

        self.conv_yolo_head_1 = nn.Conv2d(in_channels=self.input_size[-1]*2,
                                          out_channels=len(self.anchor_boxes) * (self.num_class + 5),
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=True
                                          )

        # TODO: adding weight initialization code.
    def forward(self, x):
        x = self.backbone(x)
        x = self.RA_decoder(x)
        x = rearrange(x, "b c h w -> b c (h w)")
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = rearrange(x, "b c (h w) -> b c h w", h=self.input_size[0])
        res_x = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x + res_x

        x = self.relu(self.bn_yolo_head(self.conv_yolo_head(x)))
        x = self.conv_yolo_head_1(x)
        x = rearrange(x, "b c h w -> b h w c")
        x = rearrange(x, "b h w (c1 c2) -> b h w c1 c2", c1=len(self.anchor_boxes))
        return x

    def decodeYolo(self, x):
        output_size = [int(self.config_model["input_shape"][0]), int(2 * self.config_model["input_shape"][0])]
        strides = torch.tensor(output_size) / torch.tensor(list(x.shape[1:3]))
        strides = strides.to(x.device)
        raw_xy, raw_wh, raw_conf, raw_prob = x[..., 0:2], x[..., 2:4], x[..., 4:5], x[..., 5:]
        xx, yy = torch.meshgrid([torch.arange(0, x.shape[1]), torch.arange(0, x.shape[2])])
        xy_grid = [xx.T, yy.T]
        xy_grid = torch.unsqueeze(torch.stack(xy_grid, dim=-1), dim=-2).to(x.device)
        xy_grid = torch.unsqueeze(rearrange(xy_grid, "b c h w -> c b h w"), dim=0)
        xy_grid = torch.tile(xy_grid, (x.shape[0], 1, 1, len(self.anchor_boxes), 1)).to(torch.float32)
        scale = self.yolohead_xyz_scales[0]
        pred_xy = ((torch.sigmoid(raw_xy) * scale) - 0.5 * (scale - 1) + xy_grid) * strides
        ###---------------- clipping values --------------------###
        raw_wh = torch.clamp(raw_wh, 1e-12, 1e12)
        ###-----------------------------------------------------###
        pred_wh = torch.exp(raw_wh) * self.anchor_boxes
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)

        pred_conf = torch.sigmoid(raw_conf)
        pred_prob = torch.sigmoid(raw_prob)
        return torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

    def extractYoloInfo(self, yoloformat_data):
        box = yoloformat_data[..., :4]
        conf = yoloformat_data[..., 4:5]
        category = yoloformat_data[..., 5:]
        return box, conf, category

    def loss(self, pred_raw, pred, gt, raw_boxes):
        raw_box, raw_conf, raw_category = self.extractYoloInfo(pred_raw)
        pred_box, pred_conf, pred_category = self.extractYoloInfo(pred)
        gt_box, gt_conf, gt_category = self.extractYoloInfo(gt)

        box_loss = gt_conf * (torch.square(pred_box[..., :2] - gt_box[..., :2]) +
                              torch.square(torch.sqrt(pred_box[..., 2:]) - torch.sqrt(gt_box[..., 2:])))
        iou = tf_iou2d(torch.unsqueeze(pred_box, dim=-2), raw_boxes[:, None, None, None, :, :])
        max_iou = torch.unsqueeze(torch.max(iou, dim=-1)[0], dim=-1)
        gt_conf_negative = (1.0 - gt_conf) * (max_iou < self.config_train["focal_loss_iou_threshold"]).to(torch.float32)
        conf_focal = torch.pow(gt_conf - pred_conf, 2)
        alpha = 0.01

        conf_loss = conf_focal * (gt_conf * self.bce_criterion(target=gt_conf, input=raw_conf)
                                  + alpha * gt_conf_negative * self.bce_criterion(target=gt_conf, input=raw_conf))
        ### NOTE: category loss function ###
        category_loss = gt_conf * self.bce_criterion(target=gt_category, input=raw_category)

        ### NOTE: combine together ###
        box_loss_all = torch.mean(torch.sum(box_loss, dim=[1, 2, 3, 4]))
        box_loss_all *= 1e-1
        conf_loss_all = torch.mean(torch.sum(conf_loss, dim=[1, 2, 3, 4]))
        
        category_loss_all = torch.mean(torch.sum(category_loss, dim=[1, 2, 3, 4]))
        total_loss = box_loss_all + conf_loss_all + category_loss_all
        return total_loss, box_loss_all, conf_loss_all, category_loss_all