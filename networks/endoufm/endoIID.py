import os
import torch
import torch.nn as nn
import networks.backbones as backbones
from networks.backbones.mylora import Linear as LoraLinear
from networks.backbones.mylora import DVLinear as DVLinear
from networks.backbones.mylora import RVLinear as RVLinear
from .layers import mark_only_part_as_trainable,_make_scratch, _make_fusion_block,Interpolate
import math

def hsv_to_rgb(hsv):
        h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
        #对出界值的处理
        h = h%1
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
  
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb

def rgb_to_hsv(img):

    eps=1e-8
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + eps)
    saturation[ img.max(1)[0]==0 ] = 0

    value = img.max(1)[0]
    
    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value],dim=1)
    return hsv

class HeadAlbedo(nn.Module):
    def __init__(self, features):
        super(HeadAlbedo, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            # nn.Identity(),
        )

    def forward(self, x):
        x = self.head(x)
        return x

class Headlight(nn.Module):
    def __init__(self, features):
        super(Headlight, self).__init__()
        self.upsample = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1= nn.Conv2d(features, 16, kernel_size=3, stride=1, padding=1)
        self.conv2= nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu=nn.ReLU()
        self.conv3= nn.Conv2d(32, 1, kernel_size=1, stride=1)
    def forward(self, x, image):
        x = self.upsample(x)
        x=self.conv1(x)
        x=self.relu(x)
        image=self.conv2(image)
        image=self.relu(image)
        x = torch.cat((x, image), dim=1)
        x=self.conv3(x)
        return x

class HeadAlbedoHS(nn.Module):
    def __init__(self, features):
        super(HeadAlbedoHS, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            # nn.Identity(),
        )

    def forward(self, x):
        x = self.head(x)
        return x

class DPTHead(nn.Module):
    def __init__(self, in_channels, features=128, use_bn=False, out_channels=[96, 192, 384, 768], use_clstoken=False):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.conv_light = Headlight(features)
        #RGB
        self.conv_albedo = HeadAlbedo(features)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, out_features, patch_h, patch_w, image):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            # x= self.transformer_block[i](x,sgement_embeddings[i])

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        light = self.sigmoid(self.conv_light(path_1,image))
        #RGB
        albedo = self.sigmoid(self.conv_albedo(path_1))

        return albedo,light
    
class endoiid(nn.Module):
    """Applies low-rank adaptation to a ViT model's image encoder.

    Args:
        backbone_size: size of pretrained Dinov2 choice from: "small", "base", "large", "giant"
        r: rank of LoRA
        image_shape: input image shape, h,w need to be multiplier of 14, default:(224,280)
        lora_layer: which layer we apply LoRA.
    """

    def __init__(self, 
                 backbone_size = "base", 
                 r=4, 
                 image_shape=(224,280), 
                 lora_type="lora",
                 pretrained_path=None,
                 residual_block_indexes=[],
                 include_cls_token=True,
                 use_cls_token=False,
                 use_bn=False,
                 ):
        super(endoiid, self).__init__()

        assert r > 0
        self.r = r
        self.backbone_size = backbone_size
        self.backbone = {
            "small": backbones.vits.vit_small(residual_block_indexes=residual_block_indexes,
                                              include_cls_token=include_cls_token,
                                              ),
            "base": backbones.vits.vit_base(residual_block_indexes=residual_block_indexes,
                                            include_cls_token=include_cls_token,
                                            ),
        }
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
        }
        self.decompose_head_features = {
            "small": 64,
            "base": 128,
        }
        self.decompose_head_out_channels = {
            "small": [48, 96, 192, 384],
            "base": [96, 192, 384, 768],
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.decompose_head_feature = self.decompose_head_features[self.backbone_size]
        self.decompose_head_out_channel = self.decompose_head_out_channels[self.backbone_size]
        encoder = self.backbone[self.backbone_size]

        self.image_shape = image_shape
        
        if lora_type != "none":
            for t_layer_i, blk in enumerate(encoder.blocks):
                mlp_in_features = blk.mlp.fc1.in_features
                mlp_hidden_features = blk.mlp.fc1.out_features
                mlp_out_features = blk.mlp.fc2.out_features
                if t_layer_i==0:
                    random_1_1=torch.zeros((self.r, 1))
                    random_1_1=torch.nn.init.kaiming_uniform_(random_1_1, a=math.sqrt(5))
                    random_1_2=torch.zeros((mlp_hidden_features, 1))
                    random_1_2=torch.nn.init.kaiming_uniform_(random_1_2, a=math.sqrt(5))
                    random_2_1=torch.zeros((self.r, 1))
                    random_2_1=torch.nn.init.kaiming_uniform_(random_2_1, a=math.sqrt(5))
                    random_2_2=torch.zeros((mlp_out_features, 1))
                    random_2_2=torch.nn.init.kaiming_uniform_(random_2_2, a=math.sqrt(5))
                if lora_type == "dvlora":
                    blk.mlp.fc1 = DVLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=self.r)
                    blk.mlp.fc2 = DVLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=self.r)
                elif lora_type == "lora":
                    blk.mlp.fc1 = LoraLinear(mlp_in_features, mlp_hidden_features, r=self.r)
                    blk.mlp.fc2 = LoraLinear(mlp_hidden_features, mlp_out_features, r=self.r)
                elif lora_type == "rvlora":
                    blk.mlp.fc1 = RVLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=self.r,random1=random_1_1,random2=random_1_2)
                    blk.mlp.fc2 = RVLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=self.r,random1=random_2_1,random2=random_2_2)
            
        self.encoder = encoder
        self.decompose_head = DPTHead(self.embedding_dim, self.decompose_head_feature, use_bn, out_channels=self.decompose_head_out_channel, use_clstoken=use_cls_token)
        
        if pretrained_path is not None:
            pretrained_path = os.path.join(pretrained_path, "depth_anything_{}.pth".format(self.backbone_arch))
            pretrained_dict = torch.load(pretrained_path)
            model_dict = self.state_dict()
            self.load_state_dict(pretrained_dict, strict=False)
            print("load pretrained weight from {}\n".format(pretrained_path))

        # mark_only_part_as_trainable(self.encoder)
        # mark_only_part_as_trainable(self.depth_head)
    def forward(self, image):
        
        pixel_values = torch.nn.functional.interpolate(image, size=self.image_shape, mode="bilinear", align_corners=True)
        h, w = pixel_values.shape[-2:]
        
        features = self.encoder.get_intermediate_layers(pixel_values, 4, return_class_token=True)
        patch_h, patch_w = h // 14, w // 14
        
        albedo,light = self.decompose_head(features, patch_h, patch_w,image)

        return albedo,light
