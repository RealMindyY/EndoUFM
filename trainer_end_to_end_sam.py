from __future__ import absolute_import, division, print_function

import time
import json
import datasets
import networks
import numpy as np
import torch.optim as optim
import torch.nn as nn
from rotation_warp import inverse_rotation_warp
from utils import *
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import cv2
from torchvision.transforms import Resize,Grayscale
from torchvision.utils import flow_to_image
import torch.nn.functional as F
import matplotlib.colors as mcolors
import networks.endoufm as endoufm
# from SAM_Med2D.segment_anything import sam_model_registry



class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}  # 字典
        self.parameters_to_train = []  # 列表
        self.parameters_to_train_1 = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["depth_model"] = endoufm.endoufm(
            backbone_size = "base", r=self.opt.lora_rank, lora_type=self.opt.lora_type,
            image_shape=(224,280), pretrained_path=self.opt.pretrained_path,
            residual_block_indexes=self.opt.residual_block_indexes,
            include_cls_token=self.opt.include_cls_token,
            )
        self.models["depth_model"].to(self.device)
        self.parameters_to_train += list(self.models["depth_model"].parameters())
        # self.parameters_to_train += list(filter(lambda p: p.requires_grad, self.models["depth_model"].parameters()))

        self.models["decompose_encoder"] = endoufm.sam_encoder(
            backbone_size = "base", r=self.opt.lora_rank, lora_type=self.opt.lora_type,
            image_shape=(224,280), pretrained_path=self.opt.pretrained_path,
            residual_block_indexes=self.opt.residual_block_indexes,
            include_cls_token=self.opt.include_cls_token,
            )
        self.models["decompose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["decompose_encoder"].parameters())
        
        self.models["decompose_decoder"] = endoufm.decompose_decoder(
            backbone_size = "base", r=self.opt.lora_rank, lora_type=self.opt.lora_type,
            image_shape=(224,280), pretrained_path=self.opt.pretrained_path,
            residual_block_indexes=self.opt.residual_block_indexes,
            include_cls_token=self.opt.include_cls_token,
            )
        self.models["decompose_decoder"].to(self.device)
        self.parameters_to_train += list(self.models["decompose_decoder"].parameters())


        self.models["adjust_net"]=networks.adjust_net()
        self.models["adjust_net"].to(self.device)
        self.parameters_to_train += list(self.models["adjust_net"].parameters())

        if self.use_pose_net:

            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())


        self.model_optimizer = optim.Adam(self.parameters_to_train,self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, [self.opt.scheduler_step_size], 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset,
                        #  "hamlyn":datasets.hamlynRAWDataset,
                         }
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        test_filenames=readlines(fpath.format("test"))
        img_ext = '.png'  
        if self.opt.split=='hamlyn':
            test_filenames=readlines(os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "test_files_20.txt"))
            img_ext = '.jpg' 

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)
        test_dataset = self.dataset(
            self.opt.data_path, test_filenames, self.opt.height, self.opt.width,
            [0], 4, is_train=False, img_ext=img_ext)
        self.test_loader = DataLoader(
            test_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=False)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)
        
        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()


    def set_train(self):
        """Convert all models to training mode
        """
        if self.step < self.opt.warm_up_step:
            warm_up = True
        else:
            warm_up = False
        
        for name, param in self.models["depth_model"].named_parameters():
            if "seed_" not in name:
                param.requires_grad = True
        endoufm.mark_only_part_as_trainable_rvlora(self.models["depth_model"])
    
        for name, param in self.models["decompose_decoder"].named_parameters():
            if "seed_" not in name:
                param.requires_grad = True
        endoufm.mark_only_part_as_trainable_rvlora(self.models["decompose_decoder"])

        
        for name, param in self.models["decompose_encoder"].named_parameters():
            if "seed_" not in name:
                param.requires_grad = True
        endoufm.mark_only_part_as_trainable_rvlora(self.models["decompose_encoder"])
        
        if self.step==0:
            Total_params = 0
            Trainable_params = 0
            NonTrainable_params = 0
            for name, param in self.models["depth_model"].named_parameters():
                mulValue = np.prod(param.size())
                Total_params += mulValue
                if param.requires_grad == False:
                    NonTrainable_params += mulValue
                    # print(name)
                else:
                    Trainable_params += mulValue

            # for name, param in self.models["decompose_decoder"].named_parameters():
            #     mulValue = np.prod(param.size())
            #     Total_params += mulValue
            #     if param.requires_grad == False:
            #         NonTrainable_params += mulValue
            #         # print(name)
            #     else:
            #         Trainable_params += mulValue
            
            
            # for name, param in self.models["decompose_encoder"].named_parameters():
            #     mulValue = np.prod(param.size())
            #     Total_params += mulValue
            #     if param.requires_grad == False:
            #         NonTrainable_params += mulValue
            #         # print(name)
            #     else:
            #         Trainable_params += mulValue
            
            print(f'Total params: {Total_params}')
            print(f'Trainable params: {Trainable_params}')
            print(f'Non-trainable params: {NonTrainable_params}')
            print(f'Trainable params ratio: {100 * Trainable_params / Total_params}%')

        for param in self.models["pose_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["pose"].parameters():
            param.requires_grad = True
        
        for param in self.models["adjust_net"].parameters():
            param.requires_grad = True
        

        self.models["depth_model"].train()
        self.models["pose_encoder"].train()
        self.models["pose"].train()
        # self.models["decompose_model"].train()
        self.models["decompose_encoder"].train()
        self.models["decompose_decoder"].train()
        self.models["adjust_net"].train()
        # self.models["SAM_encoder"].train()


    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.models["depth_model"].eval()
        self.models["pose_encoder"].eval()
        self.models["pose"].eval()
        # self.models["decompose_model"].eval()
        self.models["decompose_encoder"].eval()
        self.models["decompose_decoder"].eval()
        self.models["adjust_net"].eval()
        # self.models["SAM_encoder"].eval()
       

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.epoch >= (self.opt.num_epochs-10): # and self.epoch >= (self.opt.num_epochs-10)
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        print(self.model_optimizer.param_groups[0]['lr'])

        # grad_step = 4
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            # depth, pose, transform
            self.set_train()
            outputs, losses = self.process_batch(inputs)

            # loss = losses["loss"] / grad_step  # 梯度归一化
            # loss.backward()
            # # 每 N 步执行一次优化器更新
            # if (batch_idx + 1) % grad_step == 0 or (batch_idx + 1) == len(self.train_loader):
            #     self.model_optimizer.step()
            #     self.model_optimizer.zero_grad()

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:

                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
        self.test()
        self.model_lr_scheduler.step()


    def test(self):
        self.set_eval()
        pred_disps=[]
        with torch.no_grad():
            for inputs in self.test_loader:
                input_color = inputs[("color", 0, 0)].to(self.device)
                output = self.models["depth_model"](input_color)
                pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_disps.append(pred_disp)
        pred_disps = torch.cat(pred_disps)
        gt_path = os.path.join(os.path.dirname(__file__), "splits", self.opt.split,"gt_depths.npz")
        if self.opt.split=='hamlyn':
            gt_path = os.path.join(os.path.dirname(__file__), "splits", self.opt.split,"gt_depths_20.npz")
        gt_depths = torch.tensor(np.load(gt_path, fix_imports=True, encoding='latin1')["data"]).to(self.device)
        errors = []
        ratios = []
        for i in range(pred_disps.shape[0]):
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = pred_disps[i]
            torch_resize = Resize([gt_height,gt_width]) # 定义Resize类对象
            pred_disp = torch_resize(pred_disp).squeeze()

            pred_depth = 1/pred_disp

            mask = torch.logical_and(gt_depth > self.opt.min_depth, gt_depth < self.opt.max_depth)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= self.opt.pred_depth_scale_factor
            ratio = torch.median(gt_depth) / torch.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

            pred_depth[pred_depth < self.opt.min_depth] = self.opt.min_depth
            pred_depth[pred_depth > self.opt.max_depth] =self.opt.max_depth

            error = compute_depth_errors(gt_depth, pred_depth)
            if not np.isnan(torch.tensor(error)).all():
                errors.append(error)

            # errors.append(compute_depth_errors(gt_depth, pred_depth))
        mean_errors = torch.tensor(errors).mean(0)
        file = open(os.path.join(self.opt.log_dir,self.opt.model_name+'.txt'), 'a')
        file.write(("&{: 8.3f}  " * 8+"\n").format(*mean_errors.tolist()))
        file.close()
        self.set_train()


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # features = self.models["encoder"](inputs["color_aug", 0, 0])
            # outputs = self.models["depth"](features)


            outputs = self.models["depth_model"](inputs["color_aug", 0, 0])
            

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs,outputs))

        self.decompose(inputs,outputs)

        losses = self.compute_losses(inputs, outputs)


        return outputs, losses

    def predict_poses(self, inputs,disps):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:

            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
                
            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":
                    if f_i < 0:
                        inputs_all = [pose_feats[f_i], pose_feats[0]]
                    else:
                        inputs_all = [pose_feats[0], pose_feats[f_i]]                                                                    

                    # pose
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0],invert=(f_i < 0))
                       
        return outputs

    #scale
    def decompose(self,inputs,outputs):
        for f_i in self.opt.frame_ids:
            # decompose_features = self.models["decompose_encoder"](inputs[("color_aug",f_i,0)])
            # outputs[("reflectance",0,f_i)],outputs[("light",0,f_i)]=self.models["decompose"](decompose_features)
            # outputs[("reflectance",0,f_i)],outputs[("light",0,f_i)]=self.models["decompose_model"](inputs[("color_aug",f_i,0)])
            features, h, w = self.models["decompose_encoder"](inputs[("color_aug",f_i,0)])
            outputs[("reflectance",0,f_i)], outputs[("light",0,f_i)]=self.models["decompose_decoder"](features, h, w, inputs[("color_aug",f_i,0)])
            outputs[("reprojection_color", 0, f_i)]=outputs[("reflectance", 0, f_i)]*outputs[("light", 0, f_i)] 
        
        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            
            disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            source_scale = 0

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                cam_points = self.backproject_depth[0](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[0](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("warp", scale, frame_id)] = pix_coords

                outputs[("reflectance_warp", scale, frame_id)] = F.grid_sample(
                    outputs[("reflectance", 0, frame_id)],
                    outputs[("warp", scale, frame_id)],
                    padding_mode="border",align_corners=True)
                
                outputs[("light_warp", scale, frame_id)] = F.grid_sample(
                    outputs[("light", 0, frame_id)],
                    outputs[("warp", scale, frame_id)],
                    padding_mode="border",align_corners=True)
                
                outputs[("color_warp", scale, frame_id)]=F.grid_sample(
                    inputs[("color_aug", frame_id, 0)],
                    outputs[("warp", scale, frame_id)],
                    padding_mode="border",align_corners=True)
                
                # masking zero values
                mask_ones=torch.ones_like(inputs[("color_aug", frame_id, 0)])
                mask_warp=F.grid_sample(
                    mask_ones,
                    outputs[("warp", scale, frame_id)],
                    padding_mode="zeros",align_corners=True)
                valid_mask = (mask_warp.abs().mean(dim=1, keepdim=True) > 0.0).float()
                # valid_mask_tgt = (inputs[("color_aug",0,0)].abs().mean(dim=1, keepdim=True) > 1e-5).float()
                # valid_mask =  valid_mask_ref
                outputs[("valid_mask", scale, frame_id)] = valid_mask
                
                outputs[("warp_diff_color", scale, frame_id)]=torch.abs(inputs[("color_aug",0,0)]-outputs[("color_warp",scale,frame_id)])*valid_mask
                outputs[("transform", scale, frame_id)] = self.models["adjust_net"](outputs[("warp_diff_color", scale, frame_id)])
            
                outputs[("light_adjust_warp", scale, frame_id)] = outputs[("transform", scale, frame_id)] + outputs[("light_warp", scale, frame_id)] 
                outputs[("light_adjust_warp", scale, frame_id)] = torch.clamp(outputs[("light_adjust_warp", scale, frame_id)], min=0.0, max=1.0)

                outputs[("reprojection_color_warp", scale, frame_id)]=outputs[("reflectance_warp", scale, frame_id)]*outputs[("light_adjust_warp", scale, frame_id)]

    
    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    #scale
    def compute_losses(self, inputs, outputs):

        losses = {}
        total_loss = 0
     
        for scale in self.opt.scales:
            loss_reflec=0
            loss_reprojection=0
            loss_disp_smooth=0
            loss_reconstruction=0
            loss_light=0
        
            for frame_id in self.opt.frame_ids:
                loss_reconstruction += (self.compute_reprojection_loss(inputs[("color_aug", frame_id, 0)], outputs[("reprojection_color", 0, frame_id)])).mean()

            for frame_id in self.opt.frame_ids[1:]: 
                mask = outputs[("valid_mask", scale, frame_id)].detach()
                loss_reflec += (torch.abs(outputs[("reflectance",0,0)] - outputs[("reflectance_warp", scale, frame_id)]).mean(1, True) * mask).sum() / mask.sum()
                loss_reprojection += (self.compute_reprojection_loss(inputs[("color_aug", 0, 0)], outputs[("reprojection_color_warp", scale, frame_id)]) * mask).sum() / mask.sum()
                loss_light += (torch.abs(outputs[("light",0,0)] - outputs[("light_warp", scale, frame_id)]).mean(1, True) * mask).sum() / mask.sum()

            disp = outputs[("disp", scale)]
            color = inputs[("color_aug", 0, scale)]
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            loss_disp_smooth = get_smooth_loss(norm_disp, color)

            if scale==0:
                seg_mask = inputs[("seg_mask", 0, scale)].unsqueeze(dim=1)
                loss_disp_smooth_masked = get_smooth_loss_masked(norm_disp, color, seg_mask)
                total_loss+=0.003*loss_disp_smooth_masked
        
            # total_loss +=  loss_reprojection / 2.0 + 0.2*(loss_reflec / 2.0) +0.01*(loss_disp_smooth/(2 ** scale))+0.2*(loss_reconstruction/3.0)
            total_loss +=  loss_reprojection / 2.0 + 0.2*(loss_reflec / 2.0) +0.2*(loss_reconstruction/3.0)
            #权重
            
            if scale==0:
                losses["loss/loss_reflec"] = loss_reflec
                losses["loss/loss_reprojection"] = loss_reprojection
                losses["loss/loss_disp_smooth"] = loss_disp_smooth
                losses["loss/loss_reconstruction"] = loss_reconstruction
                losses["loss/loss_light"] = loss_light
        
        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()


    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            
            for s in self.opt.scales:
                writer.add_image(
                    "disp_{}/{}".format(s, j),
                   visualize_depth(outputs[("disp", s)][j]), self.step)
            for frame_id in self.opt.frame_ids:
                
                writer.add_image(
                        "inputaug_{}_{}/{}".format(frame_id, 0, j),
                        inputs[("color_aug", frame_id, 0)][j].data, self.step)
                writer.add_image(
                        "input_{}_{}/{}".format(frame_id, 0, j),
                        inputs[("color", frame_id, 0)][j].data, self.step)
                    

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)


    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
            # print("Loading Adam weights")
            # optimizer_dict = torch.load(optimizer_load_path)
            # self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        print("Adam is randomly initialized")
