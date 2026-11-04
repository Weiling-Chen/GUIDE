import time

import numpy as np
import os,sys
import argparse
from tqdm import tqdm
import os

# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'D:\anaconda3\pkgs\qt-5.9.7-vc14h73c81de_0\Library\plugins'
from torch.utils.data import DataLoader
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
from dataset.dataset import *
import utils
import math
from model import *
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from collections import OrderedDict

parser = argparse.ArgumentParser(description='GUIDE')
# 输入图像文件夹
parser.add_argument('--input_dir', default=r'D:\zjh\GUIDE\GUIDE-code\input' ,type=str, help='Directory of validation images')
# 输出图像文件夹
parser.add_argument('--result_dir', default=r'D:\zjh\GUIDE\GUIDE-code\temp',
    type=str, help='Directory for results')
# 基础模型权重
parser.add_argument('--base_model', default=r'D:\zjh\GUIDE\GUIDE-code\weights\base_model\turbid.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Restormer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
#modify表示是否要加入插件
parser.add_argument('--modify', action='store_true', default=True, help='modify')
# adapter权重
parser.add_argument('--adapter',default=r'D:\zjh\GUIDE\GUIDE-code\weights\adapters\color.pth',
    type=str, help='Path to weights')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

class MixReGhos_Block(nn.Module):
    def __init__(self, block, kernel_size=[1,3,5]):
        super(MixReGhos_Block, self).__init__()
        self.original_block = deepcopy(block)
        for param in self.original_block.parameters():
            param.requires_grad = False
        self.out_channels = self.original_block.out_channels
        self.Ghos_Mod = nn.Sequential(
            GroupedDepthwiseConv2d(self.out_channels,self.out_channels,kernel_sizes=kernel_size)
        )

    def forward(self, x):
        output = self.Ghos_Mod(self.original_block(x))
        return output

class GroupedDepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1):
        super(GroupedDepthwiseConv2d, self).__init__()

        self.in_channels = in_channels
        self.first = self.second = in_channels // 3
        self.third = in_channels - in_channels // 3 * 2
        # 创建多个深度卷积模块
        # self.depthwise_convs = nn.ModuleList([
        #     nn.Conv2d(in_channels=self.in_channels // 3, out_channels=self.in_channels // 3, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=self.in_channels//3, bias=False)
        #     for kernel_size in kernel_sizes
        # ])
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(in_channels=self.first, out_channels=self.first, kernel_size=kernel_sizes[0],
                      stride=stride, padding=kernel_sizes[0] // 2, groups=self.first, bias=False),
            nn.Conv2d(in_channels=self.second, out_channels=self.second, kernel_size=kernel_sizes[1],
                      stride=stride, padding=kernel_sizes[1] // 2, groups=self.second, bias=False),
            nn.Conv2d(in_channels=self.third, out_channels=self.third, kernel_size=kernel_sizes[2],
                      stride=stride, padding=kernel_sizes[2] // 2, groups=self.third, bias=False)

        ])
        nn.init.dirac_(self.depthwise_convs[0].weight, groups=self.first)
        nn.init.dirac_(self.depthwise_convs[1].weight, groups=self.second)
        nn.init.dirac_(self.depthwise_convs[2].weight, groups=self.third)

    def forward(self, x):
        # 将输入数据按通道拆分
        # b,c,h,w = x.shape
        x_split = torch.split(x, [self.first,self.second,self.third], dim=1)
        # 对每个分组应用深度卷积
        out_list = [
            conv(x_part)  # 卷积 -> BN -> ReLU
            for conv, x_part in zip(self.depthwise_convs, x_split)
        ]
        # 将结果沿通道方向拼接
        out = torch.cat(out_list, dim=1)

        return out

def add_ghostnet(model, model_type="mixdepth"):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            if model_type == "mixdepth":
                if layer.kernel_size != (1,1):
                    model._modules[name] = MixReGhos_Block(layer,[1,3,5])
                # else:
                #     model._modules[name] = MixReGhos_Block(layer, [1,1,1])
        else:
            add_ghostnet(layer, model_type)
    return model




from copy import deepcopy
utils.mkdir(args.result_dir)
model_restoration = Restormer(LayerNorm_type='BiasFree')
if args.modify:

    checkpoint = torch.load(args.base_model)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if 'module.' in k else k
        new_state_dict[name] = v
    model_restoration.load_state_dict(new_state_dict, strict=True)

    model_restoration = add_ghostnet(model_restoration)
    checkpoint_extra = torch.load(args.adapter)
    state_dict = checkpoint_extra["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if 'module.' in k else k
        new_state_dict[name] = v
    model_restoration.load_state_dict(new_state_dict, strict=False)
else:
    utils.load_checkpoint(model_restoration,args.base_model)



model_restoration.cuda()
model_restoration.eval()

def expand2square(timg,factor=8):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask


test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle= False,num_workers=0, drop_last=False)

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb =[]
    for ii, data_test in enumerate(tqdm(test_loader),0):
        rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))


        filenames = data_test[2]


        ## restormer
        factor  = 8
        rgb_noisy = data_test[1].cuda()
        h, w = rgb_noisy.shape[2], rgb_noisy.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = rgb_restored[:, :, :h, :w]
        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()


        psnr = psnr_loss(rgb_restored, rgb_gt)
        ssim = ssim_loss(rgb_restored, rgb_gt, channel_axis=2,data_range = 1)
        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)
        print("PSNR:", psnr, ", SSIM:", ssim, filenames[0], rgb_restored.shape)
        utils.save_img(os.path.join(args.result_dir, filenames[0]), img_as_ubyte(rgb_restored))
        with open(os.path.join(args.result_dir, 'psnr_ssim.txt'), 'a') as f:
            f.write(filenames[0] + '.PNG ---->' + "PSNR: %.4f, SSIM: %.4f] " % (psnr, ssim) + '\n')
psnr_val_rgb = sum(psnr_val_rgb) / len(test_dataset)
ssim_val_rgb = sum(ssim_val_rgb) / len(test_dataset)
print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
with open(os.path.join(args.result_dir, 'psnr_ssim.txt'), 'a') as f:
    f.write("Arch:" + args.arch + ", PSNR: %.4f, SSIM: %.4f] " % (psnr_val_rgb, ssim_val_rgb) + '\n')
