import argparse
import os
import torch
import numpy as np
import time, math, glob
from PIL import Image
from evaluate import calculate_evaluation_floder
import torchvision
from torchvision.utils import save_image
import fid_score

torch.manual_seed(1850)
parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./checkpoint/model_Denoising__95_50_1.0.pth", type=str, help="model path")


# parser.add_argument("--save", default="./results/rain/R100Loutput/", type=str, help="savepath, Default: results")
# parser.add_argument("--savetar", default="./results/rain/R100Ltarget/", type=str, help="savepath, Default: targets")
# parser.add_argument("--degset", default="./datasets/Deraining/inputcrop/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./datasets/Deraining/labelcrop/", type=str, help="target data")

# parser.add_argument("--save", default="./results/SR/DIVOUT/", type=str, help="savepath, Default: results")
# parser.add_argument("--savetar", default="./results/SR/DIVTAR/", type=str, help="savepath, Default: targets")
#
# parser.add_argument("--degset", default="./datasets/SR/DIVLRX4/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./datasets/SR/DIVHR/", type=str, help="target data")

parser.add_argument("--noise_sigma", default=50, type=int, help="gpu ids (default: 0)")
# parser.add_argument("--save", default="./results/noise/OUT/25/", type=str, help="savepath, Default: results")
# parser.add_argument("--savetar", default="./results/noise/TAR/", type=str, help="savepath, Default: targets")
# parser.add_argument("--degset", default="./datasets/Denoising/CBSD68/original/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./datasets/Denoising/CBSD68/original/", type=str, help="target data")
parser.add_argument("--save", default="./results/noise/OUT/kodak/50/", type=str, help="savepath, Default: results")
parser.add_argument("--savetar", default="./results/noise/TAR/kodak/", type=str, help="savepath, Default: targets")
parser.add_argument("--saveres", default="./results/noise/RES/", type=str, help="savepath, Default: residual")
parser.add_argument("--degset", default="./datasets/Denoising/testKODAK/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./datasets/Denoising/testKODAK/", type=str, help="target data")

parser.add_argument("--gpus", default="0", type=str, help="gpu ids")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if not os.path.exists(opt.save):
    os.mkdir(opt.save)

Tnet = torch.load(opt.model)["Tnet"]
deg_list = glob.glob(opt.degset+"*")
deg_list = sorted(deg_list)

tar_list = sorted(glob.glob(opt.tarset+"*"))
num = len(deg_list)
data_list = []

with torch.no_grad():
    for deg_name, tar_name in zip(deg_list, tar_list):
        name = tar_name.split('/')
        print(name)
        print("Processing ", deg_name)
        deg_img = Image.open(deg_name).convert('RGB')
        tar_img = Image.open(tar_name).convert('RGB')
        deg_img = np.array(deg_img)
        tar_img = np.array(tar_img)
        h,w = deg_img.shape[0],deg_img.shape[1]
        shape1 = deg_img.shape
        shape2 = tar_img.shape
        if (h%4) or (w%4) != 0:
            deg_img = deg_img[1:h, 1:w]
            tar_img = tar_img[1:h, 1:w]
        if shape1 != shape2:
            continue
        deg_img = np.transpose(deg_img, (2, 0, 1))
        deg_img = torch.from_numpy(deg_img).float() / 255
        deg_img = deg_img.unsqueeze(0)

        tar_img = np.transpose(tar_img, (2, 0, 1))
        noise = np.random.normal(size=tar_img.shape) * opt.noise_sigma / 255.0
        noise = torch.from_numpy(noise).float()

        tar_img = torch.from_numpy(tar_img).float() / 255
        tar_img = tar_img.unsqueeze(0)
        gt = tar_img
        deg_img = deg_img + noise
        data_degraded = deg_img
        if cuda:
            Tnet = Tnet.cuda()
            gt=gt.cuda()
            data_degraded = data_degraded.cuda()
        else:
            Tnet = Tnet.cpu()

        start_time = time.time()

        im_output = torch.zeros(size=data_degraded.shape)
        im_output = Tnet(data_degraded)
        res = data_degraded - im_output
        im_output = im_output.cpu()
        res = res.cpu()

        save_image(res.data*3, opt.saveres + '/' + name[-1])
        save_image(im_output.data,opt.save+'/'+name[-1])
        save_image(tar_img.data, opt.savetar+'/'+name[-1])

inception_model = torchvision.models.inception_v3(pretrained=True)
fid_value = fid_score.calculate_fid_given_paths([opt.savetar, opt.save], batch_size=50,
                                                device='cuda', dims=2048, num_workers=0)
print('FID value:', fid_value)

psnr, ssim, pmax, smax, pmin, smin=calculate_evaluation_floder(opt.savetar,opt.save)
print("PSNR: Averyge {:.5f},   best {:.5f},   worst {:.5f}".format(psnr, pmax, pmin))
print("SSIM: Averyge {:.5f},   best {:.5f},   worst {:.5f}".format(ssim, smax, smin))
