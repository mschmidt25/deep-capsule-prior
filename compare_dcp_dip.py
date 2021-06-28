"""
In this script I want to compare the ability of the DeepCapsulePrior and the DeepImagePrior to represent natural (possible noisy)
images.

"""


import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np 
from numpy.fft import fft2, ifft2

import torch
from torch.optim import Adam
from torch.nn import MSELoss

from scipy import misc
import matplotlib.pyplot as plt 
from skimage.transform import rescale

from dival.reconstructors.networks.unet import UNet
from dcp.utils.models import get_capsule_skip_model

from tqdm import tqdm
import imageio

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import pandas as pd 

import argparse

parser = argparse.ArgumentParser(description='Deep Image Prior vs Deep Capsule Prior')
parser.add_argument('type',choices=["dcp", "dip"], help='type of model')

args = parser.parse_args()

if args.type == "dcp":
    model_name = "Deep Capsule Prior"
    save_img_path = "img_dcp"
else:
    model_name = "Deep Image Prior"
    save_img_path = "img_dip"

def azimuthalAverage(img):
    y, x = np.indices(img.shape)
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = img.flat[ind]
    
    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)
    
    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr
    
    return radial_prof


### Load Image
img = misc.face()
img = rescale(img, 0.25,multichannel=True)

image_fft = fft2(img.transpose(2,0,1))
image_fftshift = np.fft.fftshift(image_fft)
power_spectrum = np.abs(image_fftshift)**2
azimuthMean = np.mean(np.asarray([azimuthalAverage(power_spectrum[i,:,:]) for i in range(3)]),axis=0)


print("shape of image: ", img.shape)

#plt.figure()
#plt.title("Groundtruth image of shape " + str(img.shape))
#plt.imshow(img)
#plt.show()

img_torch = torch.from_numpy(img)
img_torch = img_torch.permute(2,0,1).unsqueeze(0)
print("shape of image in torch: ", img_torch.shape)
### Construct DCP / DIP 

input_depth = 3 
output_depth = 3 
device = "cuda"

net_input = 0.1 * torch.randn(*img_torch.shape).to(device)

print("shape of net_input: ", net_input.shape)

if args.type == "dip":
    scales=4
    skip=4
    channels=(16, 32, 32, 64, 128, 128)
    skip_channels = [skip] * (scales)

    model = UNet(
        input_depth,
        output_depth,
        channels=channels[:scales],
        skip_channels=skip_channels[:scales],
        use_sigmoid=True,
        use_norm=True).to(device)
else:
    capsules = [[8, 2], [16, 2], [32, 2], [64, 2]]
    skip_capsules = [[4, 2], [4, 2], [4, 2], [4, 2]] 
    scales = 3#4
    eps = 1e-6
    use_bias = True 
    weight_init = 'xavier_uniform'
    same_filter = False 
    iter_rout = 3

    model = get_capsule_skip_model(
                input_depth,
                output_depth,
                capsules=capsules[:scales],
                skip_capsules=skip_capsules[:scales],
                iter_rout=iter_rout,
                weight_init=weight_init,
                same_filter=same_filter,
                use_bias=use_bias,
                eps=eps).to(device)

criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)

params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))

print("Number of trainable params: ", sum(p.numel() for p in params_trainable))


num_iters = 500

img_torch = img_torch.to(device).float()


mse_loss = []
psnr_loss = [] 
ssim_loss = [] 
for i in tqdm(range(num_iters)):
    optimizer.zero_grad()

    if args.type == "dcp":
        _, out = model(net_input)
    else: 
        out = model(net_input)
    loss = criterion(out, img_torch)

    loss.backward()
    mse_loss.append(loss.item())
    
    ### psnr and ssim 
    pred = out.detach().cpu().numpy()[0].transpose(1, 2, 0)
    gt = img_torch.detach().cpu().numpy()[0].transpose(1, 2, 0)
    psnr = peak_signal_noise_ratio(gt, pred, data_range=np.max(gt) - np.min(gt))
    ssim = structural_similarity(gt, pred,multichannel=True, data_range=np.max(gt) - np.min(gt))

    psnr_loss.append(psnr)
    ssim_loss.append(ssim)

    optimizer.step()

    if i % 10 == 0:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(13,6))
        fig.suptitle(model_name)
        f = img_torch.detach().cpu().numpy()[0,:,:,:]
        ax1.imshow(f.transpose(1, 2, 0))
        ax1.set_title("GT")
        img = out.detach().cpu().numpy()[0, :, :, :]
        ax2.imshow(img.transpose(1, 2, 0))
        ax2.set_title("Reconstruction " + str(i))

        #image_fft = fft2(img)
        #image_fftshift = np.fft.fftshift(image_fft)
        #power_spectrum = np.abs(image_fftshift)**2
        #azimuthMean_pred = np.mean(np.asarray([azimuthalAverage(power_spectrum[i,:,:]) for i in range(3)]),axis=0)

        #ax3.loglog(azimuthMean, label="GT")
        #ax3.loglog(azimuthMean_pred, label="reco")
        #ax3.set_ylim(np.min(azimuthMean), np.max(azimuthMean))
        #ax3.legend()
        #ax3.set_title("azimuthal average of power spectrum")
        plt.savefig(os.path.join(save_img_path,"iteration_{}.png".format(i)))

        plt.close()

df = pd.DataFrame({"mse": mse_loss, "psnr": psnr_loss, "ssim": ssim_loss})
df.to_csv(str(args.type) + "_quality.csv")

images_names = []
for filename in os.listdir(save_img_path):
    images_names.append(filename)


images_names.sort(key = lambda x: int(x.split(".")[0].split("_")[-1]))
images = []
for name in images_names:
    images.append(imageio.imread(os.path.join(save_img_path, name)))
imageio.mimsave(str(args.type) + '.gif', images)


fig, (ax1, ax2, ax3) = plt.subplots(1,3)

fig.suptitle(model_name)

ax1.semilogy(mse_loss)
ax1.set_title("MSE Loss")
ax1.set_xlabel("Iteration")

ax2.plot(psnr_loss)
ax2.set_title("PSNR")
ax2.set_xlabel("Iteration")

ax3.plot(ssim_loss)
ax3.set_title("SSIM")
ax3.set_xlabel("Iteration")


plt.show()