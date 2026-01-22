import math
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import torchvision.transforms as transforms
import functools
from PIL import Image
import matplotlib.patches as patches
from torch.optim.lr_scheduler import _LRScheduler
import torchdiffeq
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import requests
from astroquery.skyview import SkyView # Recommended alternative module
from astropy.coordinates import Longitude, Latitude, Angle
from astroquery.hips2fits import hips2fits
from io import BytesIO
from tqdm.notebook import tqdm
import warnings
import random
import json
from model import *
from platesloader import *
from astropy.convolution import convolve_fft, Gaussian2DKernel

def psf_image(image_data, psf_sigma=3.0):
    """
    Applies a Point Spread Function (PSF) to an astronomical image 
    using FFT convolution.

    Parameters:
    image_data (np.ndarray): The input astronomical image data (2D NumPy array).
    psf_kernel (np.ndarray or astropy.convolution.Kernel): 
        The PSF kernel (2D NumPy array or Astropy kernel object). 
        It is recommended that the PSF is normalized so the sum of its values is 1.0.

    Returns:
    np.ndarray: The convolved image data (2D NumPy array).
    """
    # Ensure the image data is a float type for convolution
    if image_data.dtype != float:
        image_data = image_data.astype(float)

    psf_kernel = Gaussian2DKernel(psf_sigma)
    psf_kernel.normalize() 
        
    # The convolve_fft function handles boundary conditions and normalization well
    convolved_image = convolve_fft(image_data, psf_kernel, normalize_kernel=True, 
                                   boundary='fill', fill_value=0.0)
    
    return convolved_image


def format_number(number):
    """Formats a number with '.' as a thousands separator."""
    num_str = str(number)
    reversed_num_str = num_str[::-1]
    parts = []
    for i in range(0, len(reversed_num_str), 3):
        parts.append(reversed_num_str[i:i+3])
        
    formatted_num = '.'.join(parts)[::-1]
    return formatted_num 


def print_training_summary(model, diffusion, n_plates, n_epochs, epochs_on_plate):
    """Helper to print a clean summary of learnable parameters before training starts."""
    print("\n" + "="*50)
    print("       DIIP TRAINING CONFIGURATION       ")
    print("="*50)
    
    # Model Params
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters (Learnable): {format_number(model_params)}")
    
    # Diffusion Params
    try:
        current_s = diffusion.get_current_s()
        s_val = current_s.item()
        if diffusion.learnable_s:
            print(f"Diffusion 's' Parameter:      Learnable (Start={s_val:.4f})")
        else:
            print(f"Diffusion 's' Parameter:      Fixed ({s_val:.4f})")
    except AttributeError:
        # Fallback if the diffusion object doesn't have the expected methods
        print("Diffusion 's' Parameter:      Status unknown (Missing get_current_s/learnable_s)")
        
    # Latent Params
    img_size = model.img_size
    print(f"Latent 'y' (per plate):       Optimization Enabled (Shape: {img_size}x{img_size})")
    
    print("-" * 50)
    print(f"Total Plates:   {n_plates}")
    print(f"Total Epochs:   {n_epochs}")
    print(f"Steps/Plate:    {epochs_on_plate}")
    print("="*50 + "\n")


def save_list(lst, filename='./mylist.json'):
    """Save a list to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)

def load_list(filename):
    """Load a list from a JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def shuffle_list(original_list):
    random.shuffle(original_list)
    return original_list


def get_float(start, end, decimal_places=4):
    """
    Generates random float between start and end
    """
    num = round(random.uniform(start, end), decimal_places)
    return num
    

class WarmupLinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(WarmupLinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Learning rate stays at initial value during warmup
            return [base_lr for base_lr in self.base_lrs]
        elif self.last_epoch >= self.total_epochs:
            # After total_epochs, LR is 0
            return [0.0 for _ in self.base_lrs]
        else:
            # Linear decay after warmup
            decay_epochs = self.total_epochs - self.warmup_epochs
            current_decay = self.last_epoch - self.warmup_epochs
            factor = 1 - (current_decay / decay_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]


def make_gif(sequence_path, output_gif, duration=300, loop=0):
    """
    Create an animated GIF from a sequence of PNG images named 
    0.png, 1.png, 2.png, ... in sequence_path.
s
    Parameters:
        seqience_path (str):    -- Path to the folder containing PNG images.
        output_gif (str):       -- output GIF name.
        duration (int):         -- Duration of each frame in milliseconds (default 300).
        loop (int):             -- Number of loops for the GIF (0 means infinite loop).
    """
    # Collect PNG files named as numbers with .png extension
    files = []
    for filename in os.listdir(sequence_path):
        if filename.endswith('.png'):
            try:
                # Extract numeric part from filename to sort properly
                num = int(os.path.splitext(filename)[0])
                files.append((num, filename))
            except ValueError:
                # Skip files that don't follow the numeric naming scheme
                pass

    # Sort files by numeric order
    files.sort(key=lambda x: x[0])

    # Load images
    frames = []
    for _, filename in files:
        img_path = os.path.join(sequence_path, filename)
        frame = Image.open(img_path)
        frames.append(frame)

    if not frames:
        raise ValueError("No valid PNG images found in the folder.")

    # Save frames as an animated GIF
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )
    print(f"GIF saved to {output_gif}")


def get_mask(size, ms, dx=0, dy=0):
    """Make binary mask of size ms shifted by 
    dx, dy from center
    """
    x = size // 2 + dx
    y = size // 2 + dy 
    m = np.ones((size, size))
    m[y - ms//2:y + ms//2, x - ms//2: x + ms//2] = 0
    return m


def get_crop(fits_path, ra, dec, cs, dx=0, dy=0, median=False, s=None, radec=False):
    """Make a fits file crop 

    Parameters:
        fits (str)          -- fits path
        ra_hms (float)      -- crop center ra coordinate
        dec_hms (float)     -- crop center dec coordinat
        cs (int)            -- crop size
        dx (int)            -- crop x-center dx (in pixels)
        dy (int)            -- crop y-center dy (in pixels)
        bkg (bool)          -- whether to subtract median
        s (float)           -- Arcsinh stretch parameter (no stretch if None)
    """
    
    wcs = get_wcs(fits_path)
    if radec:
        pass
    else:
        ra, dec = hms2radec(ra, dec)

    x, y = world2pix(ra, dec, wcs)
    x += dx
    y += dy

    with fits.open(fits_path) as hdul:
        crop = np.array(hdul[0].data)[y - cs//2:y + cs//2, x - cs//2:x + cs//2] 

    if median:
        m = np.median(crop)
        crop -= m
        print(f'Median subtracted: {m}')

    if s:
        crop = np.arcsinh(crop / s)

    return crop

    
def fixnan(arr, fill='median'):
    if fill=='median':
        median = np.nanmedian(arr)
        arr[np.isnan(arr)] = median
    else:
        arr[np.isnan(arr)] = fill
    return arr
    
    
def make_text(config):
    "Configuration description"
    
    text = ''
    text += 'method: ' + str(config.upsample_mode) + '\n'     
    text += 'skip: ' + str(config.skip) + '\n'
    text += 'skip_mode: ' + str(config.skip_mode) + '\n'
    text += 'layers: ' + str(config.n_layers) + '\n'
    text += 'filters: ' + str(config.fmaps) + '\n'
    text += 'kernel: ' + str(config.kernel) + '\n'
    text += 'dropout: ' + str(config.dropout) + '\n'
    text += 'noise: ' + str(config.noise) + '\n'
    text += 'iters: ' + str(config.n_epochs - 1) + '\n'
    text += 'lr: ' + str(config.lr) + '\n'
    
    return text
    

class Sobolev_loss(torch.nn.Module):
    def __init__(self, s, N=256):
        super(Sobolev_loss, self).__init__()
        S = np.ones((N,N))
        
        for row in range(N):
            for col in range(N):
                
                S[row][col] += (row**2 + col**2)
    
        self.S = torch.from_numpy(S**(s/2)).to('cuda')
    
    def forward(self, predicted, target):
        fft = torch.fft.fft2(predicted - target)
        loss = torch.norm(self.S * torch.sqrt(fft.real**2 + fft.imag**2))
        return loss
        

def get_h(path):
    return fits.open(path)[0].header


def save_img(data, path, ws=5, contrast=.25, 
            origin='upper', mx=None, my=None, mw=None,
            config=None, hline=None, vline=None, 
            vmin=None, vmax=None, cmap='hot', color='white'):

    size = data.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(size / 100, size / 100), dpi=100)

    if not vmin:
        zscale = ZScaleInterval(contrast=contrast)
        z1, z2 = zscale.get_limits(data)
    else:
        z1 = vmin
        z2 = vmax

    # yellow box mask
    if mx:
        rect = patches.Rectangle((mx-mw//2, my-mw//2), 
                                 mw, mw, linewidth=1, 
                                 edgecolor=color, facecolor='none', 
                                 ls='--', fill=None)

        # Add rectangle to the axes
        ax.add_patch(rect)
        
    if config:
        ax.text(.01, .55, make_text(config), fontsize=10, color='w', transform=ax.transAxes)
    if hline:
        ax.axhline(y=hline, xmin=0.0, xmax=1.0, color='r', linewidth=.75)  # Horizontal line at y=0.5 across full width
    if vline:
        ax.axvline(x=vline, ymin=0.0, ymax=1.0, color='r', linewidth=.75)  # Vertical line at x=2 across full height

    ax.imshow(data, cmap=cmap, vmin=z1, vmax=z2, origin=origin)
    ax.axis('off')
    
    fig.savefig(path, bbox_inches='tight', dpi=129.95, pad_inches=0)
    plt.close(fig)

def save_fits(data, path, overwrite=True, header=None):
    if header is not None:
        hdu = fits.PrimaryHDU(data=data, header=header)
    else:
        hdu = fits.PrimaryHDU(data=data)
        
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=overwrite)
    hdulist.close()


def get_t(data, augmentation=True, device='cpu', resize=None):
    
    if resize:
        transform = transforms.Resize((resize, resize))
        data0 = transform(np2t(data.copy()).to(device))
    else:
        data0 = np2t(data).to(device)
        
    if augmentation:
        data90 = torch.rot90(data0, k=1, dims=(0, 1))
        data180 = torch.rot90(data0, k=2, dims=(0, 1))
        data270 = torch.rot90(data0, k=3, dims=(0, 1))
        datat = torch.cat((data0, data90, data180, data270), 0)
        return datat
    else:
        return data0


def rescale(data, size):
    """
    Rescale a tensor or NumPy array to the specified spatial size using adaptive average pooling.
    
    Supports:
      - PyTorch tensors: 2D [H, W], 3D [C, H, W], or 4D [B, C, H, W]
      - NumPy arrays:    2D [H, W], 3D [H, W, C], or 4D [B, C, H, W]
    
    Note: NumPy 3D arrays are assumed to be in HWC format, while PyTorch 3D tensors are CHW.
    
    Args:
        data: Input tensor or array.
        size: Target output size (int or (int, int)).
    
    Returns:
        Rescaled data in the same type and format as input.
    """
    ad = torch.nn.AdaptiveAvgPool2d(size)
    
    if isinstance(data, torch.Tensor):
        original_shape = data.shape
        if data.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            output = ad(data.unsqueeze(0).unsqueeze(0))
            return output.squeeze(0).squeeze(0)
        elif data.dim() == 3:
            # [C, H, W] -> [1, C, H, W]
            output = ad(data.unsqueeze(0))
            return output.squeeze(0)
        elif data.dim() == 4:
            return ad(data)
        else:
            raise ValueError(f"Unsupported tensor dimension: {data.dim()}. Expected 2D, 3D, or 4D.")
    
    elif isinstance(data, np.ndarray):
        original_shape = data.shape
        original_dtype = data.dtype
        
        if data.ndim == 2:
            # [H, W] -> [1, 1, H, W]
            tensor = torch.from_numpy(np.ascontiguousarray(data[None, None, :, :]))
            output_tensor = ad(tensor)
            output_array = output_tensor.squeeze(0).squeeze(0).numpy()
            
        elif data.ndim == 3:
            # Assume HWC format: [H, W, C] -> [1, C, H, W]
            # Transpose to CHW, add batch dim
            tensor = torch.from_numpy(np.ascontiguousarray(data.transpose(2, 0, 1)[None, :, :, :]))
            output_tensor = ad(tensor)
            # Remove batch dim and transpose back to HWC
            output_array = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
            
        elif data.ndim == 4:
            # Assume BCHW format
            tensor = torch.from_numpy(np.ascontiguousarray(data))
            output_tensor = ad(tensor)
            output_array = output_tensor.numpy()
            
        else:
            raise ValueError(f"Unsupported array dimension: {data.ndim}. Expected 2D, 3D, or 4D.")
        
        # Preserve original dtype (with safe casting)
        if output_array.dtype != original_dtype:
            output_array = output_array.astype(original_dtype, casting='same_kind')
        
        return output_array
    
    else:
        raise TypeError(f"Input must be torch.Tensor or np.ndarray, got {type(data)}")
        
        
def np2t(array):
    """
    Convert a NumPy image array to a PyTorch tensor with shape (B, C, H, W).
    
    Supports:
      - (H, W)         → grayscale, becomes (1, 1, H, W)
      - (H, W, C)      → channels last (e.g. RGB), becomes (1, C, H, W)
      - (C, H, W)      → channels first, becomes (1, C, H, W)
    
    Args:
        array (np.ndarray): Input image array.
        
    Returns:
        torch.Tensor: Tensor of shape (1, C, H, W).
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    
    # Handle byte order issues
    if array.dtype.byteorder not in ('=', '|'):
        array = array.byteswap().newbyteorder()
    
    ndim = array.ndim
    if ndim == 2:
        # Grayscale: (H, W) → (1, 1, H, W)
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    elif ndim == 3:
        H, W = array.shape[-2], array.shape[-1]
        C = array.shape[0] if array.shape[0] not in (H, W) else array.shape[-1]
        
        # Heuristic: if first dim is small (≤4), assume channels-first
        if array.shape[0] <= 4 and array.shape[0] not in (H, W):
            # (C, H, W) → already correct order
            tensor = torch.from_numpy(array).unsqueeze(0)
        else:
            # Assume (H, W, C) → move channel to front
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}. Expected 2D or 3D.")
    
    return tensor


def t2np(tensor):
    """
    Convert a PyTorch tensor to a NumPy array with shape (C, H, W).
    
    Input tensor must be in (B, C, H, W) or (C, H, W) format.
    Output is always (C, H, W).
    
    Args:
        tensor (torch.Tensor): Input tensor.
        
    Returns:
        np.ndarray: Array of shape (C, H, W).
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Move to CPU and detach
    array = tensor.detach().cpu().numpy()
    
    if array.ndim == 4:
        # (B, C, H, W) → take first batch
        array = array[0]
    elif array.ndim == 3:
        # (C, H, W) → already good
        pass
    elif array.ndim == 2:
        # (H, W) → add channel dim
        array = array[np.newaxis, :, :]
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}. Expected 2D, 3D or 4D.")
    
    return array


def get_center(fits):
    d = get_d(fits)
    h = get_h(fits)
    w = get_wcs(fits)
    pix = d.shape[0] // 2
    ra, dec = pix2world(pix, pix, w)
    return ra, dec

    
def pix2world(x, y, wcs):
    ra, dec = wcs.wcs_pix2world(x, y, 0)  # 0 is for the first frame (default)
    return ra.item(), dec.item()

    
def world2pix(ra, dec, wcs):
    sky = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    pixel_coords = wcs.world_to_pixel(sky)
    return int(pixel_coords[0]), int(pixel_coords[1]) 


def hms2radec(ra_hms, dec_dms):
    
    coords = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg))
    return float(coords.ra.degree), float(coords.dec.degree)


def get_d(file, fixnans=None, median=False, arcsinh=False, f=1, scale=True):
    """
    Get data from given FITS or JPG file.
    Parameters:
        file           -- File path (supports .fits and .jpg/.jpeg)
        fixnans        -- Value to replace NaNs with (e.g., 0). If None, leave as-is.
                          (Only relevant for FITS; JPGs are integer and typically have no NaNs.)
        median         -- Whether to subtract the median
        arcsinh        -- Whether to apply arcsinh transform
        f              -- arcsinh denominator scale factor
        scale          -- If True and arcsinh is used, scale result to [0, 1] range.
                         (Note: if arcsinh=False, this has no effect in current implementation.)
    Returns:
        Processed numpy array (dtype float64)
    """
    _, ext = os.path.splitext(file)
    ext = ext.lower()

    if ext in ('.fits', '.fit'):
        with fits.open(file) as hdul:
            d = hdul[0].data
        d = d.astype(np.float64)

    elif ext in ('.jpg', '.jpeg'):
        # Load as RGB (3 channels)
        img = Image.open(file).convert('RGB')  # Ensures 3 channels
        d = np.array(img, dtype=np.float64)

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Handle NaNs (typically only in FITS)
    if np.isnan(d).any():
        print("NaN values are present in the array")
        if fixnans is not None:
            d = np.nan_to_num(d, nan=fixnans)
    elif fixnans is not None:
        # JPGs usually don't have NaNs, but apply fix if explicitly requested
        d = np.nan_to_num(d, nan=fixnans)

    if median:
        d = d - np.median(d)

    if arcsinh:
        d = np.arcsinh(d / f)

    if arcsinh and scale:
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d = (d - d_min) / (d_max - d_min)
        else:
            d = np.zeros_like(d)

    return d


def get_wcs(file):
    with fits.open(file) as hdul:
        h = hdul[0].header
    return WCS(h)
    
    
def image_d(data, ws=5, contrast=.25, 
            origin='upper', mx=None, my=None, mw=None,
            config=None, hline=None, vline=None, cmap='plasma', vmin=None, 
            vmax=None, color='white'): 

    if (vmin is not None) and (vmax is not None):
        z1 = vmin
        z2 = vmax
    else:
        zscale = ZScaleInterval(contrast=contrast)
        z1, z2 = zscale.get_limits(data)
        
    fig, ax = plt.subplots(1, 1, figsize=(ws, ws))

    if isinstance(data, torch.Tensor):
        data = t2np(data)

    # yellow box mask
    if mx:
        rect = patches.Rectangle((mx - mw // 2, my - mw // 2), 
                                 mw, mw, linewidth=1, 
                                 edgecolor=color, facecolor='none', 
                                 ls='--', fill=None)

        # Add rectangle to the axes
        ax.add_patch(rect)
        
    if config:
        ax.text(.01, .55, make_text(config), fontsize=10, color='w', transform=ax.transAxes)
    if hline:
        ax.axhline(y=hline, xmin=0.0, xmax=1.0, color='r', linewidth=.75)  # Horizontal line at y=0.5 across full width
    if vline:
        ax.axvline(x=vline, ymin=0.0, ymax=1.0, color='r', linewidth=.75)  # Vertical line at x=2 across full height
    try:
        ax.imshow(data, cmap=cmap, vmin=z1, vmax=z2, origin=origin)
    except TypeError:
        ax.imshow(np.transpose(data, (1, 2, 0)), cmap=cmap, vmin=z1, vmax=z2, origin=origin)


def hist(data1, data2=None, bins=100, ws=5):
    
    fig, ax = plt.subplots(1, 1, figsize=(ws, ws))
    a1, b1, c1 = ax.hist(data1.ravel(), bins=bins)
    print(f'Mean: {np.mean(data1)} ', f'STD: {np.std(data1)}')
    
    # second distribution
    if data2 is not None:
        a2, b2, c2 = ax.hist(data2.ravel(), bins=bins)
        print(f'Mean: {np.mean(data2)} ', f'STD: {np.std(data2)}')
        

def count_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(format_number(params) + ' parameters')