import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import argparse
import importlib
import omegaconf.dictconfig
from photutils.datasets import make_model_image, make_model_params
from astropy.table import QTable
from photutils.psf import GaussianPSF
import torch
import matplotlib.patches as patches
import matplotlib.animation as animation


def video2mp4(video, path, fps=24, dpi=400):
    """
    Create an MP4 video from numpy video arrays using matplotlib.
    Each array is displayed as an image in the video with no padding.
    """
    size = int(video[0].shape[0])
    # Create figure with exact pixel size: 1024x1024 at given DPI
    figsize = (size / dpi, size / dpi)  # width, height in inches
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # [left, bottom, width, height] in figure coords
    ax.axis('off')

    # Display first frame
    im = ax.imshow(video[0], cmap='hot' if len(video[0].shape) == 2 else None, aspect='auto')

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=video,
        blit=True, repeat=False
    )

    # Save without savefig_kwargs that interfere
    ani.save(
        path,
        writer='ffmpeg',
        fps=fps,
        dpi=dpi  # This will result in 1024x1024 if figsize is set correctly
    )

    plt.close(fig)
    print(f"Video saved to {path}")
    

def fixnan(arr):
    """
    Replace NaN values with zeros in a NumPy array.
    """
    return np.nan_to_num(arr, nan=0.0)
    
def save_fits(array, filename, header=None, overwrite=False):
    """
    Save a NumPy array as a FITS file.
    
    Parameters:
    -----------
    array : numpy.ndarray
        The array to save as FITS
    filename : str
        Output filename (should end with .fits or .fit)
    header : astropy.io.fits.Header, optional
        FITS header to include with the data
    overwrite : bool, optional
        Whether to overwrite existing file (default: False)
    
    Returns:
    --------
    None
    
    Examples:
    ---------
    >>> import numpy as np
    >>> data = np.random.random((100, 100))
    >>> save_array_as_fits(data, 'output.fits')
    
    >>> # With custom header
    >>> from astropy.io import fits
    >>> hdr = fits.Header()
    >>> hdr['OBJECT'] = 'M31'
    >>> hdr['EXPTIME'] = 300
    >>> save_array_as_fits(data, 'output.fits', header=hdr)
    """
    # Convert to float64 if it's not a supported FITS data type
    # FITS supports: uint8, int16, int32, float32, float64
    supported_dtypes = [np.uint8, np.int16, np.int32, np.float32, np.float64]
    
    if array.dtype not in supported_dtypes:
        # Choose appropriate conversion based on current dtype
        if np.issubdtype(array.dtype, np.integer):
            if array.min() >= 0 and array.max() <= 255:
                array = array.astype(np.uint8)
            elif array.min() >= -32768 and array.max() <= 32767:
                array = array.astype(np.int16)
            else:
                array = array.astype(np.int32)
        else:
            # For floating point, use float32 or float64
            if array.dtype == np.float16:
                array = array.astype(np.float32)
            elif array.dtype not in [np.float32, np.float64]:
                array = array.astype(np.float64)
    
    # Create Primary HDU
    if header is None:
        hdu = fits.PrimaryHDU(array)
    else:
        hdu = fits.PrimaryHDU(array, header=header)
    
    # Write to file
    hdu.writeto(filename, overwrite=overwrite)

    
def get_tiles(image, tile_size):
    """
    Splits a square image of size N x N (where N is a power of 2) 
    into non-overlapping square tiles of size K x K (where K is a power of 2 and K <= N).
    
    Parameters:
    -----------
    image : torch.Tensor or np.ndarray
        Input image tensor/array of shape:
        - (N, N) for grayscale
        - (C, N, N) for multi-channel (e.g., RGB)
        - (B, C, N, N) for batched multi-channel (torch only)
    tile_size : int
        Size of each tile (K). Must be a power of 2 and divide N evenly.
        
    Returns:
    --------
    tiles : torch.Tensor or np.ndarray
        Stacked tiles with shape:
        - (num_tiles, K, K) for grayscale
        - (num_tiles, C, K, K) for multi-channel
        - (B, num_tiles, C, K, K) for batched input (torch only)
        
    num_tiles : int
        Total number of tiles = (N // K) ** 2
    """
    
    # Handle input type
    is_torch = isinstance(image, torch.Tensor)
    if not is_torch:
        image = np.asarray(image)
    
    # Determine input shape and validate
    orig_shape = image.shape
    ndim = image.ndim
    
    if ndim == 2:
        # Grayscale: (H, W)
        H, W = image.shape
        C = None
        batched = False
    elif ndim == 3:
        # Multi-channel: (C, H, W) for torch or (H, W, C) for numpy
        if is_torch:
            C, H, W = image.shape
            channel_last = False
        else:
            H, W, C = image.shape
            channel_last = True
        batched = False
    elif ndim == 4 and is_torch:
        # Batched multi-channel: (B, C, H, W)
        B, C, H, W = image.shape
        channel_last = False
        batched = True
    else:
        raise ValueError(f"Unsupported input shape: {orig_shape}. "
                         "Supported: (H,W), (C,H,W), (H,W,C), or (B,C,H,W) for torch.")
    
    # Validate square image
    if H != W:
        raise ValueError(f"Input image must be square, got {H}x{W}")
    
    N = H
    K = tile_size
    
    # Validate power of 2
    if not (N > 0 and (N & (N - 1)) == 0):
        raise ValueError(f"Image size N={N} must be a power of 2")
    if not (K > 0 and (K & (K - 1)) == 0):
        raise ValueError(f"Tile size K={K} must be a power of 2")
    
    # Validate divisibility
    if N % K != 0:
        raise ValueError(f"Tile size {K} must evenly divide image size {N}")
    
    # Calculate number of tiles per dimension
    tiles_per_side = N // K
    num_tiles = tiles_per_side ** 2
    
    # Handle channel ordering for numpy
    if not is_torch and ndim == 3:
        # Convert (H, W, C) -> (C, H, W) for processing
        image = np.transpose(image, (2, 0, 1))
        proc_shape = (C, H, W)
    else:
        proc_shape = orig_shape if not batched else (B * C, H, W)
    
    # Reshape based on batching
    if batched:
        # (B, C, N, N) -> (B, C, tiles_per_side, K, tiles_per_side, K)
        image_reshaped = image.view(B, C, tiles_per_side, K, tiles_per_side, K)
        # Permute to (B, tiles_per_side, tiles_per_side, C, K, K)
        image_permuted = image_reshaped.permute(0, 2, 4, 1, 3, 5)
        # Reshape to (B, num_tiles, C, K, K)
        tiles = image_permuted.reshape(num_tiles, C, K, K)
    else:
        if ndim == 2:
            # (N, N) -> (tiles_per_side, K, tiles_per_side, K)
            if is_torch:
                image_reshaped = image.view(tiles_per_side, K, tiles_per_side, K)
                image_permuted = image_reshaped.permute(0, 2, 1, 3)
                tiles = image_permuted.reshape(num_tiles, K, K)
            else:
                image_reshaped = image.reshape(tiles_per_side, K, tiles_per_side, K)
                image_permuted = np.transpose(image_reshaped, (0, 2, 1, 3))
                tiles = image_permuted.reshape(num_tiles, K, K)
        else:  # ndim == 3, processed as (C, N, N)
            if is_torch:
                image_reshaped = image.view(C, tiles_per_side, K, tiles_per_side, K)
                image_permuted = image_reshaped.permute(1, 3, 0, 2, 4)
                tiles = image_permuted.reshape(num_tiles, C, K, K)
            else:
                image_reshaped = image.reshape(C, tiles_per_side, K, tiles_per_side, K)
                image_permuted = np.transpose(image_reshaped, (1, 3, 0, 2, 4))
                tiles = image_permuted.reshape(num_tiles, C, K, K)
    
    # Convert back to original channel ordering for numpy
    if not is_torch and ndim == 3:
        # (num_tiles, C, K, K) -> (num_tiles, K, K, C)
        tiles = np.transpose(tiles, (0, 2, 3, 1))
    elif not is_torch and ndim == 2:
        # Already correct: (num_tiles, K, K)
        pass
    
    return tiles


def merge_tiles(tiles, original_shape, tile_size):
    """
    Reconstructs a square image from non-overlapping square tiles.
    
    Parameters:
    -----------
    tiles : torch.Tensor or np.ndarray
        Stacked tiles with shape:
        - (num_tiles, K, K) for grayscale
        - (num_tiles, C, K, K) for multi-channel (torch)
        - (num_tiles, K, K, C) for multi-channel (numpy)
        - (B, num_tiles, C, K, K) for batched input (torch only)
    original_shape : tuple
        Original shape of the image before tiling.
        Supported shapes:
        - (N, N) for grayscale
        - (C, N, N) for multi-channel torch
        - (N, N, C) for multi-channel numpy
        - (B, C, N, N) for batched torch
    tile_size : int
        Size of each tile (K). Must be a power of 2 and divide N evenly.
        
    Returns:
    --------
    image : torch.Tensor or np.ndarray
        Reconstructed image with shape matching original_shape.
    """
    import torch
    import numpy as np
    
    # Handle input type
    is_torch = isinstance(tiles, torch.Tensor)
    if not is_torch:
        tiles = np.asarray(tiles)
    
    orig_shape = original_shape
    tiles_shape = tiles.shape
    
    # Determine original image properties
    ndim = len(orig_shape)
    
    if ndim == 2:
        # Grayscale: (H, W)
        H, W = orig_shape
        C = None
        batched = False
        channel_last = False
    elif ndim == 3:
        # Multi-channel: (C, H, W) for torch or (H, W, C) for numpy
        if is_torch:
            C, H, W = orig_shape
            channel_last = False
        else:
            H, W, C = orig_shape
            channel_last = True
        batched = False
    elif ndim == 4 and is_torch:
        # Batched multi-channel: (B, C, H, W)
        B, C, H, W = orig_shape
        channel_last = False
        batched = True
    else:
        raise ValueError(f"Unsupported original shape: {orig_shape}. "
                         "Supported: (H,W), (C,H,W), (H,W,C), or (B,C,H,W) for torch.")
    
    # Validate square image
    if H != W:
        raise ValueError(f"Original image must be square, got {H}x{W}")
    
    N = H
    K = tile_size
    
    # Validate power of 2
    if not (N > 0 and (N & (N - 1)) == 0):
        raise ValueError(f"Image size N={N} must be a power of 2")
    if not (K > 0 and (K & (K - 1)) == 0):
        raise ValueError(f"Tile size K={K} must be a power of 2")
    
    # Validate divisibility
    if N % K != 0:
        raise ValueError(f"Tile size {K} must evenly divide image size {N}")
    
    tiles_per_side = N // K
    num_tiles = tiles_per_side ** 2
    
    # Validate tiles shape
    if batched:
        expected_shape = (orig_shape[0], num_tiles, orig_shape[1], K, K)
        if tiles_shape != expected_shape:
            raise ValueError(f"Expected tiles shape {expected_shape}, got {tiles_shape}")
    else:
        if ndim == 2:
            expected_shape = (num_tiles, K, K)
        else:  # ndim == 3
            if is_torch:
                expected_shape = (num_tiles, C, K, K)
            else:
                expected_shape = (num_tiles, K, K, C)
        if tiles_shape != expected_shape:
            raise ValueError(f"Expected tiles shape {expected_shape}, got {tiles_shape}")
    
    # Handle channel ordering for numpy
    if not is_torch and ndim == 3:
        # Convert (num_tiles, K, K, C) -> (num_tiles, C, K, K) for processing
        tiles_proc = np.transpose(tiles, (0, 3, 1, 2))
    else:
        tiles_proc = tiles
    
    # Reconstruct based on batching and dimensions
    if batched:
        # (B, num_tiles, C, K, K) -> (B, tiles_per_side, tiles_per_side, C, K, K)
        tiles_reshaped = tiles_proc.reshape(B, tiles_per_side, tiles_per_side, C, K, K)
        # Permute to (B, C, tiles_per_side, K, tiles_per_side, K)
        tiles_permuted = tiles_reshaped.permute(0, 3, 1, 4, 2, 5)
        # Reshape to (B, C, N, N)
        image = tiles_permuted.reshape(B, C, N, N)
    else:
        if ndim == 2:
            # (num_tiles, K, K) -> (tiles_per_side, tiles_per_side, K, K)
            if is_torch:
                tiles_reshaped = tiles_proc.reshape(tiles_per_side, tiles_per_side, K, K)
                tiles_permuted = tiles_reshaped.permute(0, 2, 1, 3)
                image = tiles_permuted.reshape(N, N)
            else:
                tiles_reshaped = tiles_proc.reshape(tiles_per_side, tiles_per_side, K, K)
                tiles_permuted = np.transpose(tiles_reshaped, (0, 2, 1, 3))
                image = tiles_permuted.reshape(N, N)
        else:  # ndim == 3, processed as (num_tiles, C, K, K)
            if is_torch:
                tiles_reshaped = tiles_proc.reshape(tiles_per_side, tiles_per_side, C, K, K)
                tiles_permuted = tiles_reshaped.permute(2, 0, 3, 1, 4)
                image = tiles_permuted.reshape(C, N, N)
            else:
                tiles_reshaped = tiles_proc.reshape(tiles_per_side, tiles_per_side, C, K, K)
                tiles_permuted = np.transpose(tiles_reshaped, (2, 0, 3, 1, 4))
                image = tiles_permuted.reshape(C, N, N)
    
    # Convert back to original channel ordering for numpy
    if not is_torch and ndim == 3:
        # (C, N, N) -> (N, N, C)
        image = np.transpose(image, (1, 2, 0))
    
    return image
        
    
def add_transient(image, x, y, flux, fwhm, seed=0):
    """Add transient with defined flux and x, y coords to a sci_d image
        Parameters:
            fwhm         -- source fwhm
            x, y,        -- source coords
            flux         -- source flux
    """
    shape = image.shape

    # Создаем параметры модели для одного источника
    params = make_model_params(shape, n_sources=1, flux=(flux, flux), 
                              x_fwhm=(fwhm, fwhm), y_fwhm=(fwhm, fwhm),
                              theta=(0, 0), seed=seed)

    params['x_0'][0] = x
    params['y_0'][0] = y

    # Создаем PSF модели
    psf_model = GaussianPSF()
    # Размер модели (должен быть достаточно большим, чтобы захватить источник)
    model_shape = (int(6*fwhm), int(6*fwhm))
    # Генерируем изображение источника
    transient_image = make_model_image(shape, psf_model, params, model_shape=model_shape)

    # Добавляем транзиент к исходному изображению
    new_image = image + transient_image

    return new_image

def format_number(number):
    num_str = str(number)
    reversed_num_str = num_str[::-1]
    parts = []
    for i in range(0, len(reversed_num_str), 3):
        parts.append(reversed_num_str[i:i+3])
        
    formatted_num = '.'.join(parts)[::-1]
    return formatted_num
    
def count_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(format_number(params) + ' parameters')

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def namespace2dict(config):
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    return conf_dict


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_runner(runner_name, config):
    runner = Registers.runners[runner_name](config)
    return runner

def change_order(data):
    return data.byteswap().view(data.dtype.newbyteorder())

    
def circ_mask(height, width, center_x, center_y, radius, channels=1):
    
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    if channels > 1:
        array = np.tile(mask[..., np.newaxis], (1, 1, channels))
        return 1 - array
    else:
        return 1 - mask


def rescale(data, size):
    """
    Rescale tensor or array to (size, size) using adaptive average pooling.
    
    Supports:
      - torch.Tensor: 2D [H,W], 3D [C,H,W], 4D [B,C,H,W] (any dtype)
      - np.ndarray:   2D [H,W], 3D [H,W,C], 4D [B,C,H,W] (any dtype)
    
    size: int (e.g., 512) → output spatial size = (512, 512)
    """
    pool = torch.nn.AdaptiveAvgPool2d(size)

    if isinstance(data, torch.Tensor):
        # Always work on a float32 copy for pooling
        was_float = torch.is_floating_point(data)
        orig_dtype = data.dtype

        # Handle dimensions
        if data.dim() == 2:
            x = data.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif data.dim() == 3:
            x = data.unsqueeze(0)               # [1,C,H,W]
        elif data.dim() == 4:
            x = data                            # [B,C,H,W]
        else:
            raise ValueError(f"Unsupported tensor dim: {data.dim()}")

        # Convert to float32 for pooling (critical!)
        x_float = x.to(torch.float32)

        # Apply pooling
        out_float = pool(x_float)

        # Remove added dimensions
        if data.dim() == 2:
            out_float = out_float.squeeze(0).squeeze(0)
        elif data.dim() == 3:
            out_float = out_float.squeeze(0)

        # Convert back to original dtype if needed
        if was_float:
            return out_float.to(orig_dtype)
        else:
            # For integer types: clamp, round, cast
            out_float = out_float.clamp(min=0)
            if orig_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                out_float = out_float.round()
            return out_float.to(orig_dtype)

    elif isinstance(data, np.ndarray):
        orig_dtype = data.dtype
        is_integer = np.issubdtype(orig_dtype, np.integer)

        if data.ndim == 2:
            # [H, W] → [1,1,H,W]
            arr = data[None, None, :, :]
        elif data.ndim == 3:
            # [H, W, C] → [1, C, H, W]
            arr = data.transpose(2, 0, 1)[None, :, :, :]
        elif data.ndim == 4:
            # Assume [B, C, H, W]
            arr = data
        else:
            raise ValueError(f"Unsupported array ndim: {data.ndim}")

        # Ensure contiguous and convert to float32 tensor
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)

        # Pool
        out_tensor = pool(tensor)

        # Convert back to array
        if data.ndim == 2:
            out_array = out_tensor.squeeze(0).squeeze(0).numpy()
        elif data.ndim == 3:
            out_array = out_tensor.squeeze(0).permute(1, 2, 0).numpy()  # CHW → HWC
        else:  # ndim == 4
            out_array = out_tensor.numpy()

        # Restore original dtype
        if is_integer:
            out_array = np.clip(out_array, 0, None)
            out_array = np.round(out_array).astype(orig_dtype)
        else:
            # Preserve float or bool
            if orig_dtype == np.bool_:
                out_array = (out_array > 0.5).astype(orig_dtype)
            else:
                out_array = out_array.astype(orig_dtype, casting='same_kind')

        return out_array

    else:
        raise TypeError(f"Input must be torch.Tensor or np.ndarray, got {type(data)}")
        
        
        
def torch2np(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    if len(tensor.shape) == 4:
        return np.swapaxes(np.swapaxes(tensor[0].detach().cpu().numpy(), 0, 2), 0, 1)
    elif len(tensor.shape) == 3:
        return np.swapaxes(np.swapaxes(tensor.detach().cpu().numpy(), 0, 2), 0, 1)
    else:
        return tensor.detach().cpu().numpy()


def np2torch(array):
    
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    tensor = torch.from_numpy(array)
    
    if len(tensor.shape) == 2:
        return torch.unsqueeze(torch.unsqueeze(tensor, 0), 0)

    elif len(tensor.shape) == 3:
        return torch.unsqueeze(torch.swapaxes(torch.swapaxes(tensor, 0, 2), 1, 2), 0)


def get_h(path):
    return fits.open(path)[0].header


def get_d(path):
    return fits.open(path)[0].data

def save_img(data, path, ws=5, contrast=.25, 
            origin='upper', mx=None, my=None, mw=None,
            config=None, hline=None, vline=None, 
            vmin=None, vmax=None, cmap='hot'):

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
                                 edgecolor='yellow', facecolor='none', 
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


def image_d(data, ws=5, contrast=.25, 
            origin='upper', mx=None, my=None, mw=None,
            config=None, hline=None, vline=None, cmap='gray'): 
    
    zscale = ZScaleInterval(contrast=contrast)
    z1, z2 = zscale.get_limits(data)
    fig, ax = plt.subplots(1, 1, figsize=(ws, ws))

    # yellow box mask
    if mx:
        rect = patches.Rectangle((mx - mw // 2, my - mw // 2), 
                                 mw, mw, linewidth=1, 
                                 edgecolor='yellow', facecolor='none', 
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


def hist(data, bins=100, ws=5):
    fig, ax = plt.subplots(1, 1, figsize=(ws, ws))
    a,b,c = ax.hist(data.ravel(), bins=bins)
    print(f'Mean: {np.mean(data)} ', f'STD: {np.std(data)}')


def format_number(number):
    num_str = str(number)
    reversed_num_str = num_str[::-1]
    parts = []
    for i in range(0, len(reversed_num_str), 3):
        parts.append(reversed_num_str[i:i+3])
        
    formatted_num = '.'.join(parts)[::-1]
    return formatted_num


def count_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(format_number(params) + ' parameters')


    from diffusers import DDPMPipeline


class CustomDDPMPipeline(DDPMPipeline):
    def __call__(self, num_inference_steps: int = 100, latents: torch.Tensor = None):
        if latents is None:
            # Default behavior: generate random noise if no latents are provided
            batch_size = 1
            channels = self.unet.config.in_channels
            height, width = self.unet.config.sample_size, self.unet.config.sample_size
            latents = torch.randn(batch_size, channels, height, width).to(self.device)

        # Use custom latents for sampling as shown in Solution 2 above
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.unet(latents.float(), t)["sample"]
            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        return latents.cpu().numpy()
