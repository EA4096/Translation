import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import os
import numpy as np


# --- Interactive Plotting Setup ---
try:
    from IPython.display import display, clear_output
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# --- DiffusionUNet Components ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        # Ensure GroupNorm initialization is robust. 
        norm_groups = min(8, in_ch) if in_ch > 1 else 1
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(norm_groups, in_ch), 
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        
        norm_groups_out = min(8, out_ch) if out_ch > 1 else 1
        self.block2 = nn.Sequential(
            nn.GroupNorm(norm_groups_out, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(..., ) + (None, ) * 2] 
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        # MultiheadAttention expects (Batch, Sequence, Channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        
    def forward(self, x):
        b, c, h, w = x.shape
        # Permute to (Batch, Sequence, Channels)
        x_flat = x.view(b, c, -1).transpose(1, 2)
        x_norm = self.ln(x_flat)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        # Residual connection
        out = x_flat + attn_out
        # Permute back to (Batch, Channels, Height, Width)
        return out.transpose(1, 2).view(b, c, h, w)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 4, 2, 1)
    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, 2, 1)
    def forward(self, x):
        return self.op(x)

class DiffusionUNet(nn.Module):
    def __init__(self, img_size=64, img_channels=3, base_channels=64, channel_mults=(1, 2, 4, 8), 
                 attention_resolutions=(32, 16, 8), time_emb_dim=256):
        super().__init__()
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.img_size = img_size
        self.attention_resolutions = set(attention_resolutions) 
        
        # --- 1. Resolution Validation Check ---
        possible_resolutions = {img_size}
        temp_res = img_size
        for _ in range(len(channel_mults)):
            temp_res //= 2
            possible_resolutions.add(temp_res)
        
        unachievable_res = self.attention_resolutions - possible_resolutions
        if unachievable_res:
            raise ValueError(
                f"Requested attention resolutions {unachievable_res} are unachievable "
                f"with an input image size of {img_size}. Possible resolutions are {sorted(list(possible_resolutions))}."
            )
        # ------------------------------------

        # --- Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # --- Initial Convolution ---
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        
        # --- Downsampling Path Initialization ---
        self.downs = nn.ModuleList()
        in_ch = base_channels
        
        # List to store the number of channels for each skip connection generated
        self.skip_channels = []
        current_res = img_size 

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            # Block 1
            self.downs.append(Block(in_ch, out_ch, time_emb_dim))
            self.skip_channels.append(out_ch) # Skip from Block 1
            
            # Attention
            in_ch = out_ch
            if current_res in self.attention_resolutions:
                self.downs.append(SelfAttention(in_ch))
                self.skip_channels.append(in_ch) # Skip from Attention
            
            # Block 2 (Input = Output)
            self.downs.append(Block(in_ch, in_ch, time_emb_dim))
            self.skip_channels.append(in_ch) # Skip from Block 2
            
            # Downsample Layer
            if i < len(channel_mults) - 1:
                self.downs.append(Downsample(in_ch))
                current_res //= 2 

        # The final channel count before bottleneck
        bottleneck_channels = in_ch
        
        # --- Bottleneck ---
        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(Block(bottleneck_channels, bottleneck_channels, time_emb_dim))
        
        if current_res in self.attention_resolutions:
             self.bottleneck.append(SelfAttention(bottleneck_channels))
        
        self.bottleneck.append(Block(bottleneck_channels, bottleneck_channels, time_emb_dim))

        # --- Upsampling Path Initialization ---
        self.ups = nn.ModuleList()
        
        # Reverse the skip channels list for LIFO usage
        skip_channels_rev = self.skip_channels[::-1]
        
        current_ch = bottleneck_channels

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            
            # The number of blocks/attns in the down-stage determines how many skips we use.
            # In the new design, we simply iterate and pop.
            
            # Upsample Layer (except for the very first one after bottleneck)
            if i > 0: 
                self.ups.append(Upsample(current_ch))
            
            # Block 1 
            # Current_ch is the channel from previous upsample/bottleneck. skip_channels_rev[i*3] is the first skip.
            skip_ch1 = skip_channels_rev.pop(0) 
            self.ups.append(Block(current_ch + skip_ch1, out_ch, time_emb_dim))
            current_ch = out_ch
            
            # Attention (if it existed in the down-stage)
            if current_res in self.attention_resolutions:
                skip_ch_attn = skip_channels_rev.pop(0) 
                # Block is Block(in_ch, out_ch, ...)
                self.ups.append(Block(current_ch + skip_ch_attn, out_ch, time_emb_dim))
                current_ch = out_ch
                
            # Block 2
            skip_ch2 = skip_channels_rev.pop(0) 
            self.ups.append(Block(current_ch + skip_ch2, out_ch, time_emb_dim))
            current_ch = out_ch

            current_res *= 2
            
        self.final_conv = nn.Conv2d(current_ch, img_channels, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init_conv(x)
        
        # List to store the features (skips) to be used in the upsampling path
        skips = [x]
        
        # --- Downsampling Path (Execution) ---
        for layer in self.downs:
            if isinstance(layer, Downsample):
                x = layer(x)
            elif isinstance(layer, (Block, SelfAttention)):
                # Block layers take time embedding 't'
                if isinstance(layer, Block):
                    x = layer(x, t)
                # Attention layers do not take 't'
                elif isinstance(layer, SelfAttention):
                    x = layer(x)
                
                # All Block and SelfAttention outputs are used as skip connections
                skips.append(x) 
        
        # --- Bottleneck Path (Execution) ---
        for layer in self.bottleneck:
            if isinstance(layer, Block):
                x = layer(x, t)
            elif isinstance(layer, SelfAttention):
                x = layer(x)

        # --- Upsampling Path (Execution) ---
        # The list of skips is reversed, so we can use .pop() for LIFO retrieval
        skips = skips[::-1]
        
        for layer in self.ups:
            
            if isinstance(layer, Upsample):
                x = layer(x)
                
            else: # Must be a Block layer
                # Pop the most recent skip connection
                skip = skips.pop(0)
                # Concatenate the current feature map and the skip connection
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t)
                
        return self.final_conv(x)


# --- Diffusion Handler ---

class DiffusionHandler(nn.Module):
    """
    Implements BBDM with conditional bridge parameter 's' (fixed or learnable) and selectable objectives.
    Objectives: 'grad', 'noise', 'ysubx'
    """
    def __init__(self, timesteps=1000, s: Optional[float] = None, objective='grad'):
        super().__init__()
        self.timesteps = timesteps
        self.objective = objective
        
        # Validate objective
        if objective not in ['grad', 'noise', 'ysubx']:
            raise ValueError("objective must be one of: 'grad', 'noise', 'ysubx'")

        # --- Conditional 's' setup ---
        if s is not None:
            # Fixed 's': store as a non-trainable buffer
            self.register_buffer('s_value', torch.tensor(float(s)))
            self.learnable_s = False
        else:
            # Learnable 's': store as a learnable parameter, initialized to 1e-4
            initial_s_value = 1e-4 
            self.s_param = nn.Parameter(torch.tensor(float(initial_s_value)))
            self.learnable_s = True
            
        # Precompute m_t
        t_steps = torch.arange(timesteps + 1, dtype=torch.float32)
        m_t = t_steps / timesteps
        self.register_buffer('m_t', m_t.clamp(0.001, 0.999)) 

    def get_current_s(self):
        """ 
        Retrieves the current bridge parameter s value (0-dim tensor).
        Applies softplus if learnable.
        """
        if self.learnable_s:
            # Use softplus for stability and non-negativity
            return F.softplus(self.s_param)
        else:
            # Return the tensor buffer
            return self.s_value

    def get_variance_schedule(self, t):
        """ Dynamically calculates sqrt_delta_t based on the current (conditional) s. """
        
        current_s = self.get_current_s() # 0-dim tensor
        
        # Ensure t is a tensor and on the correct device for indexing
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.m_t.device).long()
            
        m_t_t = self.m_t[t]
        
        # If m_t_t is 0-d or 1-d, expand it for image operations later (B, 1, 1, 1)
        if m_t_t.dim() <= 1: 
            m_t_t = m_t_t.view(-1, 1, 1, 1)

        # current_s is 0-dim and broadcasts correctly with m_t_t
        delta_t = 2 * current_s * (m_t_t - m_t_t.pow(2))
        return m_t_t, torch.sqrt(delta_t + 1e-8)

    def noise_image(self, x_0: torch.Tensor, t: torch.LongTensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        BBDM Forward process. Calculates x_t and the specific target based on self.objective.
        """
        m_t, sqrt_delta_t = self.get_variance_schedule(t)
        noise = torch.randn_like(x_0).to(x_0.device)

        # 1. Generate x_t
        x_t = (1.0 - m_t) * x_0 + m_t * y + sqrt_delta_t * noise

        # 2. Determine Target based on Objective
        if self.objective == 'grad':
            target = x_t - x_0
        elif self.objective == 'noise':
            target = noise
        elif self.objective == 'ysubx':
            target = y - x_0
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        return x_t, target

    def predict_x0_from_objective(self, x_t, y, t, prediction):
        """
        Reconstructs x_0 from the model's prediction based on the objective.
        """
        m_t, sqrt_delta_t = self.get_variance_schedule(t)

        if self.objective == 'grad':
            x_0_pred = x_t - prediction
        elif self.objective == 'noise':
            numerator = x_t - m_t * y - sqrt_delta_t * prediction
            denominator = 1.0 - m_t
            x_0_pred = numerator / (denominator + 1e-8)
        elif self.objective == 'ysubx':
            x_0_pred = y - prediction
            
        return x_0_pred

    @torch.no_grad()
    def sample(self, model, y: torch.Tensor, sample_steps: Optional[int] = None, 
               deterministic=False, verbose=True) -> torch.Tensor:
        """
        BBDM Reverse process.
        """
        model.eval()
        x = y.clone() 
        device = y.device
        
        # Determine the sequence of time steps
        if sample_steps is not None and sample_steps < self.timesteps:
            times = torch.linspace(self.timesteps, 0, sample_steps + 1).long().to(device)
            time_pairs = list(zip(times[:-1], times[1:]))
            total_steps = len(time_pairs)
        else:
            times = torch.arange(self.timesteps, -1, -1).long().to(device)
            time_pairs = list(zip(times[:-1], times[1:]))
            total_steps = self.timesteps

        # Iterate through time pairs (current_t -> next_t)
        for t_curr, t_next in tqdm(time_pairs, total=total_steps, desc=f"Sampling ({self.objective})", disable=not verbose):
            
            # 1. Prepare time tensor for the model (batch size)
            t_tensor = t_curr.repeat(x.shape[0])
            
            # 2. Model Prediction at t_curr
            prediction = model(x, t_tensor) 
            
            # 3. Predict x_0
            x_0_pred = self.predict_x0_from_objective(x, y, t_curr, prediction)
            
            # 4. Step to t_next (Re-noising step)
            if t_next == 0:
                x = x_0_pred
            else:
                m_next, sqrt_delta_next = self.get_variance_schedule(t_next)
                
                # x_{next} = (1 - m_{next}) * x_0 + m_{next} * y + sqrt(delta_{next}) * z
                mean_next = (1.0 - m_next) * x_0_pred + m_next * y
                
                if deterministic:
                    x = mean_next
                else:
                    z = torch.randn_like(x_0_pred)
                    x = mean_next + sqrt_delta_next * z
        
        return x


# --- Helper Functions ---
def plot_loss(loss_history: List[float], filename: str, update_step: int, 
              total_steps: int, interactive: bool, phase_name: str):
    try:
        if interactive and IPYTHON_AVAILABLE:
            clear_output(wait=True)
            plt.figure(figsize=(10, 5))
            plt.plot(loss_history, label=phase_name)
            plt.title(f"{phase_name} Loss (Step {update_step}/{total_steps})")
            plt.xlabel("Training Step")
            plt.ylabel("Loss (MSE)")
            plt.grid(True)
            display(plt.gcf())
            plt.close()
        else:
            plt.figure(figsize=(10, 5))
            plt.plot(loss_history, label=phase_name)
            plt.title(f"{phase_name} Loss (Step {update_step}/{total_steps})")
            plt.xlabel("Training Step")
            plt.ylabel("Loss (MSE)")
            plt.grid(True)
            plt.savefig(filename)
            plt.close()
    except Exception as e:
        if not interactive:
            print(f"Plot Error: {e}")

# --- Training Function ---

def train_one_image(model, image, mask=None, fixed_x_T=None, gt=None, 
                    epochs=500, lr_unet=2e-4, lr_s=1e-2, lr_y=1e-2, 
                    weight_decay=1e-4, 
                    diffusion_handler: Optional[DiffusionHandler]=None, 
                    steps=1000, 
                    s: Optional[float] = None, 
                    objective='grad', 
                    plot_interval=50, 
                    interactive_plot=False, 
                    verbose=True, 
                    meas_rest_mse_each_epoch=None): 
    """ 
    Train bbdm on a single image.
    """
    device = next(model.parameters()).device
    
    # --- Diffusion Handler Initialization (Conditional prints based on verbose) ---
    if diffusion_handler is None:
        diffusion = DiffusionHandler(timesteps=steps, s=s, objective=objective).to(device) 
        if verbose: 
            if diffusion.learnable_s:
                s_val = F.softplus(diffusion.s_param).item()
                print(f"Initializing NEW BBDM handler with objective: '{objective}' and learnable s (Start={s_val:.4f})")
            else:
                s_val = diffusion.s_value.item()
                print(f"Initializing NEW BBDM handler with objective: '{objective}' and FIXED s={s_val:.4f}")
    else:
        diffusion = diffusion_handler.to(device)
        if verbose: 
            if diffusion.learnable_s:
                s_val = F.softplus(diffusion.s_param).item()
                print(f"Continuing training with EXISTING BBDM handler. Objective: '{objective}', Learnable s (Start={s_val:.4f})")
            else:
                s_val = diffusion.s_value.item()
                print(f"Continuing training with EXISTING BBDM handler. Objective: '{objective}', Fixed s={s_val:.4f}")

    # --- Learning Rate Constants ---
    LR_UNET = lr_unet 
    LR_S_PARAM = lr_s 
    LR_Y_PARAM = lr_y
    
    # Constants for weight decay scheduling
    WD_MAX = weight_decay
    WD_MIN = 1e-6 
    
    # --- Image & Latent (y) Setup ---
    loss_history = [] 
    mask_loss_history = []
    
    x_0 = image.to(device)
    
    if mask is None:
        mask = torch.ones(x_0.shape).to(device)
    else:
        mask = mask.to(device)
    
    learnable_y = False
    if fixed_x_T is None:
        img_size = x_0.size()[-1] 
        y = nn.Parameter(torch.randn(1, x_0.shape[1], img_size, img_size).to(device))
        learnable_y = True
        if verbose: print(f"Latent 'y' not provided. Initialized random latent and ENABLED optimization (LR={LR_Y_PARAM}).") 
    else:
        img_size = fixed_x_T.size()[-1]
        y = fixed_x_T.to(device)
        learnable_y = False
        if verbose: print("Latent 'y' provided. Using as FIXED conditioning.") 

    # --- Optimizer Initialization with Parameter Groups ---
    param_groups = [
        {'params': model.parameters(), 'lr': LR_UNET, 'weight_decay': WD_MAX, 'name': 'UNet'}
    ]
    
    if diffusion.learnable_s:
        param_groups.append(
            {'params': diffusion.s_param, 'lr': LR_S_PARAM, 'weight_decay': 1e-4, 'name': 'S_Param'} 
        )
        if verbose: print(f"Optimizer: Added 's' parameter.") 

    if learnable_y:
        param_groups.append(
            {'params': [y], 'lr': LR_Y_PARAM, 'weight_decay': 0.0, 'name': 'Latent_Y'}
        )
        if verbose: print(f"Optimizer: Added 'y' latent parameter.") 

    optimizer = torch.optim.Adam(param_groups)
    
    if verbose: 
        if weight_decay > 0:
            print(f"Regularization ENABLED on UNet: WD_MAX={WD_MAX:.2e} -> WD_MIN={WD_MIN:.2e}")
        else:
            print("Regularization DISABLED")

    # Using tqdm.auto.tqdm ensures Jupyter widget functionality
    # leave=False ensures the bar is cleared after completion
    pbar = tqdm(range(epochs), desc=f"Training ({objective})", position=0, leave=False)

    num = img_size**2
    loss_denum = torch.count_nonzero(mask)
    mask_loss_denum = torch.count_nonzero(1-mask)
    
    for step_in_phase in pbar: 
        model.train() 
        diffusion.train() 
        optimizer.zero_grad() 
        
        # --- Weight Decay Cosine Annealing Schedule ---
        current_wd = 0.0
        if weight_decay > 0:
            cosine_ratio = 0.5 * (1 + math.cos(math.pi * step_in_phase / epochs))
            current_wd = WD_MIN + (WD_MAX - WD_MIN) * cosine_ratio

            optimizer.param_groups[0]['weight_decay'] = current_wd
        
        t = torch.randint(1, diffusion.timesteps + 1, (x_0.shape[0],)).long().to(device) 
        # dbm = mask + (1 - mask) * (t.item() / diffusion.timesteps)  # dynamic binary mask
        
        x_t, target = diffusion.noise_image(x_0, t, y) 
        
        prediction = model(x_t, t)

        loss = F.mse_loss(prediction * mask, target * mask) * num / loss_denum
        
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if meas_rest_mse_each_epoch is not None:
            if step_in_phase % meas_rest_mse_each_epoch == 0:
                if gt is None:
                    print('Ground truth image is not provided')
                else:
                    model.eval()
                    prediction = diffusion.sample(model, y)  
                    mask_loss = F.mse_loss(gt, prediction * (1-mask)) * num / mask_loss_denum
                    mask_loss_history.append(mask_loss.item())
        else:
            pass
        
        current_s_val = diffusion.get_current_s().item()
        s_tag = "s" if diffusion.learnable_s else "s_fixed"
        
        # Update progress bar
        postfix_dict = {
            "loss": f"{loss.item():.5f}", 
            s_tag: f"{current_s_val:.8f}"
        }
        if weight_decay > 0:
            postfix_dict["WD"] = f"{current_wd:.2e}"
            
        pbar.set_postfix(postfix_dict)
        
        if (step_in_phase + 1) % plot_interval == 0:
            plot_loss(loss_history, "loss_plot.png", step_in_phase + 1, epochs, interactive_plot, f"BBDM ({objective})")

    final_s_val = diffusion.get_current_s().item()
    if verbose: 
        if diffusion.learnable_s:
            print(f"Training Complete. Final s: {final_s_val:.4f}")
        else:
            print(f"Training Complete. Fixed s: {final_s_val:.4f}")
        
    plot_loss(loss_history, "loss_plot.png", epochs, epochs, interactive_plot, f"BBDM ({objective})")

    if meas_rest_mse_each_epoch is None:
        return diffusion, loss_history, y
    else:
        return diffusion, loss_history, mask_loss_history, y


    