import os
import random
import tempfile
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    import folder_paths
except ImportError:
    folder_paths = None

def parse_color(color_str):
    """
    Parses color string (e.g. '0,0,0', '1,1,1', '255,255,255', 'black', 'white', '#ffffff')
    into normalized RGB float tuple (r, g, b) in [0.0, 1.0].
    """
    if not color_str:
        return (1.0, 1.0, 1.0)
    
    color_str = color_str.strip().lower()
    named_colors = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "gray": (0.5, 0.5, 0.5),
        "grey": (0.5, 0.5, 0.5),
    }
    if color_str in named_colors:
        return named_colors[color_str]
    
    if color_str.startswith("#"):
        hex_val = color_str.lstrip("#")
        if len(hex_val) == 6:
            r = int(hex_val[0:2], 16) / 255.0
            g = int(hex_val[2:4], 16) / 255.0
            b = int(hex_val[4:6], 16) / 255.0
            return (r, g, b)
    
    parts = [p.strip() for p in color_str.split(",") if p.strip()]
    if len(parts) >= 3:
        try:
            vals = [float(p) for p in parts[:3]]
            if any(v > 1.0 for v in vals):
                vals = [v / 255.0 for v in vals]
            return (max(0.0, min(1.0, vals[0])),
                    max(0.0, min(1.0, vals[1])),
                    max(0.0, min(1.0, vals[2])))
        except ValueError:
            pass
            
    return (1.0, 1.0, 1.0)


# -------------------------------------------------------------
# Color Space Conversion Helper Functions (Pure PyTorch)
# -------------------------------------------------------------
def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    Converts RGB tensor [..., 3] with values in [0, 1] to CIE-LAB.
    """
    mask = rgb > 0.04045
    rgb_lin = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

    r = rgb_lin[..., 0]
    g = rgb_lin[..., 1]
    b = rgb_lin[..., 2]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883

    delta = 6.0 / 29.0
    mask_x = x > (delta ** 3)
    mask_y = y > (delta ** 3)
    mask_z = z > (delta ** 3)

    fx = torch.where(mask_x, torch.pow(torch.clamp(x, min=1e-6), 1.0 / 3.0), (x / (3.0 * delta ** 2)) + (4.0 / 29.0))
    fy = torch.where(mask_y, torch.pow(torch.clamp(y, min=1e-6), 1.0 / 3.0), (y / (3.0 * delta ** 2)) + (4.0 / 29.0))
    fz = torch.where(mask_z, torch.pow(torch.clamp(z, min=1e-6), 1.0 / 3.0), (z / (3.0 * delta ** 2)) + (4.0 / 29.0))

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    return torch.stack([L, a, b_val], dim=-1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """
    Converts CIE-LAB tensor [..., 3] to RGB with values in [0, 1].
    """
    L = lab[..., 0]
    a = lab[..., 1]
    b_val = lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (b_val / 200.0)

    delta = 6.0 / 29.0
    mask_x = fx > delta
    mask_y = fy > delta
    mask_z = fz > delta

    x = torch.where(mask_x, torch.pow(fx, 3.0), 3.0 * (delta ** 2) * (fx - 4.0 / 29.0)) * 0.95047
    y = torch.where(mask_y, torch.pow(fy, 3.0), 3.0 * (delta ** 2) * (fy - 4.0 / 29.0)) * 1.00000
    z = torch.where(mask_z, torch.pow(fz, 3.0), 3.0 * (delta ** 2) * (fz - 4.0 / 29.0)) * 1.08883

    r_lin = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
    g_lin = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560
    b_lin = x * 0.0556434 - y * 0.2040259 + z * 1.0572252

    r_lin = torch.clamp(r_lin, min=0.0)
    g_lin = torch.clamp(g_lin, min=0.0)
    b_lin = torch.clamp(b_lin, min=0.0)

    mask_r = r_lin > 0.0031308
    mask_g = g_lin > 0.0031308
    mask_b = b_lin > 0.0031308

    r = torch.where(mask_r, 1.055 * torch.pow(r_lin, 1.0 / 2.4) - 0.055, 12.92 * r_lin)
    g = torch.where(mask_g, 1.055 * torch.pow(g_lin, 1.0 / 2.4) - 0.055, 12.92 * g_lin)
    b = torch.where(mask_b, 1.055 * torch.pow(b_lin, 1.0 / 2.4) - 0.055, 12.92 * b_lin)

    rgb = torch.stack([r, g, b], dim=-1)
    return torch.clamp(rgb, 0.0, 1.0)


class AIMZ_VAEColorMatch:
    """
    Robust VAE Color Matcher and Drift Corrector for Video and Image Pipelines.
    Dual Mode Support:
    1. Standalone Post-Processor: Put after VAEDecode (connect original_image + processed_image).
    2. All-in-One VAE Decoder: Connect original_image + samples (LATENT) + vae (VAE) to decode and match colors in 1 step!
    100% immune to frame-count mismatches (e.g. MiniMax 243 vs LTX 241 frames) and resolution differences!
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE", {"tooltip": "Reference original image/video (before VAE decode / 1st pass)"}),
                "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Correction strength (0.0 = no change, 1.0 = full color match)"}),
                "method": (["lab_moments (Reinhard)", "luminance_zones", "rgb_moments"], {"default": "lab_moments (Reinhard)", "tooltip": "Color matching algorithm:\n• lab_moments: Best perceptual color and tone transfer\n• luminance_zones: Preserves local lighting while correcting tints\n• rgb_moments: Fast global RGB channel match"}),
            },
            "optional": {
                "processed_image": ("IMAGE", {"tooltip": "Option A: Connect image from VAEDecode"}),
                "samples": ("LATENT", {"tooltip": "Option B: Connect Latent directly (auto-decodes if vae is connected!)"}),
                "vae": ("VAE", {"tooltip": "Option B: Connect VAE model to decode samples automatically"}),
                "mask": ("MASK", {"tooltip": "Optional mask to limit correction areas"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_image",)
    FUNCTION = "match_colors"
    CATEGORY = "AIMZ/Color"

    def match_colors(self, original_image, strength=0.85, method="lab_moments (Reinhard)", processed_image=None, samples=None, vae=None, mask=None):
        proc = processed_image

        # Auto-Decode if samples (LATENT) and vae are provided instead of processed_image
        if proc is None and samples is not None and vae is not None:
            latent_tensor = samples.get("samples")
            if latent_tensor is not None:
                proc = vae.decode(latent_tensor)

        if original_image is None or proc is None:
            return (proc if proc is not None else original_image,)

        if strength <= 0.001:
            return (proc,)

        orig = original_image
        device = proc.device
        dtype = proc.dtype
        orig = orig.to(device=device, dtype=dtype)

        # Batch frame count safety: handle frame mismatch (e.g., 243 vs 241 frames)
        proc_b, proc_h, proc_w, _ = proc.shape
        orig_b, orig_h, orig_w, _ = orig.shape

        corrected_frames = []

        for i in range(proc_b):
            # Safe index mapping: clamp safely to original frame index
            orig_idx = min(i, orig_b - 1)
            
            orig_f = orig[orig_idx:orig_idx+1]
            proc_f = proc[i:i+1]

            if method == "lab_moments (Reinhard)":
                orig_lab = rgb_to_lab(orig_f)
                proc_lab = rgb_to_lab(proc_f)

                orig_mean = orig_lab.mean(dim=(1, 2), keepdim=True)
                orig_std = orig_lab.std(dim=(1, 2), keepdim=True) + 1e-6

                proc_mean = proc_lab.mean(dim=(1, 2), keepdim=True)
                proc_std = proc_lab.std(dim=(1, 2), keepdim=True) + 1e-6

                matched_lab = (proc_lab - proc_mean) * (orig_std / proc_std) + orig_mean
                blended_lab = proc_lab * (1.0 - strength) + matched_lab * strength
                
                res_rgb = lab_to_rgb(blended_lab)

            elif method == "luminance_zones":
                mean_orig = orig_f.mean(dim=(1, 2), keepdim=True)
                mean_proc = proc_f.mean(dim=(1, 2), keepdim=True)
                bias = mean_orig - mean_proc

                matched_rgb = proc_f + bias
                matched_rgb = torch.clamp(matched_rgb, 0.0, 1.0)
                res_rgb = proc_f * (1.0 - strength) + matched_rgb * strength

            else: # rgb_moments
                orig_mean = orig_f.mean(dim=(1, 2), keepdim=True)
                orig_std = orig_f.std(dim=(1, 2), keepdim=True) + 1e-6

                proc_mean = proc_f.mean(dim=(1, 2), keepdim=True)
                proc_std = proc_f.std(dim=(1, 2), keepdim=True) + 1e-6

                matched_rgb = (proc_f - proc_mean) * (orig_std / proc_std) + orig_mean
                matched_rgb = torch.clamp(matched_rgb, 0.0, 1.0)
                res_rgb = proc_f * (1.0 - strength) + matched_rgb * strength

            corrected_frames.append(res_rgb)

        result_tensor = torch.cat(corrected_frames, dim=0)

        # Apply optional mask if provided
        if mask is not None:
            m = mask.to(device=device, dtype=dtype)
            if m.dim() == 2:
                m = m.unsqueeze(0).unsqueeze(-1)
            elif m.dim() == 3:
                m = m.unsqueeze(-1)
            
            if m.shape[1:3] != (proc_h, proc_w):
                m_t = m.permute(0, 3, 1, 2)
                m_t = F.interpolate(m_t, size=(proc_h, proc_w), mode="bilinear", align_corners=False)
                m = m_t.permute(0, 2, 3, 1)

            if m.shape[0] < proc_b:
                m = m.repeat(proc_b // m.shape[0] + 1, 1, 1, 1)[:proc_b]

            result_tensor = proc * (1.0 - m) + result_tensor * m

        return (result_tensor,)


class AIMZ_AutoMultiplePad:
    """
    Automatically pads images/masks/video frames to be divisible by a given multiple (e.g. 32).
    Outputs padded image, mask, original dimensions (width, height), frame count, padding offsets, and pad_info.
    Completely safe against None inputs (returns None/0 without crashing for easy dynamic branching).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multiple": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1, "tooltip": "Resolution multiple (e.g., 32 for MiniMax-H3)"}),
                "pad_mode": (["constant", "reflect", "replicate", "circular"], {"default": "constant"}),
                "pad_color": ("STRING", {"default": "white", "tooltip": "Used when pad_mode is constant (e.g. 'white', 'black', '1,1,1', '0,0,0', '#ffffff')"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image/video frame batch input. If None, returns None without error."}),
                "mask": ("MASK", {"tooltip": "Optional mask input. If provided, padded with same padding."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "PAD_INFO")
    RETURN_NAMES = ("image", "mask", "width", "height", "count", "left", "right", "top", "bottom", "pad_info")
    FUNCTION = "pad"
    CATEGORY = "AIMZ/Image"

    def pad(self, multiple=32, pad_mode="constant", pad_color="white", image=None, mask=None):
        if image is None and mask is None:
            return (None, None, 0, 0, 0, 0, 0, 0, 0, None)

        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0
        
        padded_image = None
        padded_mask = None
        count = 0

        if image is not None:
            count = image.shape[0]
            h, w = image.shape[1], image.shape[2]
        else:
            if mask.dim() == 2:
                count = 1
                h, w = mask.shape[0], mask.shape[1]
            else:
                count = mask.shape[0]
                h, w = mask.shape[1], mask.shape[2]

        pad_w = ((w + multiple - 1) // multiple * multiple) - w
        pad_h = ((h + multiple - 1) // multiple * multiple) - h

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_info = {
            "pad_left": pad_left,
            "pad_right": pad_right,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "orig_width": w,
            "orig_height": h,
            "padded_width": w + pad_w,
            "padded_height": h + pad_h,
            "count": count,
        }

        if image is not None:
            if pad_w == 0 and pad_h == 0:
                padded_image = image
            else:
                img_t = image.permute(0, 3, 1, 2)
                
                current_mode = pad_mode
                if current_mode == "reflect":
                    if pad_left >= w or pad_right >= w or pad_top >= h or pad_bottom >= h:
                        current_mode = "replicate"

                if current_mode == "constant":
                    r, g, b = parse_color(pad_color)
                    img_padded = F.pad(img_t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
                    if r != 0.0 or g != 0.0 or b != 0.0:
                        color_tensor = torch.tensor([r, g, b], device=img_t.device, dtype=img_t.dtype).view(1, 3, 1, 1)
                        mask_border = torch.ones_like(img_padded)
                        if pad_top > 0: mask_border[:, :, :pad_top, :] = 0
                        if pad_bottom > 0: mask_border[:, :, -pad_bottom:, :] = 0
                        if pad_left > 0: mask_border[:, :, :, :pad_left] = 0
                        if pad_right > 0: mask_border[:, :, :, -pad_right:] = 0
                        img_padded = torch.where(mask_border == 1, img_padded, color_tensor)
                else:
                    img_padded = F.pad(img_t, (pad_left, pad_right, pad_top, pad_bottom), mode=current_mode)

                padded_image = img_padded.permute(0, 2, 3, 1)

        if mask is not None:
            if pad_w == 0 and pad_h == 0:
                padded_mask = mask
            else:
                m = mask
                if m.dim() == 2:
                    m = m.unsqueeze(0).unsqueeze(0)
                elif m.dim() == 3:
                    m = m.unsqueeze(1)
                
                mask_mode = pad_mode if pad_mode in ["reflect", "replicate", "circular"] else "constant"
                if mask_mode == "reflect" and (pad_left >= w or pad_right >= w or pad_top >= h or pad_bottom >= h):
                    mask_mode = "replicate"

                m_padded = F.pad(m, (pad_left, pad_right, pad_top, pad_bottom), mode=mask_mode, value=0.0)
                padded_mask = m_padded.squeeze(1)

        return (padded_image, padded_mask, w, h, count, pad_left, pad_right, pad_top, pad_bottom, pad_info)


class AIMZ_AutoMultipleUnpad:
    """
    Restores padded images/masks back to their original dimensions using pad_info or manual coordinates.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "pad_info": ("PAD_INFO",),
                "left": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "right": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "top": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 8192}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "unpad"
    CATEGORY = "AIMZ/Image"

    def unpad(self, image=None, mask=None, pad_info=None, left=0, right=0, top=0, bottom=0):
        if image is None and mask is None:
            return (None, None)

        if pad_info is not None:
            p_left = pad_info.get("pad_left", 0)
            p_right = pad_info.get("pad_right", 0)
            p_top = pad_info.get("pad_top", 0)
            p_bottom = pad_info.get("pad_bottom", 0)
        else:
            p_left = left
            p_right = right
            p_top = top
            p_bottom = bottom

        unpadded_image = None
        unpadded_mask = None

        if image is not None:
            h, w = image.shape[1], image.shape[2]
            end_h = h - p_bottom if p_bottom > 0 else h
            end_w = w - p_right if p_right > 0 else w
            unpadded_image = image[:, p_top:end_h, p_left:end_w, :]

        if mask is not None:
            if mask.dim() == 2:
                h, w = mask.shape[0], mask.shape[1]
                end_h = h - p_bottom if p_bottom > 0 else h
                end_w = w - p_right if p_right > 0 else w
                unpadded_mask = mask[p_top:end_h, p_left:end_w]
            else:
                h, w = mask.shape[1], mask.shape[2]
                end_h = h - p_bottom if p_bottom > 0 else h
                end_w = w - p_right if p_right > 0 else w
                unpadded_mask = mask[:, p_top:end_h, p_left:end_w]

        return (unpadded_image, unpadded_mask)


class AIMZ_FreezeFramePad:
    """
    Pads a video frame sequence with repeated freeze frames at the start and end.
    Replaces 5 separate nodes (GetImageFromBatch x2, RepeatImages x2, ImageBatchMulti).
    Completely safe against None inputs (returns None without crashing for optional video branches).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pad_start_frames": ("INT", {"default": 15, "min": 0, "max": 1000, "step": 1, "tooltip": "Number of freeze frames to prepend to the start of the video"}),
                "pad_end_frames": ("INT", {"default": 15, "min": 0, "max": 1000, "step": 1, "tooltip": "Number of freeze frames to append to the end of the video"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Input video frames [B, H, W, C]. If None, returns None without error."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "pad_frames"
    CATEGORY = "AIMZ/Video"

    def pad_frames(self, pad_start_frames=15, pad_end_frames=15, image=None):
        if image is None:
            return (None, 0)
        
        try:
            if not isinstance(image, torch.Tensor) or image.dim() < 4 or image.shape[0] == 0:
                return (None, 0)
        except Exception:
            return (None, 0)

        chunks = []
        if pad_start_frames > 0:
            first_frame = image[0:1].repeat(pad_start_frames, 1, 1, 1)
            chunks.append(first_frame)
        
        chunks.append(image)

        if pad_end_frames > 0:
            last_frame = image[-1:].repeat(pad_end_frames, 1, 1, 1)
            chunks.append(last_frame)

        padded_video = torch.cat(chunks, dim=0)
        return (padded_video, padded_video.shape[0])


class AIMZ_AudioSilencePad:
    """
    Pads an audio sequence with exact silence at the start and end based on video frame counts.
    Replaces 8~10 separate nodes (Int x2, Math Expression x2, Empty Audio x2, Concatenate Audio x2).
    Automatically matches sample rate and channel layout, and supports dynamic FPS override.
    Completely safe against None inputs (returns None without crashing for optional audio branches).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pad_start_frames": ("INT", {"default": 15, "min": 0, "max": 10000, "step": 1, "tooltip": "Number of frames to pad with silence at the start"}),
                "pad_end_frames": ("INT", {"default": 15, "min": 0, "max": 10000, "step": 1, "tooltip": "Number of frames to pad with silence at the end"}),
                "default_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0, "tooltip": "Default FPS used when source_fps is not connected"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Input audio dictionary {'waveform': tensor, 'sample_rate': int}. If None, returns None without error."}),
                "source_fps": ("FLOAT", {"forceInput": True, "tooltip": "Optional source video FPS (from Get_FPS or video loader, overrides default_fps)"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio", "total_duration", "pad_start_sec", "pad_end_sec")
    FUNCTION = "pad_audio"
    CATEGORY = "AIMZ/Audio"

    def pad_audio(self, pad_start_frames=15, pad_end_frames=15, default_fps=24.0, audio=None, source_fps=None):
        if audio is None:
            return (None, 0.0, 0.0, 0.0)

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 44100)

        if waveform is None or not isinstance(waveform, torch.Tensor) or waveform.numel() == 0:
            return (None, 0.0, 0.0, 0.0)

        if source_fps is not None and isinstance(source_fps, (int, float)) and source_fps > 0:
            fps = float(source_fps)
        else:
            fps = float(default_fps) if default_fps > 0 else 24.0

        start_sec = (pad_start_frames / fps) if pad_start_frames > 0 else 0.0
        end_sec = (pad_end_frames / fps) if pad_end_frames > 0 else 0.0

        num_start_samples = int(round(start_sec * sample_rate))
        num_end_samples = int(round(end_sec * sample_rate))

        shape_prefix = list(waveform.shape[:-1])
        
        chunks = []
        if num_start_samples > 0:
            silence_start = torch.zeros(*shape_prefix, num_start_samples, dtype=waveform.dtype, device=waveform.device)
            chunks.append(silence_start)
        
        chunks.append(waveform)

        if num_end_samples > 0:
            silence_end = torch.zeros(*shape_prefix, num_end_samples, dtype=waveform.dtype, device=waveform.device)
            chunks.append(silence_end)

        padded_waveform = torch.cat(chunks, dim=-1)
        total_duration = round(padded_waveform.shape[-1] / sample_rate, 3)

        result_audio = {
            "waveform": padded_waveform,
            "sample_rate": sample_rate
        }

        return (result_audio, total_duration, round(start_sec, 4), round(end_sec, 4))


class AIMZ_FallbackSwitch:
    """
    Universal Any-Type Fallback Switch (Coalesce).
    Accepts ANY data type (IMAGE, AUDIO, LATENT, MODEL, CONDITIONING, etc.) like Get/Set nodes.
    If 'primary' is not None, it passes 'primary'.
    If 'primary' is None, it seamlessly outputs 'fallback'.
    If both are None, cleanly returns None without error.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "primary": ("*", {"tooltip": "1st Priority input. If not None, this value is passed through."}),
                "fallback": ("*", {"tooltip": "2nd Priority input. Passed through if primary is None."}),
                "fallback_2": ("*", {"tooltip": "Optional 3rd Priority fallback."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "coalesce"
    CATEGORY = "AIMZ/Workflow"

    def coalesce(self, primary=None, fallback=None, fallback_2=None):
        if primary is not None:
            return (primary,)
        if fallback is not None:
            return (fallback,)
        if fallback_2 is not None:
            return (fallback_2,)
        return (None,)


class AIMZ_PreviewImageNoneSafe:
    """
    A None-Safe Image Preview node.
    If image is None, it cleanly skips without crashing.
    If image exists, it saves a temp preview and outputs the image as pass-through.
    """
    def __init__(self):
        if folder_paths is not None:
            self.output_dir = folder_paths.get_temp_directory()
        else:
            self.output_dir = tempfile.gettempdir()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstubwxyz") for _ in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE", {"tooltip": "Optional image input. If None, cleanly skips without error."}),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "AIMZ/Image"

    def preview(self, images=None, filename_prefix="ComfyUI_AIMZ_Preview", prompt=None, extra_pnginfo=None):
        if images is None:
            return {"ui": {"images": []}, "result": (None,)}

        try:
            if not isinstance(images, torch.Tensor) or images.dim() < 3 or images.shape[0] == 0:
                return {"ui": {"images": []}, "result": (None,)}
        except Exception:
            return {"ui": {"images": []}, "result": (None,)}

        filename_prefix += self.prefix_append
        
        if folder_paths is not None:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        else:
            full_output_folder = self.output_dir
            filename = filename_prefix
            counter = 0
            subfolder = ""

        results = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), compress_level=1)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}, "result": (images,)}


class AIMZ_SelectiveGroupBypasser:
    """
    Selectively manages and toggles bypass states for user-chosen groups only.
    Unlike Fast Groups Bypasser which lists every group, this lets you curate
    a clean list of specific groups to toggle Active / Bypass directly from the node.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "opt_connection": ("*", {"tooltip": "Optional flow-through connection"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("opt_connection",)
    FUNCTION = "passthrough"
    OUTPUT_NODE = True
    CATEGORY = "AIMZ/Workflow"

    def passthrough(self, opt_connection=None):
        return (opt_connection,)


class AIMZ_VideoDurationSelector:
    """
    Smart Video Duration and Frame Calculator for V2V and R2V pipelines.
    - In Source Video (V2V) mode: strictly follows the source video duration (if 0.0 or not connected, outputs 0 frames).
    - In Custom Seconds (R2V) mode: uses custom_seconds to calculate exact MiniMax H3 frames.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["Source Video (V2V)", "Custom Seconds (R2V)"], {"default": "Source Video (V2V)"}),
                "custom_seconds": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 600.0, "step": 0.1, "round": 0.01, "tooltip": "Desired duration in seconds for R2V mode"}),
                "default_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0, "tooltip": "Default FPS used when source_fps is not connected"}),
                "minimax_align": ("BOOLEAN", {"default": True, "tooltip": "Align frame count to MiniMax H3 formula: max(5, round(sec * fps)) + (5 - (max(...) % 17)) % 17"}),
            },
            "optional": {
                "source_duration": ("FLOAT", {"forceInput": True, "tooltip": "Connect Get_duration (seconds)"}),
                "source_fps": ("FLOAT", {"forceInput": True, "tooltip": "Connect Get_FPS"}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("total_frames", "final_seconds", "effective_fps", "raw_frames")
    FUNCTION = "calculate_duration"
    CATEGORY = "AIMZ/Workflow"

    def calculate_duration(self, mode="Source Video (V2V)", custom_seconds=6.0, default_fps=24.0, minimax_align=True, source_duration=None, source_fps=None):
        if source_fps is not None and isinstance(source_fps, (int, float)) and source_fps > 0:
            effective_fps = float(source_fps)
        else:
            effective_fps = float(default_fps) if default_fps > 0 else 24.0

        if mode == "Source Video (V2V)":
            if source_duration is not None and isinstance(source_duration, (int, float)) and source_duration > 0:
                effective_sec = float(source_duration)
            else:
                return (0, 0.0, effective_fps, 0)
        else:
            effective_sec = float(custom_seconds)

        raw_frames = max(5, int(round(effective_sec * effective_fps)))

        if minimax_align:
            remainder = raw_frames % 17
            pad = (5 - remainder) % 17
            total_frames = raw_frames + pad
        else:
            total_frames = raw_frames

        final_seconds = round(total_frames / effective_fps, 3)

        return (total_frames, final_seconds, effective_fps, raw_frames)
