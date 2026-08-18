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
            # If values look like 0~255, normalize to 0~1
            if any(v > 1.0 for v in vals):
                vals = [v / 255.0 for v in vals]
            return (max(0.0, min(1.0, vals[0])),
                    max(0.0, min(1.0, vals[1])),
                    max(0.0, min(1.0, vals[2])))
        except ValueError:
            pass
            
    return (1.0, 1.0, 1.0)


class AIMZ_AutoMultiplePad:
    """
    Automatically pads images/masks to be divisible by a given multiple (e.g. 32 for MiniMax-H3).
    Completely safe against None inputs (returns None without crashing for easy dynamic branching).
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
                "image": ("IMAGE", {"tooltip": "Optional image input. If None, returns None without error."}),
                "mask": ("MASK", {"tooltip": "Optional mask input. If provided, padded with same padding."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "PAD_INFO")
    RETURN_NAMES = ("image", "mask", "left", "right", "top", "bottom", "pad_info")
    FUNCTION = "pad"
    CATEGORY = "AIMZ/Image"

    def pad(self, multiple=32, pad_mode="constant", pad_color="white", image=None, mask=None):
        if image is None and mask is None:
            return (None, None, 0, 0, 0, 0, None)

        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0
        
        padded_image = None
        padded_mask = None

        # Determine reference H, W
        if image is not None:
            # ComfyUI Image format: [B, H, W, C]
            h, w = image.shape[1], image.shape[2]
        else:
            # Mask format: [B, H, W] or [H, W]
            h, w = mask.shape[-2], mask.shape[-1]

        # Calculate padding needed
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
        }

        # Process Image
        if image is not None:
            if pad_w == 0 and pad_h == 0:
                padded_image = image
            else:
                # [B, H, W, C] -> [B, C, H, W]
                img_t = image.permute(0, 3, 1, 2)
                
                # Check for reflect padding constraint (padding cannot exceed dimension size)
                current_mode = pad_mode
                if current_mode == "reflect":
                    if pad_left >= w or pad_right >= w or pad_top >= h or pad_bottom >= h:
                        current_mode = "replicate"

                if current_mode == "constant":
                    r, g, b = parse_color(pad_color)
                    # Constant padding in PyTorch accepts single scalar value for N-D tensors
                    # For multi-channel RGB, pad with 0 then fill borders
                    img_padded = F.pad(img_t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
                    if r != 0.0 or g != 0.0 or b != 0.0:
                        # Colorize padded region
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

        # Process Mask
        if mask is not None:
            if pad_w == 0 and pad_h == 0:
                padded_mask = mask
            else:
                m = mask
                if m.dim() == 2:
                    m = m.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                elif m.dim() == 3:
                    m = m.unsqueeze(1)  # [B, 1, H, W]
                
                mask_mode = pad_mode if pad_mode in ["reflect", "replicate", "circular"] else "constant"
                if mask_mode == "reflect" and (pad_left >= w or pad_right >= w or pad_top >= h or pad_bottom >= h):
                    mask_mode = "replicate"

                m_padded = F.pad(m, (pad_left, pad_right, pad_top, pad_bottom), mode=mask_mode, value=0.0)
                padded_mask = m_padded.squeeze(1)

        return (padded_image, padded_mask, pad_left, pad_right, pad_top, pad_bottom, pad_info)


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
