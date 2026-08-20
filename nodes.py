import os
import random
import tempfile
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    import cv2
    ADVANCED_CV2_AVAILABLE = True
except ImportError:
    ADVANCED_CV2_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from skimage import exposure
    ADVANCED_SKLEARN_AVAILABLE = True
except ImportError:
    ADVANCED_SKLEARN_AVAILABLE = False

try:
    import folder_paths
except ImportError:
    folder_paths = None

def parse_color(color_str):
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
# 1:1 Direct Port of VAEColorCorrector Algorithms from EasyColorCorrector
# -------------------------------------------------------------
def _safe_clamp_colors(image_np, preserve_gradients=True):
    if preserve_gradients:
        image_float = image_np.astype(np.float32)
        below_zero = image_float < 0
        above_255 = image_float > 255
        if np.any(below_zero):
            negative_values = image_float[below_zero]
            image_float[below_zero] = -5 * np.log(1 + np.exp(-negative_values / 5))
        if np.any(above_255):
            high_values = image_float[above_255]
            image_float[above_255] = 255 + 5 * np.log(1 + np.exp((high_values - 255) / 5))
        return np.clip(image_float, 0, 255).astype(np.uint8)
    else:
        return np.clip(image_np, 0, 255).astype(np.uint8)

def _match_to_reference_colors(processed_np, original_np, strength=0.8):
    proc_float = processed_np.astype(np.float32)
    orig_float = original_np.astype(np.float32)
    mean_orig = np.mean(orig_float, axis=(0, 1), keepdims=True)
    mean_proc = np.mean(proc_float, axis=(0, 1), keepdims=True)
    std_orig = np.std(orig_float, axis=(0, 1), keepdims=True) + 1e-6
    std_proc = np.std(proc_float, axis=(0, 1), keepdims=True) + 1e-6
    normalized = (proc_float - mean_proc) / std_proc
    rescaled = normalized * std_orig + mean_orig
    matched = proc_float * (1.0 - strength) + rescaled * strength
    return _safe_clamp_colors(matched, preserve_gradients=True)


class AIMZ_VAEColorMatch:
    """
    1:1 Exact Port of VAEColorCorrector (EasyColorCorrector) with full batch/frame mismatch safety,
    resolution auto-scaling, and pure PyTorch/NumPy backend support.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE", {"tooltip": "Original reference image/video (before VAE decode / 1st pass)"}),
                "processed_image": ("IMAGE", {"tooltip": "Image/video after VAE decode with color shifts"}),
                "correction_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Correction strength (0.0 = no change, 1.0 = full color match)"}),
                "method": (
                    ["advanced_3d_lut", "luminance_zones", "histogram_matching", "statistical_matching"], 
                    {"default": "advanced_3d_lut", "tooltip": "Color correction method from VAEColorCorrector"}
                ),
                "auto_preserve": ("BOOLEAN", {"default": False, "tooltip": "Auto-detect and preserve changed/inpainted areas (set False to correct entire full frame)"}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Optional VAE model input for pipeline consistency"}),
                "mask": ("MASK", {"tooltip": "Optional mask - white areas will be preserved"}),
                "edge_feather": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1, "tooltip": "Feather edges between corrected/preserved areas"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_image",)
    FUNCTION = "correct_vae_colors"
    CATEGORY = "AIMZ/Color"

    def correct_vae_colors(
        self, original_image, processed_image, correction_strength=0.85, 
        method="advanced_3d_lut", auto_preserve=False, vae=None, mask=None, edge_feather=5
    ):
        if original_image is None or processed_image is None:
            return (processed_image if processed_image is not None else original_image,)

        if correction_strength <= 0.001:
            return (processed_image,)

        device = processed_image.device
        
        orig_b, orig_h, orig_w, _ = original_image.shape
        proc_b, proc_h, proc_w, _ = processed_image.shape

        # Match spatial resolution if different
        if (orig_h, orig_w) != (proc_h, proc_w):
            print(f"⚠️ Resolution mismatch: resizing original {orig_w}x{orig_h} to match processed {proc_w}x{proc_h}")
            orig_scaled = F.interpolate(
                original_image.permute(0, 3, 1, 2),
                size=(proc_h, proc_w),
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            orig_scaled = original_image

        corrected_batch = []

        # Process each frame in processed_image safely
        for i in range(proc_b):
            orig_idx = min(i, orig_b - 1)
            orig_img = orig_scaled[orig_idx]
            proc_img = processed_image[i]
            current_mask = mask[i] if (mask is not None and i < mask.shape[0]) else (mask[0] if mask is not None else None)

            # Convert to numpy uint8
            orig_np = (orig_img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            proc_np = (proc_img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            # 1. Analyze VAE Characteristics
            vae_adjustment = 1.0
            vae_color_bias = None
            if vae is not None:
                vae_adjustment, vae_color_bias = self._analyze_vae_characteristics(orig_np, proc_np)

            # 2. Balance Strength
            adj_strength = self._balance_correction_strength(method, correction_strength) * vae_adjustment
            adj_strength = min(1.0, max(0.0, adj_strength))

            # 3. Apply Method
            if method == "advanced_3d_lut":
                corrected_np = self._advanced_3d_lut_correction(orig_np, proc_np, adj_strength, vae_color_bias)
            elif method == "luminance_zones":
                if vae_color_bias is not None:
                    corrected_np = self._vae_aware_luminance_correction(proc_np, orig_np, adj_strength, vae_color_bias)
                else:
                    corrected_np = _match_to_reference_colors(proc_np, orig_np, adj_strength)
            elif method == "histogram_matching":
                corrected_np = self._histogram_matching_correction(orig_np, proc_np, adj_strength, vae_color_bias)
            else: # statistical_matching
                corrected_np = self._statistical_matching_correction(orig_np, proc_np, adj_strength, vae_color_bias)

            corrected_tensor = torch.from_numpy(corrected_np.astype(np.float32) / 255.0).to(device)

            # 4. Handle Mask / Auto-preserve
            if current_mask is not None:
                corrected_tensor = self._apply_mask_preservation(proc_img, corrected_tensor, current_mask, edge_feather, device)
            elif auto_preserve:
                corrected_tensor = self._auto_preserve_inpainted(orig_img, proc_img, corrected_tensor, edge_feather, device)

            corrected_batch.append(corrected_tensor)

        result = torch.stack(corrected_batch, dim=0)
        return (result,)

    def _analyze_vae_characteristics(self, original_np, processed_np):
        try:
            orig_float = original_np.astype(np.float32)
            proc_float = processed_np.astype(np.float32)
            gray_orig = np.mean(orig_float, axis=2)

            shadows_mask = gray_orig < 85
            midtones_mask = (gray_orig >= 85) & (gray_orig <= 170)
            highlights_mask = gray_orig > 170

            vae_bias = {}
            for zone_name, mask in [("shadows", shadows_mask), ("midtones", midtones_mask), ("highlights", highlights_mask)]:
                if np.sum(mask) > 100:
                    orig_zone = orig_float[mask]
                    proc_zone = proc_float[mask]
                    bias_r = np.mean(proc_zone[:, 0]) - np.mean(orig_zone[:, 0])
                    bias_g = np.mean(proc_zone[:, 1]) - np.mean(orig_zone[:, 1])
                    bias_b = np.mean(proc_zone[:, 2]) - np.mean(orig_zone[:, 2])
                    vae_bias[zone_name] = np.array([bias_r, bias_g, bias_b])
                else:
                    vae_bias[zone_name] = np.array([0.0, 0.0, 0.0])

            total_bias = np.mean([np.abs(bias).sum() for bias in vae_bias.values()])
            if total_bias > 15:
                vae_adjustment = 0.85
            elif total_bias > 8:
                vae_adjustment = 0.92
            else:
                vae_adjustment = 0.98

            return vae_adjustment, vae_bias
        except Exception:
            return 0.9, None

    def _balance_correction_strength(self, method, strength):
        if method == "luminance_zones":
            return strength
        elif method == "histogram_matching":
            return strength * 0.85
        elif method == "statistical_matching":
            return min(1.0, strength * 1.15)
        elif method == "advanced_3d_lut":
            return strength * 0.7
        return strength

    def _advanced_3d_lut_correction(self, original_np, processed_np, strength, vae_color_bias=None):
        if not ADVANCED_SKLEARN_AVAILABLE:
            return _match_to_reference_colors(processed_np, original_np, strength)
        try:
            orig_samples = original_np[::4, ::4].reshape(-1, 3)
            proc_samples = processed_np[::4, ::4].reshape(-1, 3)
            n_clusters = min(64, max(8, len(orig_samples) // 10))

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            proc_clusters = kmeans.fit_predict(proc_samples)
            proc_centers = kmeans.cluster_centers_

            orig_centers = np.zeros_like(proc_centers)
            for i in range(n_clusters):
                cluster_mask = proc_clusters == i
                if np.sum(cluster_mask) > 0:
                    orig_centers[i] = np.mean(orig_samples[cluster_mask], axis=0)
                else:
                    orig_centers[i] = proc_centers[i]

            proc_flat = processed_np.reshape(-1, 3).astype(np.float32)
            distances = np.linalg.norm(proc_flat[:, np.newaxis, :] - proc_centers[np.newaxis, :, :], axis=2)
            closest_clusters = np.argmin(distances, axis=1)

            min_distances = np.min(distances, axis=1)
            max_distance = np.percentile(min_distances, 90) + 1e-6

            for i in range(n_clusters):
                cluster_mask = closest_clusters == i
                if np.sum(cluster_mask) > 0:
                    color_shift = orig_centers[i] - proc_centers[i]
                    if vae_color_bias is not None:
                        cluster_luminance = np.mean(proc_centers[i])
                        if cluster_luminance < 85:
                            zone_bias = vae_color_bias.get("shadows", np.array([0.0, 0.0, 0.0]))
                        elif cluster_luminance <= 170:
                            zone_bias = vae_color_bias.get("midtones", np.array([0.0, 0.0, 0.0]))
                        else:
                            zone_bias = vae_color_bias.get("highlights", np.array([0.0, 0.0, 0.0]))
                        color_shift -= zone_bias * 0.5

                    cluster_distances = min_distances[cluster_mask]
                    distance_weights = np.clip(1.0 - cluster_distances / max_distance, 0.1, 1.0)
                    for c in range(3):
                        shift_amount = color_shift[c] * strength * distance_weights
                        shift_amount = np.clip(shift_amount, -50, 50)
                        proc_flat[cluster_mask, c] += shift_amount

            corrected_np = proc_flat.reshape(processed_np.shape)
            return _safe_clamp_colors(corrected_np, preserve_gradients=True)
        except Exception:
            return _match_to_reference_colors(processed_np, original_np, strength)

    def _vae_aware_luminance_correction(self, processed_np, original_np, strength, vae_color_bias):
        try:
            corrected_np = _match_to_reference_colors(processed_np, original_np, strength)
            corrected_float = corrected_np.astype(np.float32)
            gray = np.mean(corrected_float, axis=2)

            shadows_mask = gray < 85
            midtones_mask = (gray >= 85) & (gray <= 170)
            highlights_mask = gray > 170
            bias_strength = strength * 0.3

            for zone_name, mask in [("shadows", shadows_mask), ("midtones", midtones_mask), ("highlights", highlights_mask)]:
                if zone_name in vae_color_bias and np.sum(mask) > 0:
                    bias = vae_color_bias[zone_name]
                    for c in range(3):
                        bias_correction = np.clip(bias[c] * bias_strength, -30, 30)
                        corrected_float[mask, c] -= bias_correction

            return _safe_clamp_colors(corrected_float, preserve_gradients=True)
        except Exception:
            return _match_to_reference_colors(processed_np, original_np, strength)

    def _histogram_matching_correction(self, original_np, processed_np, strength, vae_color_bias=None):
        if not ADVANCED_SKLEARN_AVAILABLE:
            return _match_to_reference_colors(processed_np, original_np, strength)
        try:
            corrected_np = processed_np.astype(np.float32)
            original_float = original_np.astype(np.float32)
            for c in range(3):
                matched_channel = exposure.match_histograms(corrected_np[:, :, c], original_float[:, :, c])
                corrected_np[:, :, c] = processed_np[:, :, c] * (1 - strength) + matched_channel * strength

            if vae_color_bias is not None:
                gray = np.mean(corrected_np, axis=2)
                bias_strength = strength * 0.2
                shadows_mask = gray < 85
                midtones_mask = (gray >= 85) & (gray <= 170)
                highlights_mask = gray > 170
                for zone_name, mask in [("shadows", shadows_mask), ("midtones", midtones_mask), ("highlights", highlights_mask)]:
                    if zone_name in vae_color_bias and np.sum(mask) > 0:
                        bias = vae_color_bias[zone_name]
                        for c in range(3):
                            bias_correction = np.clip(bias[c] * bias_strength, -30, 30)
                            corrected_np[mask, c] -= bias_correction

            return _safe_clamp_colors(corrected_np, preserve_gradients=True)
        except Exception:
            return _match_to_reference_colors(processed_np, original_np, strength)

    def _statistical_matching_correction(self, original_np, processed_np, strength, vae_color_bias=None):
        corrected_np = processed_np.astype(np.float32)
        original_float = original_np.astype(np.float32)

        for c in range(3):
            proc_mean = np.mean(corrected_np[:, :, c])
            proc_std = np.std(corrected_np[:, :, c])
            orig_mean = np.mean(original_float[:, :, c])
            orig_std = np.std(original_float[:, :, c])

            if proc_std > 0:
                normalized = (corrected_np[:, :, c] - proc_mean) / proc_std
                rescaled = normalized * orig_std + orig_mean
                corrected_np[:, :, c] = corrected_np[:, :, c] * (1 - strength) + rescaled * strength

        if vae_color_bias is not None:
            gray = np.mean(corrected_np, axis=2)
            bias_strength = strength * 0.25
            shadows_mask = gray < 85
            midtones_mask = (gray >= 85) & (gray <= 170)
            highlights_mask = gray > 170
            for zone_name, mask in [("shadows", shadows_mask), ("midtones", midtones_mask), ("highlights", highlights_mask)]:
                if zone_name in vae_color_bias and np.sum(mask) > 0:
                    bias = vae_color_bias[zone_name]
                    for c in range(3):
                        bias_correction = np.clip(bias[c] * bias_strength, -30, 30)
                        corrected_np[mask, c] -= bias_correction

        return _safe_clamp_colors(corrected_np, preserve_gradients=True)

    def _apply_mask_preservation(self, processed_img, corrected_img, mask, edge_feather, device):
        correction_mask = 1.0 - mask.to(device)
        if edge_feather > 0 and ADVANCED_CV2_AVAILABLE:
            correction_mask_np = correction_mask.cpu().numpy()
            correction_mask_np = cv2.GaussianBlur(
                correction_mask_np, (edge_feather * 2 + 1, edge_feather * 2 + 1), edge_feather / 3
            )
            correction_mask = torch.from_numpy(correction_mask_np).to(device)
        correction_mask = correction_mask.unsqueeze(-1)
        return processed_img * (1 - correction_mask) + corrected_img * correction_mask

    def _auto_preserve_inpainted(self, original_img, processed_img, corrected_img, edge_feather, device):
        diff = torch.abs(original_img - processed_img)
        diff_magnitude = torch.mean(diff, dim=-1)
        threshold = torch.quantile(diff_magnitude, 0.7)
        inpainted_mask = (diff_magnitude > threshold).float()
        correction_mask = 1.0 - inpainted_mask
        if edge_feather > 0 and ADVANCED_CV2_AVAILABLE:
            correction_mask_np = correction_mask.cpu().numpy()
            correction_mask_np = cv2.GaussianBlur(
                correction_mask_np, (edge_feather * 2 + 1, edge_feather * 2 + 1), edge_feather / 3
            )
            correction_mask = torch.from_numpy(correction_mask_np).to(device)
        correction_mask = correction_mask.unsqueeze(-1)
        return processed_img * (1 - correction_mask) + corrected_img * correction_mask


# -------------------------------------------------------------
# Auto Multiple Pad / Unpad / Freeze Frame / Audio / Switch Nodes
# -------------------------------------------------------------
class AIMZ_AutoMultiplePad:
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
                if current_mode == "reflect" and (pad_left >= w or pad_right >= w or pad_top >= h or pad_bottom >= h):
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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pad_start_frames": ("INT", {"default": 15, "min": 0, "max": 1000, "step": 1, "tooltip": "Number of freeze frames to prepend to the start"}),
                "pad_end_frames": ("INT", {"default": 15, "min": 0, "max": 1000, "step": 1, "tooltip": "Number of freeze frames to append to the end"}),
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
