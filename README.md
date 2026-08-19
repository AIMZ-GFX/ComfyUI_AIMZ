# ComfyUI_AIMZ

**ComfyUI_AIMZ** is a versatile suite of essential workflow helpers, quality-of-life improvements, and utility custom nodes for ComfyUI, crafted by **AIMZ / CEC Tools**.

---

## 🛠️ Nodes Included

### 1. AIMZ Auto Multiple Pad
Automatically pads images, masks, or video frame batches to be divisible by any chosen multiple (default: **32**) with zero quality loss.
* **None-Safe Pass-through:** If input image/video or mask is `None` (empty branch/slot), it passes through `None` and `0` without raising exceptions, enabling seamless dynamic conditional branching.
* **Dimensions & Frame Count:** Also outputs `width`, `height`, and `count` (frame count), replacing `GetImageSizeAndCount` with full None-safety!
* **Default Settings:** Default pad mode is `constant` with `white` background.
* **Flexible Pad Modes:** Supports `constant`, `reflect`, `replicate`, and `circular`.
* **Color Customization:** Supports named colors (`white`, `black`, `gray`), hex (`#ffffff`), and normalized RGB tuples (`1,1,1`, `0,0,0`).
* **Mask Synchronization:** Accurately pads associated masks with identical offsets.

### 2. AIMZ Freeze Frame Pad
Pads video frame sequences with repeated freeze frames at the beginning and/or end (e.g. 15 frames for MiniMax-H3 motion buffer).
* **Replaces 5 Nodes:** Replaces `GetImageFromBatch` (start), `RepeatImages` (start), `GetImageFromBatch` (end), `RepeatImages` (end), and `ImageBatchMulti` into a single, clean node.
* **None-Safe:** If video is `None`, passes through `None` and `count=0` cleanly without errors.

### 3. AIMZ Auto Multiple Unpad
Restores padded images or masks back to their original input dimensions using `pad_info` or manual coordinate offsets.

### 4. AIMZ Preview Image (None Safe)
A bulletproof replacement for ComfyUI's standard `PreviewImage` node.
* **None-Safe:** If the input image is `None` or an empty branch, it skips cleanly **without throwing `TypeError: NoneType` errors**.
* **Pass-Through Output:** Also outputs the image so you can chain it cleanly in your workflows.

---

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open **ComfyUI Manager** -> **Custom Nodes Manager**.
2. Search for **`comfyui_aimz`** or **`AIMZ`**.
3. Click **Install** and restart ComfyUI.

### Method 2: Install via Git URL
1. In ComfyUI Manager, select **Install via Git URL**.
2. Enter: `https://github.com/AIMZ-GFX/ComfyUI_AIMZ`
3. Restart ComfyUI.

### Method 3: Manual Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AIMZ-GFX/ComfyUI_AIMZ.git
```

---

## 📄 License
MIT License
