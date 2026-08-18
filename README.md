# ComfyUI_AIMZ

A collection of high-utility custom nodes for ComfyUI by AIMZ / CEC Tools.

## Nodes Included

### 1. AIMZ Auto Multiple Pad (32x Safe)
Automatically pads images and masks to be divisible by any chosen multiple (default: **32** for MiniMax-H3 / DiT models).
* **None-Safe Pass-through:** If input image or mask is `None` (empty slot), it passes through `None` without crashing, enabling seamless dynamic multi-reference branching.
* **Flexible Pad Modes:** Supports `reflect`, `replicate`, `constant`, and `circular`.
* **Color Customization:** Supports named colors (`black`, `white`, `gray`), hex (`#ffffff`), and RGB tuples (`0,0,0`, `1,1,1`).
* **Mask Synchronization:** Accurately pads associated masks with identical offsets.

### 2. AIMZ Auto Multiple Unpad
Restores padded images or masks back to their original dimensions using `pad_info` or manual coordinate offsets.

---

## Installation

### Method 1: ComfyUI Manager (Git URL)
1. In ComfyUI, open **Manager** -> **Install via Git URL**.
2. Enter: `https://github.com/khwhite0413/ComfyUI_AIMZ`
3. Restart ComfyUI.

### Method 2: Manual Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/khwhite0413/ComfyUI_AIMZ.git
```

---

## License
MIT License
