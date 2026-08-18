# ComfyUI_AIMZ

**ComfyUI_AIMZ** is a versatile suite of essential workflow helpers, quality-of-life improvements, and utility custom nodes for ComfyUI, crafted by **AIMZ / CEC Tools**.

---

## 🛠️ Nodes Included

### 1. AIMZ Auto Multiple Pad (32x Safe)
Automatically pads images and masks to be divisible by any chosen multiple (e.g. 8, 16, 32, 64) with zero quality loss.
* **None-Safe Pass-through:** If input image or mask is `None` (empty branch/slot), it passes through `None` without raising exceptions, enabling seamless dynamic conditional branching.
* **Flexible Pad Modes:** Supports `reflect` (recommended for seamless AI generation), `replicate`, `constant`, and `circular`.
* **Color Customization:** Supports named colors (`black`, `white`, `gray`), hex (`#ffffff`), and normalized RGB tuples (`0,0,0`, `1,1,1`).
* **Mask Synchronization:** Accurately pads associated masks with identical offsets.

### 2. AIMZ Auto Multiple Unpad
Restores padded images or masks back to their original input dimensions using `pad_info` or manual coordinate offsets.

*(More utility nodes will be added continuously to this suite!)*

---

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open **ComfyUI Manager** -> **Custom Nodes Manager**.
2. Search for **`comfyui-aimz`** or **`AIMZ`**.
3. Click **Install** and restart ComfyUI.

### Method 2: Install via Git URL
1. In ComfyUI Manager, select **Install via Git URL**.
2. Enter: `https://github.com/khwhite0413/ComfyUI_AIMZ`
3. Restart ComfyUI.

### Method 3: Manual Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/khwhite0413/ComfyUI_AIMZ.git
```

---

## 📄 License
MIT License
