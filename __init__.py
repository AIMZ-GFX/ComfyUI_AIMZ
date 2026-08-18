from .nodes import AIMZ_AutoMultiplePad, AIMZ_AutoMultipleUnpad, AIMZ_PreviewImageNoneSafe

NODE_CLASS_MAPPINGS = {
    "AIMZ_AutoMultiplePad": AIMZ_AutoMultiplePad,
    "AIMZ_AutoMultipleUnpad": AIMZ_AutoMultipleUnpad,
    "AIMZ_PreviewImageNoneSafe": AIMZ_PreviewImageNoneSafe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIMZ_AutoMultiplePad": "AIMZ Auto Multiple Pad",
    "AIMZ_AutoMultipleUnpad": "AIMZ Auto Multiple Unpad",
    "AIMZ_PreviewImageNoneSafe": "AIMZ Preview Image (None Safe)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
