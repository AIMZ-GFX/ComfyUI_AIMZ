from .nodes import AIMZ_AutoMultiplePad, AIMZ_AutoMultipleUnpad

NODE_CLASS_MAPPINGS = {
    "AIMZ_AutoMultiplePad": AIMZ_AutoMultiplePad,
    "AIMZ_AutoMultipleUnpad": AIMZ_AutoMultipleUnpad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIMZ_AutoMultiplePad": "AIMZ Auto Multiple Pad (32x Safe)",
    "AIMZ_AutoMultipleUnpad": "AIMZ Auto Multiple Unpad",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
