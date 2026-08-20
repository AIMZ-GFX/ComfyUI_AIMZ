from .nodes import (
    AIMZ_AutoMultiplePad,
    AIMZ_AutoMultipleUnpad,
    AIMZ_FreezeFramePad,
    AIMZ_PreviewImageNoneSafe,
    AIMZ_SelectiveGroupBypasser,
)

NODE_CLASS_MAPPINGS = {
    "AIMZ_AutoMultiplePad": AIMZ_AutoMultiplePad,
    "AIMZ_AutoMultipleUnpad": AIMZ_AutoMultipleUnpad,
    "AIMZ_FreezeFramePad": AIMZ_FreezeFramePad,
    "AIMZ_PreviewImageNoneSafe": AIMZ_PreviewImageNoneSafe,
    "AIMZ_SelectiveGroupBypasser": AIMZ_SelectiveGroupBypasser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIMZ_AutoMultiplePad": "AIMZ Auto Multiple Pad",
    "AIMZ_AutoMultipleUnpad": "AIMZ Auto Multiple Unpad",
    "AIMZ_FreezeFramePad": "AIMZ Freeze Frame Pad",
    "AIMZ_PreviewImageNoneSafe": "AIMZ Preview Image (None Safe)",
    "AIMZ_SelectiveGroupBypasser": "AIMZ Selective Group Bypasser",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
