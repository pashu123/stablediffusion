vulkan_flags = {
    "device": "vulkan",
    "iree_opt_flags": [
        "--iree-flow-enable-conv-nchw-to-nhwc-transform",
        "--iree-flow-enable-padding-linalg-ops",
        "--iree-flow-linalg-ops-padding-size=32",
        "--iree-vulkan-target-triple=rdna2-unknown-linux",
    ],
    "save_vmfb": True,
    "load_vmfb": True,
}

cuda_flags = {
    "device": "cuda",
    "iree_opt_flags": [
        "--iree-flow-enable-conv-nchw-to-nhwc-transform",
        "--iree-flow-enable-padding-linalg-ops",
        "--iree-flow-linalg-ops-padding-size=32",
    ],
    "save_vmfb": True,
    "load_vmfb": True,
}
