import os

import torch
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from shark_utils.opt_flags import vulkan_flags, cuda_flags


def _compile_module(shark_module, model_name, extra_args=[]):
    if cuda_flags["load_vmfb"] or cuda_flags["save_vmfb"]:
        device = (
            cuda_flags["device"]
            if "://" not in cuda_flags["device"]
            else "-".join(cuda_flags["device"].split("://"))
        )
        extended_name = "{}_{}".format(model_name, device)
        vmfb_path = os.path.join(os.getcwd(), extended_name + ".vmfb")
        if (
            cuda_flags["load_vmfb"]
            and os.path.isfile(vmfb_path)
            and not cuda_flags["save_vmfb"]
        ):
            print(f"loading existing vmfb from: {vmfb_path}")
            shark_module.load_module(vmfb_path, extra_args=extra_args)
        else:
            if cuda_flags["load_vmfb"]:
                print("Saving to {}".format(vmfb_path))
            else:
                print(
                    "No vmfb found. Compiling and saving to {}".format(
                        vmfb_path
                    )
                )
            path = shark_module.save_module(
                os.getcwd(), extended_name, extra_args
            )
            shark_module.load_module(path, extra_args=extra_args)
    else:
        shark_module.compile(extra_args)
    return shark_module


# Downloads the model from shark_tank and returns the shark_module.
def get_shark_model(tank_url, model_name, extra_args=[]):
    from shark.shark_downloader import download_torch_model

    mlir_model, func_name, inputs, golden_out = download_torch_model(
        model_name, tank_url=tank_url
    )
    shark_module = SharkInference(
        mlir_model,
        func_name,
        device=cuda_flags["iree_opt_flags"],
        mlir_dialect="linalg",
    )
    return _compile_module(shark_module, model_name, extra_args)


# Converts the torch-module into a shark_module.
def compile_through_fx(
    model, inputs, model_name, extra_args=cuda_flags["iree_opt_flags"]
):

    mlir_module, func_name = import_with_fx(model, inputs)

    shark_module = SharkInference(
        mlir_module,
        func_name,
        device=cuda_flags["device"],
        mlir_dialect="linalg",
    )

    return _compile_module(shark_module, model_name, extra_args)
