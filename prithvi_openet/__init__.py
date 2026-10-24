"""Prithvi-OpenET and Prithvi-OpenET-Tower: inference.

    from prithvi_openet import load_model, example_inputs, predict
    model, config = load_model("weights/prithvi_openet_opt_f_e")
    out = predict(model, example_inputs(np.load("examples/conus_T13TGE_2020-07.npz"), config))
    out["et_mean"]      # (224, 224) ET in mm month-1, five-head mean
"""
from .inputs import HLS_BANDS, example_inputs, normalize, predict
from .model import MEMBERS, PrithviOpenET, load_model

__all__ = ["HLS_BANDS", "MEMBERS", "PrithviOpenET", "example_inputs", "load_model", "normalize", "predict"]
