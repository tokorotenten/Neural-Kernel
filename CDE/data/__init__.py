from typing import Dict, Any
from .synthetic import generate_bimodal, generate_skewed, generate_ring
from .uci import generate_uci


def generate_data(cfg):
    name = cfg.data.name
    if name == "bimodal":
        X, y, Xp, yp = generate_bimodal(cfg.data.num_seeds, cfg.data.N_x, cfg.data.N_y)
        return X, y, Xp, yp, None
    elif name == "skewed":
        X, y, Xp, yp = generate_skewed(cfg.data.num_seeds, cfg.data.N_x, cfg.data.N_y)
        return X, y, Xp, yp, None
    elif name == "ring":
        X, y, Xp, yp = generate_ring(cfg.data.num_seeds, cfg.data.N_x, cfg.data.N_y)
        return X, y, Xp, yp, None
    elif name == "uci":
        X, y, Xp, yp, UCI = generate_uci(cfg.data.num_seeds, cfg.data.data_path)
        return X, y, Xp, yp, UCI
    else:
        raise ValueError(f"Dataset {name} not known")