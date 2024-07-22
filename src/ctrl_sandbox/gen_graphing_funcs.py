import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Tuple

import ctrl_sandbox.gen_typing as gt


def set_lim_based_on_defined_lim(
    x_lim: Tuple[float, float], x: gt.npArr64, y: gt.npArr64
) -> Tuple[np.float64, np.float64]:
    mask = (x >= x_lim[0]) & (x <= x_lim[1])
    return (np.min(y[mask]), np.max(y[mask]))
