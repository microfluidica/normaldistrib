import numpy as np
from scipy.special import erfc

__version__ = "1.0.0"


def erfv(x, slope=20, scale=1, sep=5, center=0):
    x = np.asanyarray(x)
    return scale * (
        0.5 * erfc(slope * (x - center + sep)) + 0.5 * erfc(slope * (-x + sep + center))
    )


def erfvd(x, slope=20, scalea=1, scaleb=1, sep=5, center=0):
    x = np.asanyarray(x)
    return scalea * 0.5 * erfc(slope * (x - center + sep)) + scaleb * 0.5 * erfc(
        slope * (-x + sep + center)
    )


def gaussian(x, center, tau, scale):
    x = np.asanyarray(x)
    return (
        scale
        * (np.exp(-((x - center) ** 2) / 2 / tau / tau))
        / np.sqrt(2 * np.pi)
        / tau
    )


def erfv_stepr(x, slope=20, scale=1, sep=5, center=0):
    x = np.asanyarray(x)
    return scale * (0.5 * erfc(slope * (-x + sep + center)))


def erfv_stepl(x, slope=20, scale=1, sep=5, center=0):
    x = np.asanyarray(x)
    return scale * (0.5 * erfc(slope * (x - center + sep)))


def mserf(x, slope=20, scale=1, sep=5, center=0, halfwidth=60):
    return (
        erfv_stepr(x, slope, scale, sep, center - (halfwidth + 1))
        * erfv_stepl(x, slope, scale, sep, center + (halfwidth + 1))
    ) ** 0.5
