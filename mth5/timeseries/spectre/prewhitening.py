"""
    This module has methods for prewhiteing time series to reduce spectral leakage before FFT.
"""
import xarray as xr

from typing import Literal, Optional

def apply_prewhitening(
    prewhitening_type: Literal["first difference", ""],
    run_xrds_input: xr.Dataset,
) -> xr.Dataset:
    """
    Applies pre-whitening to time series to avoid spectral leakage when FFT is applied.

    Development Notes:
    - Consider passing the stft object, or a prewhitening object instead of a string literal as first argument.
    Parameters
    ----------
    prewhitening_type: Literal["first difference", ]
        Placeholder to allow for multiple methods of prewhitening.  Currently only
        "first difference" is supported.
    run_xrds_input : xr.Dataset
        Time series to be pre-whitened (can be multivariate).

    Returns
    -------
    run_xrds : xr..Dataset
        pre-whitened time series

    """
    if not prewhitening_type:
        msg = "No prewhitening specified - skipping this step"
        logger.info(msg)
        return run_xrds_input

    if prewhitening_type == "first difference":
        run_xrds = run_xrds_input.differentiate("time")
    else:
        msg = f"{prewhitening_type} pre-whitening not implemented"
        logger.exception(msg)
        raise NotImplementedError(msg)
    return run_xrds


def apply_recoloring(
    prewhitening_type: Literal["first difference", ],
    stft_obj: xr.Dataset,
) -> xr.Dataset:
    """
    Inverts the pre-whitening operation in frequency domain.  Modifies the input xarray in-place.

    Parameters
    ----------
    prewhitening_type: Literal["first difference", ]
        Placeholder to allow for multiple methods of prewhitening.  Currently only
        "first difference" is supported.
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients to be recoloured


    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Recolored time series of Fourier coefficients.
    """

    # No recoloring needed if prewhitening not appiled, or recoloring set to False
    if not prewhitening_type:
        msg = "No prewhitening was defined -- skipping recoloring"
        return stft_obj

    if prewhitening_type == "first difference":
        freqs = stft_obj.frequency.data
        jw = 1.0j * 2 * np.pi * freqs
        stft_obj /= jw

        # suppress nan and inf to mute later warnings
        if jw[0] == 0.0:
            cond = stft_obj.frequency != 0.0
            stft_obj = stft_obj.where(cond, complex(0.0))
    # elif prewhitening_type == "ARMA":
    #     from statsmodels.tsa.arima.model import ARIMA
    #     AR = 3 # TODO: add this to processing config
    #     MA = 4 # TODO: add this to processing config

    else:
        msg = f"{prewhitening_type} recoloring not yet implemented"
        logger.error(msg)
        raise NotImplementedError(msg)

    return stft_obj
