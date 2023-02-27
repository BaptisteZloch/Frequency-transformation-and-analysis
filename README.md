# Frequency Transformation and Analysis
Repository aimed to gather my works on analyzing and processing a signal using frequency related tools over cryptos and stocks using several approaches.
This repo contains several notebooks:
- `01_Fourrier_smoothing.ipynb` : Created a low pass filter using Fast Fourrier Transformation (FFT) in order to denoise a signal. The new signal could be used as input in an ML model or to generate signals.
- `02_Fourrier_extrapolation.ipynb` : Used Fast Fourrier Transformation in output some value as "predictions" using an extrapolation of the fitted fourrier curve.
- `03_Trend_Decomposition.ipynb` : Generated sub signals from a raw signal in order to generate insight or trading indicators.
- `04_Wavelets_smoothing.ipynb` : Created a low pass filter using Discrete Wavelet Transformation (DWT) in order to denoise a signal. The new signal could be used as input in an ML model or to generate signals.
- `05_Wavelets_spectrum.ipynb` : -
- `06_EMD_smoothing.ipynb` : Implemented a filter using Empirical Mode Decomposition (EMD) in order to extract components from a signal.
- `07_Hurst_exponent_and_fractal.ipynb` : Implemented several Hurst exponent calculation to generate a fractal index for a time serie. The second part in about Multifractal Detrended Fluctuation Analysis (MFDFA) to analyze the complexity of a time serie.