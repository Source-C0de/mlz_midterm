"""Shared library for the Laboratory Quality Prediction System.

`src.data` holds the label definition and the feature matrix builder; both
`train.py` and `predict.py` import from here so that training and serving
always agree on the exact same inputs.
"""
