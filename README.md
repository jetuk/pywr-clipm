# An OpenCL interior point method solver for Pywr.

This project contains an interior point method (IPM) solver for [Pywr](https://github.com/pywr/pywr).
The IPM is implemented in OpenCL to solve multiple linear programs simultaneously.

The current version does not implement all of the features of the standard Pywr solver algorithms (e.g
aggregated and virtual storage node constraints).
