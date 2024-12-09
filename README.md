This is the repository of the **"Optimal and Approximate Adaptive Stochastic Quantization"** (NeurIPS 2024), by Ran Ben Basat, Yaniv Ben-Itzhak, Michael Mitzenmacher, and Shay Vargaftik.

Our methods allow computing optimal and near optimal sets of quantization values for a given input vector, with the goal of stochatically quantizing it with minimal expected squared error.

The repository includes:

* A Visual Studio project (compiles successfully with Microsoft Visual Studio Community 2022 (64-bit) - Version 17.11.4). It includes a main.cpp file that runs all QUIVER variants (including the weighted ones).
* Python bindings that allow running the code directly from Python (to install, run `setup.py install' from the python_bindings directory).
* A script (speed_error_tests.py) that runs a complete experiment, varying the dimension, number of quantization values, distribution, etc. By default, it runs the exact QUIVER variants, and the approximate variants can run using `--type approx'.
* A script (plotQUIVER.py) that plots the results. It requires at least one of the variants (exact or approx) to be generated using `speed_error_tests.py' first.

To cite, please use:

```
@inproceedings{ben2024optimal,
  title={{Optimal and Approximate Adaptive Stochastic Quantization}},
  author={Ben Basat, Ran and Ben-Itzhak, Yaniv and Mitzenmacher, Michael and Vargaftik, Shay},
  booktitle={NeurIPS},
  year={2024}
}
```
