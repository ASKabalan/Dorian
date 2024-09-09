 ```
▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
▐                                                               ▌
▐   ██████████                       ███                        ▌
▐  ░░███░░░░███                     ░░░                         ▌
▐   ░███   ░░███  ██████  ████████  ████   ██████   ████████    ▌
▐   ░███    ░███ ███░░███░░███░░███░░███  ░░░░░███ ░░███░░███   ▌
▐   ░███    ░███░███ ░███ ░███ ░░░  ░███   ███████  ░███ ░███   ▌
▐   ░███    ███ ░███ ░███ ░███      ░███  ███░░███  ░███ ░███   ▌
▐   ██████████  ░░██████  █████     █████░░████████ ████ █████  ▌
▐  ░░░░░░░░░░    ░░░░░░  ░░░░░     ░░░░░  ░░░░░░░░ ░░░░ ░░░░░   ▌
▐                                                               ▌
▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
```

Deflection Of Rays In Astrophysical Numerical simulations

Dorian is a Python package to compute full-sky ray-traced weak gravitational lensing maps starting from cosmological simulations.

For technical details, see the [related paper](https://arxiv.org/abs/2406.08540).

# Installation

The package can be installed with: 
```
pip install dorian-astro
```

You will need the following dependencies:

- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [healpy](https://healpy.readthedocs.io/)
- [h5py](https://www.h5py.org/)
- [ducc0](https://gitlab.mpcdf.mpg.de/mtr/ducc)

# Usage

Using the code to perform a weak lensing simulation is as simple as calling the ```raytrace``` function. For instructions see the [example notebook](https://gitlab.mpcdf.mpg.de/fferlito/dorian/-/blob/main/examples/tutorial.ipynb) and the [template script](https://gitlab.mpcdf.mpg.de/fferlito/dorian/-/blob/main/examples/raytracing_example_script.py).

The code needs you to point to the ```simDir``` folder, where all the mass shells are arranged in the Gadget-4 format (based on hdf5) and directory structure, e.g.:

```
simDir
 |-mapsdir_001
 | |-maps_001.0.hdf5
 | |-maps_001.1.hdf5
 | `-maps_001.2.hdf5
 |-mapsdir_002
 | |-maps_002.0.hdf5
 | |-maps_002.1.hdf5
 | `-maps_002.2.hdf5
 `-mapsdir_003
   |-maps_003.0.hdf5
   |-maps_003.1.hdf5
   `-maps_003.2.hdf5
```

Where each ```mapsdir``` refers to a single mass-shell, and each mass-shell can be divided into multiple files (three in the example above). Note that the code supports mass-shells with variable thickness.

If your mass-shells are in another format (e.g. HEALPix maps stored into numpy arrays), you can use the ```write_massmap``` function to convert each of them in the compatible format.

# Examples

- [Notebook: performing a weak lensing simulation and computing the convergence power spectrum](https://gitlab.mpcdf.mpg.de/fferlito/dorian/-/blob/main/examples/tutorial.ipynb).
- [Script: performing a weak lensing simulation](https://gitlab.mpcdf.mpg.de/fferlito/dorian/-/blob/main/examples/raytracing_example_script.py).

# Authors

This package has been developed by [Fulvio Ferlito](https://gitlab.mpcdf.mpg.de/fferlito), with contributions from: Christopher Davies, Alessandro Greco, Martin Reinecke and Volker Springel.

# Contact

If you have any question, suggestion, or need help with the code, don't hesitate to contact the [author](mailto:fulvioferlito@gmail.com).

# Citation

If you use this code for you work, please cite the [related paper](https://arxiv.org/abs/2406.08540).


