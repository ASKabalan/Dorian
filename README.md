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

> This repo is a fork of https://gitlab.mpcdf.mpg.de/fferlito/dorian 
> The goal of this repo is simplify the usage and make it usuable on raw lightcone arrays rather than read file from disk

Deflection Of Rays In Astrophysical Numerical simulations

Dorian is a Python package to compute full-sky ray-traced weak gravitational lensing maps starting from cosmological simulations.

For technical details, see the [related paper](https://arxiv.org/abs/2406.08540).

# Installation

The package can be installed with: 
```
pip install -e .
```

You will need the following dependencies:

- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [healpy](https://healpy.readthedocs.io/)
- [h5py](https://www.h5py.org/)
- [ducc0](https://gitlab.mpcdf.mpg.de/mtr/ducc)

# Usage

This fork simplifies the workflow significantly. You can now pass density maps (as numpy arrays) directly to the raytracer without needing to format files on disk.

```python
import numpy as np
from dorian.lensing import raytrace_from_density

# 1. Prepare your data
# density_maps: shape (n_shells, npix)
# redshifts: shape (n_shells,)
# ...

# 2. Run raytracing
results = raytrace_from_density(
    density_maps=density_maps,
    redshifts=redshifts,
    z_source=1.0,
    box_size=2000.0,
    n_particles=1024**3,
    omega_m=0.3,
    h=0.7,
    omega_l=0.7,
    nside=512,
    interp='bilinear'
)

kappa_map = results['convergence_raytraced']
```

For a complete working example, check `examples/raytracing_demo.ipynb`.

# Examples

- [examples/raytracing_demo.ipynb](examples/raytracing_demo.ipynb): The recommended starting point. Demonstrates loading a sample lightcone and running the raytracer.

# Authors

This package has been developed by [Fulvio Ferlito](https://gitlab.mpcdf.mpg.de/fferlito), with contributions from: Christopher Davies, Alessandro Greco, Martin Reinecke and Volker Springel.

# Contact

If you have any question, suggestion, or need help with the code, don't hesitate to contact the [author](mailto:fulvioferlito@gmail.com).

If you have questions regarding  this fork please create a github issue

# Citation

If you use this code for you work, please cite the [related paper](https://arxiv.org/abs/2406.08540).


