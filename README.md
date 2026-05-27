# Milestone-8
## Successful demonstration of parametrization by non-invasive approach and cell tear-down.
Documentation of non-destructive cell parameterization and resulting model-based simulation results.

## Introduction

  This repository aims at using the tools developed in DigiBatt to extract equivalent circuit parameters from data generated in the project.\
  The resulting equivalent curcuit model is used to demonstrate module level simulations. 
## Successful demonstration of parametrization by non-invasive approach and cell tear-down.
Documentation of non-destructive cell parameterization and resulting model-based simulation results.

## Introduction

  This repository aims at using the tools developed in DigiBatt to extract equivalent circuit parameters from data generated in the project.\
  The resulting equivalent curcuit model is used to demonstrate module level simulations. 

### Tools
- bdat (https://digibatt.github.io/bdat/)
- pybop (https://pybop-docs.readthedocs.io/en/latest/index.html)
### Tools
- bdat (https://digibatt.github.io/bdat/)
- pybop (https://pybop-docs.readthedocs.io/en/latest/index.html)


### Repository structure
- `data/` bundles `DigiBatt-BAK-5000-N21700CG-006-GITT-data.parquet` and stores any generated parameters. 
- `utils.py`: 
  - `load_gitt()` will load the bundled GITT data in `DigiBatt-BAK-5000-N21700CG-006-GITT-data.csv` and return it as a `bdat.CyclingData` instance. 
- `fit.py` executes as a script and uses pybop to extract open-circuit voltage, resistances and timeconstants from the bundled GITT data. Parameters are extracted from each rest pulse and the adjacent current pulses. 
- `simulate` contains modules defining a `CellGrid`, `Module`, as well as a simple `DAESolver` and `Interpolant`. The models and solver are simple implementations in numpy/scipy. 
- `simulate.py`
    - `dcir()` runs a DCIR simulation and returns a convenient `SimulationResults` object.
    - `cycle` runs a simple cycling simulation and returns a convenient `SimulationResults` object.

### Simulation models
- `CellGrid` is a simple equivalent circuit model with 4 states; *SOC*, *Pol-1*, *Pol-2* and *Hysteresis*. Circuit elements $OCV$, $R_0$, $R_1$, $R_2$, $\tau_1$, and $\tau_2$ are obtained through the parametrization of the GITT. The circuit elements can be modelled using a lookup table,  a spline, or the average value. The `CellGrid` is vectorized to $n_\mathrm{series}$, $n_\mathrm{parallel}$ cells for plugging into a modulec model. 
- `Module` handles the current- and potential distribution in the bussbar, and uses to previously mentioned `CellGrid` model to handle the battery cell dynamics.  

- `CellGrid` is a simple equivalent circuit model with 4 states; *SOC*, *Pol-1*, *Pol-2* and *Hysteresis*. Circuit elements $OCV$, $R_0$, $R_1$, $R_2$, $\tau_1$, and $\tau_2$ are obtained through the parametrization of the GITT. The circuit elements can be modelled using a lookup table,  a spline, or the average value. The `CellGrid` is vectorized to $n_\mathrm{series}$, $n_\mathrm{parallel}$ cells for plugging into a modulec model. 
- `Module` handles the current- and potential distribution in the bussbar, and uses to previously mentioned `CellGrid` model to handle the battery cell dynamics.  

## Further work

### Physics-motivated equivalent circuit models

#### Electrode level open circuit potential
$$
OCV\left(SOC, \theta\right) = OCP^{PE}\left(SOC, \theta_{1}^{PE}, \theta_{0}^{PE}\right) - OCP^{NE}\left(SOC, \theta_{1}^{NE}, \theta_{0}^{NE}\right)
$$

where, for composite electrodes

$$
I_{tot} = \sum_{i=1}^{n} J_{i} \\
U_{1} = U_{2} \\
\vdots \\
U_{n-1} = U_{n}  
$$

$^\ast$ Should we enforce the electrode voltage or potential?

#### Charge transfer resistance
Somthing like

$$
R_{ct} \propto \frac{1}{STO^{\alpha}\left(1-STO\right)^{1-\alpha}} \\
\tau_{ct} \approx C_{cl} \times R_{ct}
$$

#### Diffusion resistance
Someting like 

$$
\tau_{D} \approx \frac{l^{2}_{D}}{D}
$$

where I guess the resistance is time- and current dependent. 
