# Milestone-8: Successful demonstration of parametrization by non-invasive approach and cell tear-down. 
## Documentation of non-destructive cell parameterization and resulting model-based simulation results.

This repository aims at leveraging the tools developed in DigiBatt to extract equivalent circuit parameters from data generated in the project.\
The resulting equivalent curcuit model is used to demonstrate module level simulations.

- Timeseries measurements from Galvanostatic Intermittent Titration Technique (GITT) performed by DLR is included in `/data/DigiBatt-BAK-5000-N21700CG-006-GITT-data.parquet`.
- bdat (https://digibatt.github.io/bdat/) was used to wrap the the provided GITT measurements and segment the data into steps.
- PyBaMM (https://pybamm.org/) was used to to define an Equivalent Circuit Model (ECM) for use in the paramerter extraction process.   
- PyBOP (https://pybop-docs.readthedocs.io/en/latest/index.html) was used used to optimize the ECM parameters in the PyBaMM model. 

### Repository structure
- `/data/` includes timeseries GITT measurements `DigiBatt-BAK-5000-N21700CG-006-GITT-data.parquet` and ECM parameters extracted from the timeseries data. 
- `utils.py`: 
  - `load_gitt()` loads the bundled GITT data in `DigiBatt-BAK-5000-N21700CG-006-GITT-data.csv` and return it as a `bdat.CyclingData` instance. 
- `fit.py` executes as a script and uses PyBOP to extract Open-circuit voltage (OCV), ohmic resistance $\left(R_0\right)$ and R-C parameters ($R_i$ and $\tau_i$ where $i=\left[1,2,3...\right]$) from the bundled GITT data. 
- `/simulate/` contains modules defining a `CellGrid`, `Module`, as well as a simple `DAESolver` and `Interpolant`. The models and solver are simple implementations in numpy/scipy that uses the extracted equivalent circuit parameters to define a module level model. 
- `simulate.py`
    - `dcir()` runs a DCIR simulation and returns a convenient `SimulationResults` object.
    - `cycle()` runs a simple cycling simulation and returns a convenient `SimulationResults` object.
- `/figures/` stores any figures generated.  
- The notebooks `sim_cycle` and `sim_dcir` demonstrates some simulations and graphs.

### Simulation models
`CellGrid` is a simple equivalent circuit model with $2+n$ states; *SOC*, *Hyst* and $n$ $R-C$ polarization branches.
Circuit elements $OCV$, $R_0$, $R_i$ and $\tau_i$ $\left(i=\left[1,2,...,n\right]\right)$ were obtained through the parametrization of the GITT pulses. The circuit elements can be modelled using a lookup table,  a spline, or the average value. The `CellGrid` is vectorized to $n_\mathrm{series}$, $n_\mathrm{parallel}$ cells for plugging into a the module model.\
`Module` Provides the expressions to simulate the current- and potential distribution in a bussbar, where the `CellGrid` contribute to the voltage difference between potential nodes in the main current path.
`DAESolver` solves the simulation problem on the form
  $$\frac{\partial x}{\partial t} = f\left(x,z,u\right)$$
  $$0 = g\left(x,z,u\right)$$
  $$x_{k} = x_{k-1} + \Delta t \cdot f\left(x_{k-1},z_{k-1},u_{k-1}\right)$$
  $$z_{k} \rightarrow 0 = g\left(x_{k}, z_{k}, u_{k}\right)$$

where $x$ is a matrix of the cell states, $z$ is a vector of currents and potentials in the bussbar and $u$ is the module current.  
The model is set up to accept a mean and standard deviation of parameters such as cell SOH, cell SOR, bussbar series resistance, *etc.*

### Comment on results
With constraints to the timeconstants, $\tau_i<\tau_{i+1}$, `SciPyMinimize` was the only available optimizer. \
The timeconstants and resistances would grow to *very* large values if unconstrained in the range $\approx 0\%-30\%~SOC$.
 
## Prospects for further work
### Electrode level ECM
Extract the individual electrode ECM parameters to investigate the source of the aformentioned large resistances and timeconstants
  
#### Electrode level open circuit potential
Separating the OCV model into electrode-level open-circuit potential (OCP) for each electrode allows one to also estimate the individual electrode capacities. 

$$
OCV\left(SOC, \theta\right) = OCP^{PE}\left(SOC, \theta_{1}^{PE}, \theta_{0}^{PE}\right) - OCP^{NE}\left(SOC, \theta_{1}^{NE}, \theta_{0}^{NE}\right)
$$

which introduces a set of nonlinear equations for composite electrodes (*i.e.* silicon-graphite). 

$$
I_{tot} = \sum_{i=1}^{n} J_{i} \\
U_{1} = U_{2} \\
\vdots \\
U_{n-1} = U_{n}  
$$

### State- and parameter esimation
With an initial model for $R_0$, $R_1$, *etc.* at beginning of life (BOL), the resistance models can be expressed more generically as 

$$
R_i(SOC,\dots) = R_i^{BOL}(SOC,\dots) \cdot SOR
$$

where SOR is the average resistance growth, analogous to SOH for capacity decline. With an electrode-level OCP model, the total parameter-set for esimtation is then given as $p=\left[SOH, SOR, \theta_{1}^{NE}, \theta_{0}^{NE}, \theta_{1}^{PE}, \theta_{0}^{PE}\right]$ alongside the states $x=\left[SOC, H, U_1,\dots,U_n\right]$. With the states and parameters changing on different timescales, their estimation must also be separated, using *e.g.* a Rao-Blackwellised particle filter. 

