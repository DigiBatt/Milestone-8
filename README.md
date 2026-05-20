# Milestone-8
This repositiry aims at using the tools developed in DigiBatt to parametrize and equivalent circuit model, and using the resulting parameters in a module level simulation. 

## DigiBatt tools used:
- bdat
- pybop


## Structure
- `utils.py`: Miscelaneous utility functions. `utils.load_gitt()` will load the bundled GITT data in `DigiBatt-BAK-5000-N21700CG-006-GITT-data.csv` and return it as a `bdat.CyclingData` instance. 
- `fit.py` executes as a script and uses pybop to fit open-circuit voltage and timesconstants for each relaxation pulse in bundled GITT data.   
- `simulate.py` will run simulations of a single cell and full distributed model using the parameters obtained in `fit.py`. Note that this is written as a simple example using numpy and scipy, not pybamm.  
- `data/` bundles `DigiBatt-BAK-5000-N21700CG-006-GITT-data.csv`
- *.json is ignored by git.

### Simulation models
- `CellGrid` is a simple equivalent circuit model with 4 states; *SOC*, *Pol-1*, *Pol-2* and *Hysteresis*. Circuit elements $OCV$, $R_0$, $R_1$, $R_2$, $\tau_1$, and $\tau_2$ are obtained through the parametrization of the GITT data and is consume *via* a lookup table, spline or average value. The model is scaled to $n_\mathrm{series}$, $n_\mathrm{parallel}$ for plugging into a module. 
- `Module` 

## Futher work
### ... toss this. 
The open circuit potential of an electrode material is a thermodynamic property, where the potential of

$$
x\mathrm{Li}^{+} + xe^{-} + \mathrm{H} \rightleftharpoons Li_{x}H
$$

is measured versus $\mathrm{Li}^{+}/\mathrm{Li}^{0}$. 

One can then use available literature data to generate *prior* SOC-OCP curves for graphite-silicon blends of different mass fractions. Repeating the exercise for different cathode materials, a final database of prior SOC-OCV curves can me bade for a range of anode-cathode combinations. This can be used in some machine learnign algorithm to quickly determine the electrode materials and balancing in a cell. 


## AI Disclaimer
- Copilot was used to figure out which spline in `scipy.interpolate` was the most suitable.
- Copilot was used to add logging statements in `simulate.py` with the prompt `Add debugging log statements to make finalizing simulate.py easier`. 