# Simulation-gas-in-cell
The script is simulation a confined cesium gas within a square glass cell. 

Read the PhD thesis Generation of non-classical states in a hybrid spin-optomechanical system for details.
https://nbi.ku.dk/english/theses/phd-theses/christian-folkersen-baerentsen/Christian-Baerentsen.pdf

#RUN the simulation
run run_simulation_multithreading.py to start the simulation. I recommend it to do it in an anaconda prompt if using spyder, as spyder doesn't support multicore use.

#Plot data
The data is saved as a pickle file
Use montecarlo_thermal_cell_plotting.py to extract the data.

Example
````
from montecarlo_thermal_cell_plotting import plotting 

h=plotting("12_04_2023_1")
freq, PSD = h.PSD() #returns the frequency and the power spectral density data
````
