import numpy as np
import pickle
from interpolating_functions import evaluate_pflanz, evaluate_mit # Import the function
from interpolating_functions import plot_pflanz_interpolation, plot_MIT_interpolation # Import the function
import matplotlib.pyplot as plt

#### EXTRACTING INTERPLATOR DATA FROM PICKLE FILE
# Load the interpolators
with open("combined_data.pkl", "rb") as f:
    data = pickle.load(f)

################################################################################
######################## Define recovery parameters ############################
################################################################################

atm_density = 1.18  # Atmospheric density                               [kg/m^3] 
mass        = 25    # Mass of the body at recovery event                [kg]
v_ls        = 24    # Velocity of the body at the time of line stretch  [m/s]

g           = 9.81  # Gravitational Acceleration                        [m/s^2]  

CD0         = 0.8   # Nominal parachute drag coefficient                [-]
S0          = 12    # Nominal parachute surface area                    [m^2]
D0          = 4*np.sqrt(12/np.pi) # Parachute nominal diameter          [m]

Cx          = 1.4   # Opening shock coefficient at infinite mass        [-]

nfill       = 11.7  # Canopy fill constant                              [-]

################################################################################
############################### Computation ####################################
################################################################################

tf          = nfill*D0/v_ls     # Parachute inflation time              [s]

A_ballistic = 2*mass/(S0*CD0*atm_density*v_ls*tf) # Ballistic Parameter [-]

Rm          = atm_density*(S0*CD0)**(3/2)/mass     # Mass Ratio         [-]

# Following OSCalc Manual: Choose which value you prefer:
if Rm > 0.1:
    Drag_integral = 0.5 # Between 0.4 and 0.5
elif Rm > 0.01:
    Drag_integral = (0.5+0.2)/2
else:
    Drag_integral = 0.2

n_gen_fill = v_ls*tf*Drag_integral/np.sqrt(S0*CD0)

force_nominal = 0.5*atm_density*(v_ls**2)*S0*CD0
print(f'Steady State Force = {force_nominal:.2f} N')

# Values according to Pflanz
pflanz_sol = evaluate_pflanz(data,'0.5', A_ballistic)
fig1, ax1 = plt.subplots() 
plot_pflanz_interpolation(data, '0.5', A_ballistic, pflanz_sol, ax1)
print(f'Pflanz X1 = {pflanz_sol:.4f}')
print(f'Pflanz Ck = {pflanz_sol*Cx:.4f}')
print(f'Pflanz Force = {pflanz_sol*Cx*force_nominal:.2f} N')

# Values according to MIT
mit_sol = evaluate_mit(data, n_gen_fill, Rm)
fig2, ax2 = plt.subplots()
plot_MIT_interpolation(data, n_gen_fill, Rm, mit_sol, ax2)
plt.show()
print(f'MIT Ck = {mit_sol:.4f}')
print(f'MIT Force = {mit_sol*force_nominal:.2f} N')








