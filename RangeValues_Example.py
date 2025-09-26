# In this example code, a range of values is computed following a Montecarlo approach. 
# No actual case by case simlation and for loops are needed, as this is a simple array 
# element-wise computation. This is only done for Pflanz but could easily be extended
# for MIT method aswell. As postprocessing example, the force on the 95th percentile 
# and the average opening shock are computed

import numpy as np
import pickle
from interpolating_functions import evaluate_pflanz
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
########################### Define parameter PDF ###############################
################################################################################

N           = 1000000    # Iterations per Degree of Freedom

# The main uncertainties for this example will be the:
#   - Nominal Drag Coefficient CD0: Equally likely to be between 0.7 and 0.8
#   - Opening force coefficient at Inifite mass Cx: Equal between 1 and 1.2
#   - Velocity at inflation: Normal distribution of average 24 m/s and sigma 5 m/s

# To check consistency and debugging, you can force Python to use the same 
# Pseudo-random number generator eveytime with this command:
# np.random.seed(42)

# Nominal Drag Coefficient CD0
CD0_vec = np.random.uniform(0.7,0.8,N)

# Opening force coefficient at infinite mass Cx
Cx_vec = np.random.uniform(1,1.2,N)

# Velocity
v_vec = np.random.normal(24,5,N)

################################################################################
############################### Computation ####################################
################################################################################

tf          = nfill*D0/v_vec     # Parachute inflation time              [s]

A_ballistic = 2*mass/(S0*CD0_vec*atm_density*v_vec*tf) # Ballistic Parameter [-]

# Equivalent Drag force at steady state, the peak is then adjusted by the Cx and X1 term
force_nominal = 0.5*atm_density*(v_vec**2)*S0*CD0_vec

# Values according to Pflanz
pflanz_sol = evaluate_pflanz(data,'0.5', A_ballistic,1)

# Opening Shock
pflanz_force = force_nominal*Cx_vec*pflanz_sol

################################################################################
################################# RESULTS ######################################
################################################################################

# Determine number of bins (choose one rule)
# num_bins = int(np.sqrt(len(pflanz_force)))  # Square Root Rule
# num_bins = int(np.log2(len(pflanz_force)) + 1)  # Sturges' Rule
num_bins = int(2 * len(pflanz_force)**(1/3))  # Rice Rule for large dataset

plt.figure(2)
plt.hist(pflanz_force, bins=num_bins, color="blue", alpha=0.7, edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Opening Shock")
plt.grid(True)
plt.show()

# Example post processing
# Calculate the 95th percentile

print('RESULTS')
print(f'Average Opening Shock: {np.average(pflanz_force):.1f} N')
print(f"With 95% confidence, the Opening shock is at most: {np.percentile(pflanz_force, 95):.1f} N")

