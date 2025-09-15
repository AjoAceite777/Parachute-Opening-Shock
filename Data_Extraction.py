###############################################################################
# THIS CODE IS FOR DEBUGGING PURPOSES ONLY AND SHOULD NOT BE NECESSARY TO RUN #
###############################################################################

# This script extracts the raw data from the txt files outputed from Webplotdigitizer 
# and finds the coefficients and functions for the different curves in Pflanz Method
# and Moment Impulse Theorem method from OSCalc. The desired values are already stored in the 
# pickle file "combined_data.pkl"

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle

from scipy.interpolate import PchipInterpolator # For the Pflanz data

####################################################
########## MOMENT IMPULSE THEOREM SECTION ##########
####################################################

# Get the current working directory
current_directory = os.getcwd()  

# Path to the file relative to the current directory
relative_path = "M.I.T Data"

# Combine the current directory with the relative path (optional but useful for clarity)
folder_path = os.path.join(current_directory, relative_path)

# Interpolating polynomial order for different datasets
interp_order = np.array([3,5])

# Placeholder for all coefficients from curves
coeffs_dict = {}

# Filenames of the data
file_list = ["ngenfill_lower_raw.txt","ngenfill_upper_raw.txt"]

# Different plot titles
plot_title = ["Opening Shock Factor 1 <= n_gen <= 4","Opening Shock Factor n_gen >= 4"]

# Dictionary names for the coefficients
dict_name = ['lower','upper']

for i, file_name in enumerate(file_list):

    # Specify the file path
    file_path = os.path.join(folder_path, file_name)

    # Load the data
    data_array = np.loadtxt(file_path)  # Delimiter is space

    # Raw data was extracted with webplotdigitizer and did not have enough float resolution. So x-values need to be scaled back down 
    data_array[:,0] /= 10000

    log_col = np.log10(data_array[:,0])

    # Find coefficients for the interpolating polynomial
    poly_coeffs = np.polyfit(log_col,data_array[:,1],interp_order[i])

    # Save interpolating polynomial coefficients in a dictionary list
    coeffs_dict[dict_name[i]] = poly_coeffs.tolist()

    # Create the interpolating function
    polynomial = np.poly1d(poly_coeffs)

    # Create x-axis values for the plotting
    log_x=np.linspace(-4,1,100)
    
    # Visualize the data and the fitted polynomial
    plt.figure(i)
    plt.scatter(data_array[:, 0], data_array[:, 1], label="Raw Data", color="blue", alpha=0.6)
    plt.plot(10**log_x, polynomial(log_x), label="Data", color="red", linewidth=2)
    
    plt.gca().set_xscale("log") 
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    plt.xlim(0.0001,10)
    plt.ylim(0,2)
    
    plt.xlabel("Mass Ratio - Rm")
    plt.ylabel("Opening Shock Factor - Ck")
    plt.title(plot_title[i])
    plt.legend()
    

####################################################
################## PFLANZ SECTION ##################
####################################################

# Path to the file relative to the current directory
relative_path = "Pflanz Data"

# Combine the current directory with the relative path
folder_path = os.path.join(current_directory, relative_path)

# Data - X1 Force Reduction Factor from Pflanz Method
curve_cases = np.array([5,1,2])
# Note: case 5 corresponds to Curve with n = 0.5

# Generate x-space for the Pflanz Graph
PflanzData_log_x = np.linspace(np.log10(0.01),np.log10(1000),200)

# Formatting for the final plot
curve_style = ['dashed','solid','dashdot']
curve_label = ['n= 1/2','n = 1','n = 2']

# Placeholder for all interpolators from curves
pflanz_dict = {}
dict_name = ['0.5','1','2']

# Iterate through all curve data txt files
for i in range(0,3):
    
    # Precerate a nx2 array to vertically concatenate Curve I and Curve II
    entire_curve = np.empty((0,2))
    
    for j in range(1,3):
        
        # Generate Pflanz file name
        file_name = f'X1_Curve_{j}_N{curve_cases[i]}.txt'
                
        # Specify the file path
        file_path = os.path.join(folder_path, file_name)

        # Load the data
        data_array = np.loadtxt(file_path)  # Delimiter is space
        
        if j > 1: # Curve number 2 needs y axis rescale and add a final value for correct interpolation
            data_array[:,0] *= 100 # Scale the y axis 
            data_array = np.vstack([data_array,[1000,1]]) 
            
        # Now that the points from both curves has been extracted, we must stack them to create the entire curve
        entire_curve = np.vstack([entire_curve, data_array])
                
    # Now that Curve I and Curve II data are joined, you can create the interpolating function
    pchip_func = PchipInterpolator(np.log10(entire_curve[:,0]),np.log10(entire_curve[:,1]))
    PflanzData_log_y = pchip_func(PflanzData_log_x)
    
    # Save interpolating polynomial coefficients in a dictionary list
    pflanz_dict[dict_name[i]] = pchip_func
    
    plt.figure(3)
    plt.plot(10**PflanzData_log_x, 10**PflanzData_log_y, label=curve_label[i], color="red", linewidth=2, linestyle=curve_style[i])
    
plt.gca().set_xscale("log") 
plt.gca().set_yscale("log") 
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.xlim(0.01,1000)
plt.ylim(0.01,1)

plt.xlabel("Ballistic Coefficient A [-]")
plt.ylabel("Opening Force Reduction Factor - X1")
plt.title("Pflanz Interpolated Data")
plt.legend()

plt.show()


# Combine coefficients and the interpolator into a dictionary
export_data = {
    "MIT_coeffs": coeffs_dict,
    "Pflanz_functions": pflanz_dict,
}

# Save the combined data to a pickle file
with open("combined_data.pkl", "wb") as f:
    pickle.dump(export_data, f)

print("Data saved successfully.")

        
