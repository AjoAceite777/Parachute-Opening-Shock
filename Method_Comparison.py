import numpy as np
import pickle
from interpolating_functions import evaluate_pflanz
from interpolating_functions import plot_pflanz_interpolation
from interpolating_functions import evaluate_mit
from interpolating_functions import plot_MIT_interpolation
import matplotlib.pyplot as plt

#### EXTRACTING INTERPLATOR DATA FROM PICKLE FILE
# Load the interpolators
with open("combined_data.pkl", "rb") as f:
    data = pickle.load(f)

################################################################################
######################## COMPARISON: Cx variation ############################
################################################################################

CD0         = 0.8   # Nominal parachute drag coefficient                [-]

Cx_vec          = [1, 1.2, 1.4, 1.6, 1.8]   # Opening shock coefficient at infinite mass        [-]

nfill_vec       = [4, 8, 10, 14] # Canopy fill constant              [-]

# Computation

# Entire domain for Pflanz Method
A_ballistic = 10**np.linspace(np.log10(0.01),np.log10(1000),200) # Ballistic Parameter [-]

pflanz_05_sol = evaluate_pflanz(data, "0.5", A_ballistic)
pflanz_1_sol = evaluate_pflanz(data, "1", A_ballistic)
pflanz_2_sol = evaluate_pflanz(data, "2", A_ballistic)

fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)

# Flatten axes into 1D array for easy looping
axes = axes.ravel()

# Store line handles for global legend
handles, labels = None, None

for i, Cx in enumerate(Cx_vec):
    
    ax1 = axes[i]
    ax1.plot(A_ballistic, pflanz_05_sol*Cx, label="pflanz n = 0.5", color="black", linewidth=2, linestyle="dashed")
    ax1.plot(A_ballistic, pflanz_1_sol*Cx, label="pflanz n = 1", color="black", linewidth=2, linestyle="solid")
    ax1.plot(A_ballistic, pflanz_2_sol*Cx, label="pflanz n = 2", color="black", linewidth=2, linestyle='dashdot')

    for i, nfill in enumerate(nfill_vec):
        
        Rm = 1/A_ballistic * np.sqrt(np.pi*CD0)/nfill # Mass Ratio         [-]

        Drag_integral = np.where(
            Rm > 0.1, 
            0.5,                           # Case 1
            np.where(
                Rm > 0.01, 
                (0.5+0.2)/2,               # Case 2
                0.2                        # Case 3
            )
        )

        n_gen_fill = nfill * 2 * Drag_integral / np.sqrt(np.pi * CD0)

        MIT_sol = evaluate_mit(data, n_gen_fill, Rm)

        mit_label = f"nfill = {nfill}"

        ax1.plot(A_ballistic, MIT_sol, label=mit_label, linewidth=2)
        
    if handles is None:  # only capture on the first subplot
        handles, labels = ax1.get_legend_handles_labels()


    ax1.set_xscale("log") 
    # ax1.set_yscale("log") 
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax1.set_xlim(min(A_ballistic),max(A_ballistic))
    ax1.set_ylim(0,2)

    ax1.set_xlabel("Ballistic Coefficient A [-]")
    ax1.set_ylabel("Ck")
    ax1.set_title(f"Cx = {Cx}; CD0 = {CD0}")
    #plt.figtext(0.70, 0.2, f'Cx = {Cx}\nCD0 = {CD0}', bbox=dict(facecolor='white', edgecolor='black', pad=10.0))

# Use the last (6th) subplot for the legend
axes[-1].axis("off")  # hide axes
axes[-1].legend(handles, labels, loc="center", fontsize=10)

fig.suptitle("Comparison for different Cx values", fontsize=14)
plt.tight_layout()

################################################################################
######################## COMPARISON: CD0 variation ############################
################################################################################

CD0_vec         = [0.4, 0.6, 0.8, 1, 1.2]   # Nominal parachute drag coefficient                [-]

Cx          = 1.4   # Opening shock coefficient at infinite mass        [-]

nfill_vec       = [4, 8, 10, 14] # Canopy fill constant              [-]

# Computation

# Entire domain for Pflanz Method
A_ballistic = 10**np.linspace(np.log10(0.01),np.log10(1000),200) # Ballistic Parameter [-]

pflanz_05_sol = evaluate_pflanz(data, "0.5", A_ballistic)
pflanz_1_sol = evaluate_pflanz(data, "1", A_ballistic)
pflanz_2_sol = evaluate_pflanz(data, "2", A_ballistic)

fig2, axes2 = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)

# Flatten axes into 1D array for easy looping
axes2 = axes2.ravel()

# Store line handles for global legend
handles, labels = None, None

for i, CD0 in enumerate(CD0_vec):
    
    ax1 = axes2[i]
    ax1.plot(A_ballistic, pflanz_05_sol*Cx, label="pflanz n = 0.5", color="black", linewidth=2, linestyle="dashed")
    ax1.plot(A_ballistic, pflanz_1_sol*Cx, label="pflanz n = 1", color="black", linewidth=2, linestyle="solid")
    ax1.plot(A_ballistic, pflanz_2_sol*Cx, label="pflanz n = 2", color="black", linewidth=2, linestyle='dashdot')

    for i, nfill in enumerate(nfill_vec):
        
        Rm = 1/A_ballistic * np.sqrt(np.pi*CD0)/nfill # Mass Ratio         [-]

        Drag_integral = np.where(
            Rm > 0.1, 
            0.5,                           # Case 1
            np.where(
                Rm > 0.01, 
                (0.5+0.2)/2,               # Case 2
                0.2                        # Case 3
            )
        )

        n_gen_fill = nfill * 2 * Drag_integral / np.sqrt(np.pi * CD0)

        MIT_sol = evaluate_mit(data, n_gen_fill, Rm)

        mit_label = f"nfill = {nfill}"

        ax1.plot(A_ballistic, MIT_sol, label=mit_label, linewidth=2)
        
    if handles is None:  # only capture on the first subplot
        handles, labels = ax1.get_legend_handles_labels()


    ax1.set_xscale("log") 
    # ax1.set_yscale("log") 
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax1.set_xlim(min(A_ballistic),max(A_ballistic))
    ax1.set_ylim(0,2)

    ax1.set_xlabel("Ballistic Coefficient A [-]")
    ax1.set_ylabel("Ck")
    ax1.set_title(f"Cx = {Cx}; CD0 = {CD0}")
    #plt.figtext(0.70, 0.2, f'Cx = {Cx}\nCD0 = {CD0}', bbox=dict(facecolor='white', edgecolor='black', pad=10.0))

# Use the last (6th) subplot for the legend
axes2[-1].axis("off")  # hide axes
axes2[-1].legend(handles, labels, loc="center", fontsize=10)

fig2.suptitle("Comparison for different CD0 values", fontsize=14)
plt.tight_layout()


plt.show()