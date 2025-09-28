import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval

def evaluate_mit(data, n_gen_fill, x_values):
    """
    Evaluate a polynomial using MIT coefficients for a given case.

    Case determination:
        - n_gen_fill >= 4       -> "upper"
        - 1 <= n_gen_fill < 4   -> "lower"
        - n_gen_fill < 1        -> error

    Parameters:
        data (dict): The loaded pickle data containing 'MIT_coeffs'.
        n_gen_fill (float or array-like): Generalized fill constant used to determine which MIT interpolation case to use
        x_values (float or array-like): Points at which to evaluate the polynomial.        

    Returns:
        np.ndarray: Evaluated y-values.
    """
    # Extract coefficients for the case
    coeffs_dict = data.get("MIT_coeffs", {})
    if not coeffs_dict:
        raise ValueError("No MIT coefficients found in data")
    
    # Normalize to arrays
    x_values = np.atleast_1d(x_values)
    n_gen_fill = np.atleast_1d(n_gen_fill)
    
    if len(n_gen_fill) != len(x_values):
        raise ValueError("'n_gen_fill' must be same length as 'x_values'")
    
    # Determine case from n_gen_fill
    case = np.full_like(x_values, None, dtype=object)

    mask_upper = n_gen_fill >= 4
    mask_lower = (n_gen_fill >= 1) & (n_gen_fill < 4)
    mask_invalid = n_gen_fill < 1
    
    case[mask_upper] = "upper"
    case[mask_lower] = "lower"
    
    if mask_invalid.any():
        print(f"Cases below 1: {np.sum(n_gen_fill < 1)}. These Values are not represented")    
    
    sol = np.empty_like(x_values, dtype=float)
    used_cases = set()
    
    # Evaluate per case
    for c in np.unique(case[case != None]): # There are two cases "upper" and "lower". Only loop twice. Once for each. 
        # Must ignore None values created by the invalid <1 case
        coeffs = coeffs_dict.get(c)
        if coeffs is None:
            raise ValueError(f"No coefficients found for case: {c}")

        mask = (case == c)
        polynomial = np.poly1d(coeffs)
        sol[mask] = polynomial(np.log10(x_values[mask])) # x_values[mask] only uses the x_values pertaining to a certain case
        used_cases.add(c)                                          
    
    return sol

def plot_MIT(data, case, ax):
    """
    This function plot both MIT graphs
    """    
    plot_resolution = 100
    MIT_x=10**np.linspace(-4,1,plot_resolution)
    
    match case:
        case 'lower':
            plot_title = "Opening Shock Factor 1 <= n_gen <= 4 - Lower"
            n_gen_fill = np.ones(plot_resolution)*2 # Creates a fake n_gen_fill array so evaluate_MIT works
        case 'upper':
            plot_title = "Opening Shock Factor n_gen >= 4 - Upper"
            n_gen_fill = np.ones(plot_resolution)*5 # Creates a fake n_gen_fill array so evaluate_MIT works
        case _:
            raise ValueError(f"case must be either lower or upper")
                                                   
    ax.plot(MIT_x, evaluate_mit(data,n_gen_fill,MIT_x), label="Data", color="red", linewidth=2)   
    ax.set_xscale("log") 
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    ax.set_xlim(0.0001,10)
    ax.set_ylim(0,2)
    
    ax.set_xlabel("Mass Ratio - Rm")
    ax.set_ylabel("Opening Shock Factor - Ck")
    ax.set_title(plot_title)
    ax.legend()
    
def plot_MIT_interpolation(data, n_gen_fill, x_values, sol, ax):

    # Normalize to arrays
    x_values = np.atleast_1d(x_values)
    n_gen_fill = np.atleast_1d(n_gen_fill)
    
    if len(n_gen_fill) != len(x_values):
        raise ValueError("'n_gen_fill' must be same length as 'x_values'")
    
    # Determine case from n_gen_fill
    case = np.full_like(x_values, None, dtype=object)

    mask_upper = n_gen_fill >= 4
    mask_lower = (n_gen_fill >= 1) & (n_gen_fill < 4)
    mask_invalid = n_gen_fill < 1
    
    case[mask_upper] = "upper"
    case[mask_lower] = "lower"
    
    if mask_invalid.any():
        print(f"Cases below 1: {np.sum(n_gen_fill < 1)}. These Values are not represented")    
       
    cases_used = np.unique([c for c in case if c is not None])
           
    # Loop over used cases
    for i, c in enumerate(cases_used):
        mask = (case == c)
        
        # If more than one case is used (n_gen_fill vector has cases above 4 and below 4) 
        # -> create new figure/axes for each after the first
        if i == 0:
            current_ax = ax  # use the provided one for the first case
        else:
            fig, current_ax = plt.subplots()  # new figure for extra cases

        # Draw the MIT interpolating polynomial for this case
        plot_MIT(data, c, current_ax)
                
        if mask.sum() == 1:  # single value in this particular case
            xv = x_values[mask][0]
            yv = sol[mask][0]

            current_ax.plot([xv, xv], [0, yv], color="blue")
            current_ax.plot([0, xv], [yv, yv], color="blue")
            
        elif mask.sum() > 1:  # multiple values in this particlar case
            xv = x_values[mask]
            yv = sol[mask]
            
            current_ax.plot([np.min(xv), np.min(xv)], [0, np.max(yv)], color="blue")
            current_ax.plot([np.max(xv), np.max(xv)], [0, np.min(yv)], color="blue")
            
        else:
            continue # No points for this case. Shouldnt happen   

############################################################
##################### PFLANZ FUNCTIONS #####################
############################################################

def evaluate_pflanz(data, case, x_values):
    """
    Evaluate a PCHIP interpolator for a given case.

    Parameters:
        data (dict): The loaded pickle data containing 'Pflanz_functions'.
        case (str): The specific case (e.g., '0.5', '1', '2').
        x_values (array-like): Points at which to evaluate the interpolator.       

    Returns:
        array: Interpolated y-values.
    """
    # Extract the PCHIP interpolator for the case
    interpolator = data['Pflanz_functions'].get(case)
    if interpolator is None:
        raise ValueError(f"No PCHIP interpolator found for case: {case}")
    
    sol = 10**interpolator(np.log10(x_values))
        
    return sol

def plot_pflanz(data, case, ax):
    """
    This function just returns the plot of the Pflanz method
    """
    # Instead of simply doing this: PflanzData_log_x = np.linspace(0.01,1000,200)
    # By generating it this way, the resolution of points is evenly spread in log space
    # Else, most points are contained towads higher values of Ballistic Coefficient
    
    PflanzData_log_x = 10**np.linspace(np.log10(0.01),np.log10(1000),200)
    
    curve_style = ['dashed','solid','dashdot']
    curve_label = ['n= 1/2','n = 1','n = 2']  
    
    match case:
        case '0.5':
            curve_label = 'n = 1/2'
            curve_style = 'dashed'
        case '1':
            curve_label = 'n = 1'
            curve_style = 'solid'
        case '2':
            curve_label = 'n = 2'
            curve_style = 'dashdot'
        case _:
            raise ValueError(f"case must be either 0.5, 1 or 2")
    
    PflanzData_log_y = evaluate_pflanz(data,case,PflanzData_log_x)
        
    ax.plot(PflanzData_log_x, PflanzData_log_y, label=curve_label, color="red", linewidth=2, linestyle=curve_style) 
    ax.set_xscale("log") 
    ax.set_yscale("log") 
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    ax.set_xlim(0.01,1000)
    ax.set_ylim(0.01,1)
    
    ax.set_xlabel("Ballistic Coefficient A [-]")
    ax.set_ylabel("Opening Force Reduction Factor - X1")
    ax.set_title("Pflanz Interpolated Data")

def plot_pflanz_interpolation(data, case, x_values, pflanz_sol, ax):

    valid_cases = {"0.5", "1", "2"}  # set of allowed strings
    if not isinstance(case, str) or case not in valid_cases:
        raise ValueError(f"case must be one of {valid_cases}, got {case!r}")

    if isinstance(x_values,float):          
        plot_pflanz(data,case, ax)
        ax.plot([x_values,x_values],[0, pflanz_sol],color="blue")
        ax.plot([0,x_values],[pflanz_sol, pflanz_sol],color="blue")  
                                       
    elif isinstance(x_values,np.ndarray):            
        plot_pflanz(data, case, ax)
        
        line = ax.get_lines()[0] # Get first 2D Line Object
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        
        min_x = np.min(x_values)
        max_x = np.max(x_values)    
        
        min_sol = np.min(pflanz_sol)
        max_sol = np.max(pflanz_sol)                
        
        ax.fill_between(x_data, y_data, where=(x_data >= min_x) & (x_data <= max_x), color="red", alpha=0.8, label="Shaded Region")
                                
        ax.plot([min_x,min_x],[0, min_sol],color="blue")
        ax.plot([max_x,max_x],[0, max_sol],color="blue")
                
    else:
        raise ValueError(f'Plotting cannot be performed with multiple input values')                              
