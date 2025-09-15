import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval

def evaluate_mit(data, case, x_values, plotting = None):
    """
    Evaluate a polynomial using MIT coefficients for a given case.

    Parameters:
        data (dict): The loaded pickle data containing 'MIT_coeffs'.
        case (str): The specific case (e.g., 'lower', 'upper').
        x_values (array-like): Points at which to evaluate the polynomial.

    Returns:
        array: Evaluated y-values.
    """
    # Extract coefficients for the case
    coeffs = data['MIT_coeffs'].get(case)
    if coeffs is None:
        raise ValueError(f"No coefficients found for case: {case}")
    
    # Evaluate the polynomial
    polynomial = np.poly1d(coeffs)
    
    sol = polynomial(np.log10(x_values))
    
    if plotting is not None:
        if np.isscalar(x_values):
            plot_MIT(data,case)
            plt.plot([x_values,x_values],[0, sol],color="blue")
            plt.plot([0,x_values],[sol, sol],color="blue")            
            plt.show()
        elif isinstance(x_values,np.ndarray):
            plot_MIT(data,case)
            
            plt.plot([np.min(x_values),np.min(x_values)],[0, np.min(sol)],color="blue")
            plt.plot([np.max(x_values),np.max(x_values)],[0, np.max(sol)],color="blue")
            
        
            plt.plot([x_values,x_values],[0, sol],color="blue")
            plt.plot([0,x_values],[sol, sol],color="blue")            
            plt.show()
        else:
            raise ValueError(f'Plotting cannot be performed with multiple input values')                              
    
    return sol

def evaluate_pflanz(data, case, x_values, plotting = None):
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
    
    if plotting is not None:
        if isinstance(x_values,float):
            plot_pflanz(data,case)
            plt.plot([x_values,x_values],[0, sol],color="blue")
            plt.plot([0,x_values],[sol, sol],color="blue")            
            plt.show()            
        elif isinstance(x_values,np.ndarray):
            pflanz_ax = plot_pflanz(data, case, return_axes=True)
            
            line = pflanz_ax.get_lines()[0] # Get first 2D Line Object
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            
            min_x = np.min(x_values)
            max_x = np.max(x_values)    
            
            min_sol = np.min(sol)
            max_sol = np.max(sol)                
            
            pflanz_ax.fill_between(x_data, y_data, where=(x_data >= min_x) & (x_data <= max_x), color="red", alpha=0.8, label="Shaded Region")
                                  
            pflanz_ax.plot([min_x,min_x],[0, min_sol],color="blue")
            pflanz_ax.plot([max_x,max_x],[0, max_sol],color="blue")
                    
            plt.show()
        else:
            raise ValueError(f'Plotting cannot be performed with multiple input values')                              
    
    return sol

def plot_pflanz(data, case, return_axes=False):
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
    
    fig, ax = plt.subplots()    
    ax.plot(PflanzData_log_x, PflanzData_log_y, label=curve_label, color="red", linewidth=2, linestyle=curve_style) 
    ax.set_xscale("log") 
    ax.set_yscale("log") 
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    ax.set_xlim(0.01,1000)
    ax.set_ylim(0.01,1)
    
    ax.set_xlabel("Ballistic Coefficient A [-]")
    ax.set_ylabel("Opening Force Reduction Factor - X1")
    ax.set_title("Pflanz Interpolated Data")
    
    if return_axes:
        return ax
    
           
           
def plot_MIT(data, case):
    """
    This function plot both MIT graphs
    """    
    
    MIT_x=10**np.linspace(-4,1,100)
    
    match case:
        case 'lower':
            plot_title = "Opening Shock Factor 1 <= n_gen <= 4 - Lower"
        case 'upper':
            plot_title = "Opening Shock Factor n_gen >= 4 - Upper"
        case _:
            raise ValueError(f"case must be either lower or upper")
                                        
                                            
    plt.plot(MIT_x, evaluate_mit(data,case,MIT_x), label="Data", color="red", linewidth=2)   
    plt.gca().set_xscale("log") 
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    plt.xlim(0.0001,10)
    plt.ylim(0,2)
    
    plt.xlabel("Mass Ratio - Rm")
    plt.ylabel("Opening Shock Factor - Ck")
    plt.title(plot_title)
    plt.legend()
    
    

           
       
    
    
