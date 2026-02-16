"""
Utility to visualize Jacobian sparsity and coloring in OpenMDAO.
"""

import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om


def visualize_sparsity(prob, of=None, wrt=None, show=True, save_filename=None):
    """
    Visualize the sparsity pattern of the Jacobian matrix.
    
    Parameters
    ----------
    prob : Problem
        OpenMDAO Problem instance (must be set up)
    of : str or list, optional
        Outputs of interest (default: all outputs)
    wrt : str or list, optional
        Inputs of interest (default: all inputs)
    show : bool
        Whether to display the plot
    save_filename : str, optional
        Filename to save the plot
    """
    # Compute the Jacobian
    prob.compute_totals(of=of, wrt=wrt)
    
    # Get the Jacobian matrix from the driver
    J = prob.compute_totals(of=of, wrt=wrt, return_format='array')
    
    # Create sparsity plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create binary mask: 1 for non-zero, 0 for zero
    sparsity_mask = (np.abs(J) > 1e-12).astype(float)
    
    # Plot sparsity pattern
    ax.imshow(sparsity_mask, cmap='Greys', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Design Variables (inputs)', fontsize=12)
    ax.set_ylabel('Objectives/Constraints (outputs)', fontsize=12)
    ax.set_title('Jacobian Sparsity Pattern\n(Black = Non-zero, White = Zero)', fontsize=14)
    
    # Calculate sparsity percentage
    total_elements = J.size
    non_zero_elements = np.sum(sparsity_mask)
    sparsity_pct = (1 - non_zero_elements / total_elements) * 100
    
    ax.text(0.02, 0.98, f'Sparsity: {sparsity_pct:.2f}%\nNon-zero: {int(non_zero_elements)}/{total_elements}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_coloring_info(prob, show=True):
    """
    Print information about coloring if available.
    
    Parameters
    ----------
    prob : Problem
        OpenMDAO Problem instance (must be set up)
    show : bool
        Whether to print information
    """
    driver = prob.driver
    
    if hasattr(driver, '_coloring_info') and driver._coloring_info:
        coloring_info = driver._coloring_info
        print("\n" + "="*60)
        print("Coloring Information")
        print("="*60)
        print(f"Total colors: {coloring_info.get('total_colors', 'N/A')}")
        print(f"Improvement: {coloring_info.get('improvement_pct', 0):.2f}%")
        print(f"Sparsity: {coloring_info.get('sparsity', 'N/A')}")
        print("="*60 + "\n")
    else:
        if show:
            print("No coloring information available. Coloring may not be enabled or beneficial.")
    
    # Try to get coloring from compute_totals
    try:
        prob.compute_totals()  # This may trigger coloring computation
        if hasattr(driver, '_coloring_info'):
            coloring_info = driver._coloring_info
            if show and coloring_info:
                print("\nColoring Information:")
                print(f"  Total colors: {coloring_info.get('total_colors', 'N/A')}")
                print(f"  Improvement: {coloring_info.get('improvement_pct', 0):.2f}%")
    except:
        pass

