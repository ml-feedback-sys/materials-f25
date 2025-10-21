import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, Poly, lambdify

def derive_cubic_symbolic(noisy=True):
    """
    Derive the cubic equation symbolically once.
    Returns symbolic coefficients as functions of s_hat and p.
    Credit to Kelly Jiang for converting this to sympy!
    """
    s_hat, p, u = symbols('s_hat p u')
    if noisy:
        f = (s_hat+u)**4 + 6*(s_hat+u)**2*(p**2+u**2) + 3*(p**2+u**2)**2 + 2*u**2
    else:
        f = (s_hat+u)**4 + 6*(s_hat+u)**2*(p**2+1) + 3*(p**2+1)**2 + 2*u**2
    f_prime = diff(f, u)
    poly = Poly(f_prime, u)
    coeffs = poly.all_coeffs()
    
    # Convert symbolic coefficients to numerical functions
    coeff_funcs = [lambdify((s_hat, p), coeff, 'numpy') for coeff in coeffs]
    return coeff_funcs, str(f)

def solve_cubic(s_hat, p, coeff_funcs):
    """
    Solve the cubic equation for given numerical values of s_hat and p.
    """
    # Evaluate the symbolic coefficients at the given s_hat and p
    coefficients = [func(s_hat, p) for func in coeff_funcs]
    
    # Use numpy's roots function for cubic equation
    roots = np.roots(coefficients)
    return roots

def objective_function(s_hat, u, p, noisy=True):
    term1 = (s_hat + u)**4
    if noisy:
        term2 = 6*(s_hat+u)**2*(p**2+u**2)
        term3 = 3*(p**2+u**2)**2
    else:
        term2 = 6*(s_hat+u)**2*(p**2)
        term3 = 3*(p**2)**2
    term4 = 2 * u**2
    return term1 + term2 + term3 + term4

def select_best_root(s_hat, p, coeff_funcs, noisy=True):
    """
    Among real roots, select the one that minimizes the objective function
    """
    roots = solve_cubic(s_hat, p, coeff_funcs)
    
    # Filter for real roots (imaginary part close to zero)
    real_roots = []
    for root in roots:
        if np.abs(root.imag) < 1e-10:  # Tolerance for numerical errors
            real_roots.append(root.real)
    
    # If no real roots, return NaN
    if len(real_roots) == 0:
        return np.nan
    
    # Evaluate objective function for each real root
    best_root = None
    best_value = np.inf  # Looking for minimum
    
    for u in real_roots:
        obj_value = objective_function(s_hat, u, p, noisy=noisy)
        if obj_value < best_value:  # Minimize
            best_value = obj_value
            best_root = u
    
    return best_root

def compute_solution(noisy=True):
    """
    Compute the optimal u for all grid points for a given noisy setting
    """
    print(f"Deriving symbolic cubic equation (noisy={noisy})...")
    coeff_funcs, fstring = derive_cubic_symbolic(noisy=noisy)
    print("Done! Now solving numerically for each grid point...")
    
    # Create grid
    s_values = np.linspace(-3, 3, 200)
    p_values = np.linspace(0, 3, 200)
    S, P = np.meshgrid(s_values, p_values)
    
    # Initialize array for optimal u
    U_optimal = np.zeros_like(S)
    
    # Solve for each point in the grid
    for i in range(len(p_values)):
        for j in range(len(s_values)):
            U_optimal[i, j] = select_best_root(S[i, j], P[i, j], coeff_funcs, noisy=noisy)
    
    # Compute objective function values
    Obj_values = np.zeros_like(S)
    for i in range(len(p_values)):
        for j in range(len(s_values)):
            u_opt = U_optimal[i, j]
            if not np.isnan(u_opt):
                Obj_values[i, j] = objective_function(S[i, j], u_opt, P[i, j], noisy=noisy)
            else:
                Obj_values[i, j] = np.nan
    
    return S, P, U_optimal, Obj_values, fstring


# Compute solutions for both cases
S, P, U_noisy, Obj_noisy, fstring_noisy = compute_solution(noisy=True)
_, _, U_clean, Obj_clean, fstring_clean = compute_solution(noisy=False)

# Choose 5 p values to plot
p_plot_values = [0.3, 0.8, 1.3, 1.8, 2.3]
# Get the grid values
s_values = S[0, :]
p_values = P[:, 0]

# Extract line data for specific p values from the grid
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(p_plot_values)))

# Create side-by-side line plots for optimal u
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for i, p_val in enumerate(p_plot_values):
    # Find the index of the closest p value in the grid
    p_idx = np.argmin(np.abs(p_values - p_val))
    
    # Extract the slice for this p value
    U_noisy_slice = U_noisy[p_idx, :]
    ax1.plot(s_values, U_noisy_slice, label=f'p = {p_val}', color=colors[i], linewidth=2)

ax1.set_title('Optimal u (Multiplicative Noise)', fontsize=14)
ax1.set_xlabel('ŝ', fontsize=12)
ax1.set_ylabel('u (optimal)', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

for i, p_val in enumerate(p_plot_values):
    # Find the index of the closest p value in the grid
    p_idx = np.argmin(np.abs(p_values - p_val))
    
    # Extract the slice for this p value
    U_clean_slice = U_clean[p_idx, :]
    ax2.plot(s_values, U_clean_slice, label=f'p = {p_val}', color=colors[i], linewidth=2)

ax2.set_title('Optimal u (Additive Noise)', fontsize=14)
ax2.set_xlabel('ŝ', fontsize=12)
ax2.set_ylabel('u (optimal)', fontsize=12)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create side-by-side line plots for objective function values
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

for i, p_val in enumerate(p_plot_values):
    # Find the index of the closest p value in the grid
    p_idx = np.argmin(np.abs(p_values - p_val))
    
    # Extract the slice for this p value
    Obj_noisy_slice = Obj_noisy[p_idx, :]
    ax3.plot(s_values, Obj_noisy_slice, label=f'p = {p_val}', color=colors[i], linewidth=2)

ax3.set_title('Minimum J(u) (Multiplicative Noise)', fontsize=14)
ax3.set_xlabel('ŝ', fontsize=12)
ax3.set_ylabel('J(u) (minimum)', fontsize=12)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

for i, p_val in enumerate(p_plot_values):
    # Find the index of the closest p value in the grid
    p_idx = np.argmin(np.abs(p_values - p_val))
    
    # Extract the slice for this p value
    Obj_clean_slice = Obj_clean[p_idx, :]
    ax4.plot(s_values, Obj_clean_slice, label=f'p = {p_val}', color=colors[i], linewidth=2)

ax4.set_title('Minimum J(u) (Additive Noise)', fontsize=14)
ax4.set_xlabel('ŝ', fontsize=12)
ax4.set_ylabel('J(u) (minimum)', fontsize=12)
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create overlay comparison plot
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))

for i, p_val in enumerate(p_plot_values):
    # Find the index of the closest p value in the grid
    p_idx = np.argmin(np.abs(p_values - p_val))
    
    # Extract the slices for this p value
    U_noisy_slice = U_noisy[p_idx, :]
    U_clean_slice = U_clean[p_idx, :]
    
    ax5.plot(s_values, U_noisy_slice, label=f'p = {p_val} (noisy)', 
             color=colors[i], linewidth=2, linestyle='-')
    ax5.plot(s_values, U_clean_slice, color=colors[i], linewidth=2, 
             linestyle='--', alpha=0.7)

ax5.set_title('Optimal u: Multiplicative (solid) vs Additive (dashed)', fontsize=14)
ax5.set_xlabel('ŝ', fontsize=12)
ax5.set_ylabel('u (optimal)', fontsize=12)
ax5.legend(loc='best')
ax5.grid(True, alpha=0.3)

for i, p_val in enumerate(p_plot_values):
    # Find the index of the closest p value in the grid
    p_idx = np.argmin(np.abs(p_values - p_val))
    
    # Extract the slices for this p value
    Obj_noisy_slice = Obj_noisy[p_idx, :]
    Obj_clean_slice = Obj_clean[p_idx, :]
    
    ax6.plot(s_values, Obj_noisy_slice, label=f'p = {p_val} (noisy)', 
             color=colors[i], linewidth=2, linestyle='-')
    ax6.plot(s_values, Obj_clean_slice, color=colors[i], linewidth=2, 
             linestyle='--', alpha=0.7)

ax6.set_title('Minimum J(u): Noisy (Multiplicative) vs Clean (Additive)', fontsize=14)
ax6.set_xlabel('ŝ', fontsize=12)
ax6.set_ylabel('J(u) (minimum)', fontsize=12)
ax6.legend(loc='best')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()