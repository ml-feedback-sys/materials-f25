import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, Poly, lambdify

def derive_cubic_symbolic(noisy=True):
    """
    Derive the cubic equation symbolically once.
    Returns symbolic coefficients as functions of s_hat and p.
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

# Create side-by-side plots for optimal u
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

im1 = ax1.contourf(S, P, U_noisy, levels=50, cmap='viridis')
ax1.set_title('Optimal u (Multiplicative Noise)\n' + fstring_noisy, fontsize=10)
ax1.set_xlabel('ŝ', fontsize=12)
ax1.set_ylabel('p', fontsize=12)
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('u (optimal)', fontsize=12)

im2 = ax2.contourf(S, P, U_clean, levels=50, cmap='viridis')
ax2.set_title('Optimal u (Additive Noise)\n' + fstring_clean, fontsize=10)
ax2.set_xlabel('ŝ', fontsize=12)
ax2.set_ylabel('p', fontsize=12)
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('u (optimal)', fontsize=12)

plt.tight_layout()
plt.show()

# Create side-by-side plots for objective function values
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))

im3 = ax3.contourf(S, P, Obj_noisy, levels=50, cmap='plasma')
ax3.set_title('Minimum J(u) (Mult)', fontsize=12)
ax3.set_xlabel('ŝ', fontsize=12)
ax3.set_ylabel('p', fontsize=12)
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label('J(u) (minimum)', fontsize=12)

im4 = ax4.contourf(S, P, Obj_clean, levels=50, cmap='plasma')
ax4.set_title('Minimum J(u) (Add)', fontsize=12)
ax4.set_xlabel('ŝ', fontsize=12)
ax4.set_ylabel('p', fontsize=12)
cbar4 = plt.colorbar(im4, ax=ax4)
cbar4.set_label('J(u) (minimum)', fontsize=12)

plt.tight_layout()
plt.show()

# Optional: Plot the difference between noisy and clean
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(18, 7))

diff_u = U_noisy - U_clean
im5 = ax5.contourf(S, P, diff_u, levels=50, cmap='RdBu_r')
ax5.set_title('Difference in Optimal u (Mult - Add)', fontsize=12)
ax5.set_xlabel('ŝ', fontsize=12)
ax5.set_ylabel('p', fontsize=12)
cbar5 = plt.colorbar(im5, ax=ax5)
cbar5.set_label('Δu', fontsize=12)

diff_obj = Obj_noisy - Obj_clean
im6 = ax6.contourf(S, P, diff_obj, levels=50, cmap='RdBu_r')
ax6.set_title('Difference in Minimum J(u) (Mult - Add)', fontsize=12)
ax6.set_xlabel('ŝ', fontsize=12)
ax6.set_ylabel('p', fontsize=12)
cbar6 = plt.colorbar(im6, ax=ax6)
cbar6.set_label('ΔJ(u)', fontsize=12)

plt.tight_layout()
plt.show()