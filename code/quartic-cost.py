import numpy as np
import matplotlib.pyplot as plt

def solve_cubic(s_hat, sigma):
    """
    Solve: u³ + 3ŝu² + (3ŝ² + 3σ² + 1)u + (ŝ³ + 3ŝσ²) = 0
    Returns all three roots (real or complex)
    """
    a = 1
    b = 3 * s_hat
    c = 3 * s_hat**2 + 3 * sigma**2 + 1
    d = s_hat**3 + 3 * s_hat * sigma**2
    
    # Use numpy's roots function for cubic equation
    coefficients = [a, b, c, d]
    roots = np.roots(coefficients)
    
    return roots

def objective_function(s_hat, a, sigma):
    """
    Calculate: (ŝ + a)⁴ + 6(ŝ + a)²σ² + 3σ⁴ + 2a²
    """
    term1 = (s_hat + a)**4
    term2 = 6 * (s_hat + a)**2 * sigma**2
    term3 = 3 * sigma**4
    term4 = 2 * a**2
    return term1 + term2 + term3 + term4

def select_best_root(s_hat, sigma):
    """
    Among real roots, select the one that maximizes the objective function
    """
    roots = solve_cubic(s_hat, sigma)
    
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
    best_value = -np.inf
    
    for a in real_roots:
        obj_value = objective_function(s_hat, a, sigma)
        if obj_value > best_value:
            best_value = obj_value
            best_root = a
    
    return best_root

# Create grid
s_values = np.linspace(-3, 3, 200)
sigma_values = np.linspace(0, 3, 200)
S, Sigma = np.meshgrid(s_values, sigma_values)

# Initialize array for optimal u
U_optimal = np.zeros_like(S)

# Solve for each point in the grid
for i in range(len(sigma_values)):
    for j in range(len(s_values)):
        U_optimal[i, j] = select_best_root(S[i, j], Sigma[i, j])

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.contourf(S, Sigma, U_optimal, levels=50, cmap='viridis')
ax.set_title('Optimal u that maximizes (ŝ + u)⁴ + 6(ŝ + u)²σ² + 3σ⁴ + 2u²', 
             fontsize=12)
ax.set_xlabel('ŝ', fontsize=12)
ax.set_ylabel('σ', fontsize=12)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('u (optimal)', fontsize=12)
plt.tight_layout()
plt.show()

# Optional: Also plot the objective function value at the optimal u
Obj_values = np.zeros_like(S)
for i in range(len(sigma_values)):
    for j in range(len(s_values)):
        u_opt = U_optimal[i, j]
        if not np.isnan(u_opt):
            Obj_values[i, j] = objective_function(S[i, j], u_opt, Sigma[i, j])
        else:
            Obj_values[i, j] = np.nan

fig2, ax2 = plt.subplots(figsize=(10, 8))
im2 = ax2.contourf(S, Sigma, Obj_values, levels=50, cmap='plasma')
ax2.set_title('Maximum value of (ŝ + u)⁴ + 6(ŝ + u)²σ² + 3σ⁴ + 2u²', 
              fontsize=12)
ax2.set_xlabel('ŝ', fontsize=12)
ax2.set_ylabel('σ', fontsize=12)
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Objective value', fontsize=12)
plt.tight_layout()
plt.show()