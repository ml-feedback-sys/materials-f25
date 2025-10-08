import numpy as np
import matplotlib.pyplot as plt

def solve_cubic(s_hat, p):
    """
    Solve: u³ + ((12ŝ + 18p)/10)u² + ((6ŝ² + 12ŝp + 12p² + 1)/10)u 
           + ((ŝ³ + 3ŝ²p + 3ŝp² + 3p³)/10) = 0
    Returns all three roots (real or complex)
    """
    a = 1
    b = (12*s_hat + 18*p) / 10
    c = (6*s_hat**2 + 12*s_hat*p + 12*p**2 + 1) / 10
    d = (s_hat**3 + 3*s_hat**2*p + 3*s_hat*p**2 + 3*p**3) / 10
    
    # Use numpy's roots function for cubic equation
    coefficients = [a, b, c, d]
    roots = np.roots(coefficients)
    
    return roots

def objective_function(s_hat, u, p):
    """
    Calculate: J(u) = (ŝ + u)⁴ + 6(ŝ + u)²(p + u)² + 3(p + u)⁴ + 2u²
    """
    term1 = (s_hat + u)**4
    term2 = 6 * (s_hat + u)**2 * (p + u)**2
    term3 = 3 * (p + u)**4
    term4 = 2 * u**2
    return term1 + term2 + term3 + term4

def select_best_root(s_hat, p):
    """
    Among real roots, select the one that maximizes the objective function
    """
    roots = solve_cubic(s_hat, p)
    
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
    
    for u in real_roots:
        obj_value = objective_function(s_hat, u, p)
        if obj_value > best_value:
            best_value = obj_value
            best_root = u
    
    return best_root

# Create grid
s_values = np.linspace(-3, 3, 200)
p_values = np.linspace(0, 3, 200)
S, P = np.meshgrid(s_values, p_values)

# Initialize array for optimal u
U_optimal = np.zeros_like(S)

# Solve for each point in the grid
for i in range(len(p_values)):
    for j in range(len(s_values)):
        U_optimal[i, j] = select_best_root(S[i, j], P[i, j])

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.contourf(S, P, U_optimal, levels=50, cmap='viridis')
ax.set_title('Optimal u that maximizes J(u) = (ŝ + u)⁴ + 6(ŝ + u)²(p + u)² + 3(p + u)⁴ + 2u²', 
             fontsize=11)
ax.set_xlabel('ŝ', fontsize=12)
ax.set_ylabel('p', fontsize=12)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('u (optimal)', fontsize=12)
plt.tight_layout()
plt.show()

# Optional: Also plot the objective function value at the optimal u
Obj_values = np.zeros_like(S)
for i in range(len(p_values)):
    for j in range(len(s_values)):
        u_opt = U_optimal[i, j]
        if not np.isnan(u_opt):
            Obj_values[i, j] = objective_function(S[i, j], u_opt, P[i, j])
        else:
            Obj_values[i, j] = np.nan

fig2, ax2 = plt.subplots(figsize=(10, 8))
im2 = ax2.contourf(S, P, Obj_values, levels=50, cmap='plasma')
ax2.set_title('Maximum value of J(u)', fontsize=12)
ax2.set_xlabel('ŝ', fontsize=12)
ax2.set_ylabel('p', fontsize=12)
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('J(u) (maximum)', fontsize=12)
plt.tight_layout()
plt.show()