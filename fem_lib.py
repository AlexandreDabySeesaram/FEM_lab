#coding=utf8

################################################################################
###                                                                          ###
### Created by Alexandre Daby-Seesaram                                       ###
###                                                                          ###
### ENSTA, Institut Polytechnique de Paris                                   ###
###                                                                          ###
################################################################################


"""
FEM 1D library (fem_lib.py)
===============================
This module contains core functions for 1D Finite Element Method analysis
using NumPy and SciPy. 

Functions included:
* Gauss-Legendre Quadrature rules (order 1, 2, and 3).
* P1 (linear) and P2 (quadratic) shape functions and their reference derivatives.
* Global stiffness and load vector assembly.
* Dirichlet boundary condition enforcement:
   - penalisation Method
   - Lagrange Multipliers Method
"""

import numpy as np

def gauss_quadrature(order):
    """
    Returns Gauss integration points and weights on the reference interval [-1, 1].
    """
    if order == 1:
        return np.array([0.0]), np.array([2.0])
    elif order == 2:
        val = 1.0 / np.sqrt(3.0)
        return np.array([-val, val]), np.array([1.0, 1.0])
    elif order == 3:
        val = np.sqrt(3.0 / 5.0)
        return np.array([-val, 0.0, val]), np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    else:
        raise ValueError("Only Gauss order 1, 2, and 3 are supported.")

def shape_functions(element_type, xi):
    """
    Computes shape functions N and reference derivatives dN/d_xi at point xi.
    
    Parameters:
        element_type: "P1" (linear, 2 nodes) or "P2" (quadratic, 3 nodes)
        xi: position in reference domain [-1, 1]
        
    Returns:
        N: numpy array of shape function values
        dN_dxi: numpy array of reference derivatives
    """
    if element_type == "P1":
        # 2 nodes: node 0 at xi=-1, node 1 at xi=1
        N       = np.array([0.5 * (1.0 - xi), 
                            0.5 * (1.0 + xi)])
        dN_dxi  = np.array([-0.5, 
                            0.5])
    elif element_type == "P2":
        # 3 nodes: node 0 at xi=-1, node 1 at xi=0, node 2 at xi=1
        N       = np.array([0.5 * xi * (xi - 1.0), 
                            1.0 - xi**2, 
                            0.5 * xi * (xi + 1.0)])
        dN_dxi  = np.array([xi - 0.5, 
                           -2.0 * xi, 
                           xi + 0.5])
    else:
        raise ValueError(f"Unknown element type: {element_type}")
        
    return N, dN_dxi

def assemble_global_system(x_nodes, elements, element_type, E, A, q_dist):
    """
    Assembles the global stiffness matrix K and load vector F.
    
    Parameters:
        x_nodes: coordinates of all nodes (shape: num_nodes)
        elements: element connectivity list/array of shape (num_elements, num_local_nodes)
        element_type: "P1" or "P2"
        E: Young's Modulus (constant or function)
        A: Cross-sectional area (constant or function)
        q_dist: distributed load (constant or function)
        
    Returns:
        K: global stiffness matrix (numpy array of shape num_nodes x num_nodes)
        F: global load vector (numpy array of shape num_nodes)
    """
    num_nodes       = len(x_nodes)
    K               = np.zeros((num_nodes, num_nodes))
    F               = np.zeros(num_nodes)
    
    # Use 3-point Gauss quadrature (exact for polynomials up to degree 5)
    xi_g, w_g = gauss_quadrature(3)
    
    for elem in elements:
        # Physical coordinates of element nodes
        x_elem      = x_nodes[elem]
        x_start     = x_elem[0]
        x_end       = x_elem[-1]
        Le          = x_end - x_start
        
        # Jacobian for 1D mapping from [-1, 1] to [x_start, x_end]
        # dx/d_xi = Le / 2
        J           = Le / 2.0
        
        num_local   = len(elem)
        Ke          = np.zeros((num_local, num_local))
        Fe          = np.zeros(num_local)
        
        # Gauss integration loop
        for xi, w in zip(xi_g, w_g):
            N, dN_dxi = shape_functions(element_type, xi)
            
            # Physical derivatives: dN/dx = dN/dxi * dxi/dx = dN/dxi / J
            dN_dx   = dN_dxi / J
            
            # Evaluate properties at the physical coordinate x(xi)
            x_val   = 0.5 * (x_start + x_end) + J * xi
            E_val   = E(x_val) if callable(E) else E
            A_val   = A(x_val) if callable(A) else A
            q_val   = q_dist(x_val) if callable(q_dist) else q_dist
            
            # Integrate element stiffness matrix
            Ke     += E_val * A_val * np.outer(dN_dx, dN_dx) * J * w
            
            # Integrate element load vector
            Fe     += q_val * N * J * w
            
        # Assemble into global system
        for a in range(num_local):
            global_a        = elem[a]
            F[global_a]    += Fe[a]
            for b in range(num_local):
                global_b                = elem[b]
                K[global_a, global_b]  += Ke[a, b]
                
    return K, F

def apply_bc_penalisation(K, F, fixed_dofs, fixed_vals, pen_factor=1e12):
    """
    Applies Dirichlet boundary conditions using the penalisation method.
    K_ii = K_ii + penalty
    F_i = F_i + penalty * val
    """
    K_bc = K.copy()
    F_bc = F.copy()
    
    # Calculate penalty parameter based on diagonal scale of K
    diag_scale      = np.max(np.abs(np.diagonal(K)))
    penalty         = diag_scale * pen_factor
    
    for dof, val in zip(fixed_dofs, fixed_vals):
        K_bc[dof, dof]  += penalty
        F_bc[dof]       += penalty * val
        
    return K_bc, F_bc

def apply_bc_lagrange(K, F, fixed_dofs, fixed_vals):
    """
    Applies Dirichlet boundary conditions using Lagrange multipliers.
    Expands the system size by the number of constraints.
    
    [ K   C^T ] { u }   { F }
    [ C    0  ] { L } = { g }
    """
    num_nodes   = len(F)
    num_bc      = len(fixed_dofs)
    
    # Expand system matrices
    K_exp = np.zeros((num_nodes + num_bc, num_nodes + num_bc))
    F_exp = np.zeros(num_nodes + num_bc)
    
    # Copy original system into the top-left block
    K_exp[:num_nodes, :num_nodes]   = K
    F_exp[:num_nodes]               = F
    
    # Apply constraints in bottom row and right column
    for c, (dof, val) in enumerate(zip(fixed_dofs, fixed_vals)):
        K_exp[num_nodes + c, dof]   = 1.0
        K_exp[dof, num_nodes + c]   = 1.0
        F_exp[num_nodes + c]        = val
        
    return K_exp, F_exp
