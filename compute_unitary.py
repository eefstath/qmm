# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,tags,-all
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compute Unitary Matrix

# %% [markdown]
# ## General Description
#
# This is a python programm that takes a matrix and makes it unitary based on its norm.
#
# Matrix is **NOT** given as input, rather than it is defined here

# %% [markdown]
# ## Instructions
#
# The operator norm, or spectral norm is the largest singular value of A.
#
# Singular values are the square roots of the eigenvalues of $A^{\dagger}A$
#
# Steps:
# - Compute $P=A^{\dagger}A$
# - Compute the eigenvalues of $P$.
# - Find the maximum eigenvalue.
# - Compute the square root of the maximum eigenvalue.
#
# Or oneliner:
# $||A||_{2} = \sqrt{\lambda_{\max}(A^{\dagger}A)}$
#
# This equals to the norm of A, which:
# - If less than 1: top left corner of the matrix is $C=A/||A||_{2}$
# - If greater than 1: top left corner of the matrix is $A$
#
# So TopLeft equals A or C.
# And the rest of the Unitary matrix U, equals to:
# $$ U =
# \begin{pmatrix}
# TopLeft & \sqrt{I - TopLeft^{\dagger} TopLeft} \\\\
# \sqrt{I - TopLeft TopLeft^{\dagger}} & - TopLeft^{\dagger}
# \end{pmatrix}
# $$

# %% [markdown]
# # Code

# %% [markdown]
# ## Imports, colors and helping functions

# %%
# Imports
import sys
from datetime import datetime
import numpy as np
from scipy.linalg import sqrtm

# %%
# Colors
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
RESET = '\033[0m'

# %%
# Logging function to print (at least debug) logs
#   Args:
#       level: log level [debug, info, error]
#       rest of args: message
def print_log(level, *args):
    # Set debug mode
    debug = False

    # Set timestamp format
    timestamp = datetime.now().strftime('%m-%d %H:%M:%S')
    
    # Set colors based on level
    if level == 'debug': color = YELLOW
    elif level == 'info': color = BLUE
    elif level == 'error': color = RED
    else: print_log('error', 'Invalid log level')

    # Get message
    message = ' '.join(map(str, args))
    if (level != 'debug') or (debug and level == 'debug'):
        print(f'[{timestamp}][{color}{level}{RESET}]\n{message}')

    # Exit if error
    if level == 'error': sys.exit(1)

# %%
# Function that calculates norm of a matrix
#   Args:
#       A: Matrix
def calc_norm(A):
    # Get conjugate of A
    A_conjugate = np.conjugate(A)
    print_log('debug', "Matrix A, conjugate:\n", A_conjugate)

    # Get transpose of A conjugate
    A_dagger = np.transpose(A_conjugate)
    print_log('debug', "Matrix A, transpose conjugate:\n", A_dagger)

    # Product of A_dagger and A
    P = np.dot(A_dagger, A)
    print_log('debug', "Matrix A, dagger dot product A conjugate:\n", P)

    # Get eigenvalues of P
    eigenvalues = np.linalg.eigvals(P)
    print_log('debug', "Eigenvalues of matrix P:\n", eigenvalues)

    # Get max eigenvalue
    max_eigenvalue = np.max(eigenvalues)
    print_log('debug', "Max eigenvalue of matrix P:\n", max_eigenvalue)

    # Get square root of max eigenvalue
    norm_A = np.sqrt(max_eigenvalue)

    print_log('info', "Norm of matrix A:\n", norm_A)
    return norm_A

# %%
# Function that calculates a unitary matrix that contains a NON unitary matrix A
#   in its top left corner
#   Args:
#       A: Non unitary matrix
#       norm_A: Norm of A
def calc_unitary(A, norm_A):

    # Get number of rows/columns of A (I guess they are equal)
    n = A.shape[0]
    print_log('debug', "Number of rows/columns of matrix A:\n", n)

    # Check if norm is lower or equal to 1
    if norm_A <= 1:
        # Top left corner will contain A
        top_left = A
    # Check if norm is greater than 1
    else:
        # Top left corner will contain C, which is A/norm_A
        top_left = A/norm_A
       
    # Calculate X_dagger@X and X@X_dagger (needed later on)
    # Where X is top left corner
    top_left_dagger = np.transpose(np.conjugate(top_left))
    top_left_x_top_left_dagger = np.dot(top_left, top_left_dagger)
    top_left_dagger_x_top_left = np.dot(top_left_dagger, top_left)
    print_log('debug', "Matrix top left:\n", top_left)

    # Bottom left 
    bottom_left = sqrtm(np.identity(n) - top_left_dagger_x_top_left)
    print_log('debug', "Matrix left right:\n", bottom_left)

    # Top right 
    top_right = sqrtm(np.identity(n) - top_left_x_top_left_dagger)
    print_log('debug', "Matrix top right:\n", top_right)

    # Bottom right
    bottom_right = -top_left_dagger
    print_log('debug', "Matrix bottom right:\n", bottom_right)

    # Unitary matrix U
    U = np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])

    print_log('info', "Unitary matrix U:\n", U)
    return U

# %%
# Validate Unitary
#    Args:
#        U: Unitary matrix
def validate_unitary(U):
    # Get number of rows/columns of U (I guess they are equal)
    n = U.shape[0]

    # Check if U is unitary by computing U@U_dagger
    U_dagger = np.transpose(np.conjugate(U))
    U_dagger_x_U = np.dot(U_dagger, U)

    # Check if U_dagger_x_U is identity matrix
    if np.allclose(U_dagger_x_U, np.identity(n)):
        print_log('info', "U is unitary")
    else:
        print_log('info', "U is not unitary")

# %% [markdown]
# ## Code Example 1
#
# Non Unitary Matrix example
#
# $$ A =
# \begin{pmatrix}
# 2-\sqrt{2} & 0 & 0 & 0 \\\\
# 0 & 2\sqrt{2} & 0 & 0 \\\\
# 0 & 0 & \sqrt{2} & 0 \\\\
# 0 & 0 & 0 & 2
# \end{pmatrix}
# $$

# %%
# Define A example
A = np.array([[2-np.sqrt(2), 0, 0, 0],
              [0, 2*np.sqrt(2), 0, 0],
              [0, 0, 2*np.sqrt(2), 0],
              [0, 0, 0, 2]])
print_log('debug', "Matrix A:\n", A)


# %%
# Calculate norm of A for a non unitary matrix
normA = calc_norm(A)

# %%
# Calculate unitary matrix for a non unitary matrix
U = calc_unitary(A, normA)

# %%
# Validate Unitary
validate_unitary(U)

# %% [markdown]
# ## Code Example 2 (Pennylane)
#
# https://pennylane.ai/qml/demos/tutorial_intro_qsvt
#
# Non Unitary Matrix example
#
# $$ B =
# \begin{pmatrix}
# 0.1 & 0.2 \\\\
# 0.3 & 0.4
# \end{pmatrix}
# $$

# %%
# All in one:
# Define B example
B = np.array([[0.1, 0.2],
              [0.3, 0.4]])
print_log('debug', "Matrix B:\n", B)

# Calculate norm of B for a non unitary matrix
normB = calc_norm(B)

# Calculate unitary matrix for a non unitary matrix
U = calc_unitary(B, normB)

# Validate Unitary
validate_unitary(U)
