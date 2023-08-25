"""
    lu_decomposition!(A)

Perform LU decomposition on a square matrix `A` in-place.
"""
function lu_decomposition!(A)
    n = size(A, 1)
    for k = 1:n
        for i = k+1:n
            A[i, k] /= A[k, k]
            for j = k+1:n
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end
    return A
end

"""
    back_substitution(A, b)

Perform back-substitution to solve Ax = b, where `A` is an LU-decomposed matrix.
Returns the solution vector `x`.
"""
function back_substitution(A, b)
    n = size(A, 1)
    x = similar(b)
    y = similar(b)

    # Forward substitution for Ly = b
    for i = 1:n
        y[i] = b[i]
        for j = 1:i-1
            y[i] -= A[i, j] * y[j]
        end
    end

    # Backward substitution for Ux = y
    for i = n:-1:1
        x[i] = y[i]
        for j = i+1:n
            x[i] -= A[i, j] * x[j]
        end
        x[i] /= A[i, i]
    end

    return x
end

"""
    matrix_inverse(A)

Calculate the inverse of a matrix `A` using LU decomposition and back-substitution.
"""
function matrix_inverse(A)
    n = size(A, 1)
    A_inv = Array{Float32}(I, n, n)  # Initialize as identity matrix

    # Perform LU decomposition
    lu_decomposition!(A)

    # Solve for each column
    for i = 1:n
        b = A_inv[:, i]
        A_inv[:, i] = back_substitution(A, b)
    end

    return A_inv
end

# Test the functions
A = rand(3, 3)  # Random 3x3 matrix
A_copy = copy(A)  # Keep a copy for verification

# Calculate the inverse
A_inv = matrix_inverse(A)

# Verify the solution
identity_approx = A_copy * A_inv

println("A * A_inv should be approximately the identity matrix:")
display(Array(identity_approx))

