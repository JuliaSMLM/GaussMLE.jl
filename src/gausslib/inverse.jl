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

"""
    matrix_inverse!(A, A_inv, n)

Calculate the inverse of a matrix `A` using LU decomposition and store result in `A_inv`.
Returns true if successful, false if matrix is singular.
"""
function matrix_inverse!(A, A_inv, n)
    try
        # Initialize A_inv as identity matrix
        fill!(A_inv, zero(eltype(A_inv)))
        for i = 1:n
            A_inv[i, i] = one(eltype(A_inv))
        end
        
        # Make a copy of A for decomposition (don't modify original)
        A_copy = copy(A)
        
        # Perform LU decomposition
        lu_decomposition!(A_copy)
        
        # Solve for each column
        for i = 1:n
            b = A_inv[:, i]
            A_inv[:, i] = back_substitution(A_copy, b)
        end
        
        return true
    catch
        return false
    end
end


