"""
Test Cholesky implementation correctness
"""

using LinearAlgebra
using StaticArrays

# Copy the implementation
@inline function static_cholesky_decomposition!(A::MMatrix{N,N,T}) where {N,T}
    @inbounds for j = 1:N
        for i = j:N
            sum_val = A[i, j]
            for k = 1:j-1
                sum_val -= A[i, k] * A[j, k]
            end

            if i == j
                if sum_val <= zero(T)
                    return false
                end
                A[i, i] = sqrt(sum_val)
            else
                A[i, j] = sum_val / A[j, j]
            end
        end
    end
    return true
end

@inline function static_cholesky_inverse!(A_inv::MMatrix{N,N,T}, L::MMatrix{N,N,T}) where {N,T}
    # First invert L (lower triangular)
    L_inv = MMatrix{N,N,T}(undef)
    @inbounds for j = 1:N
        L_inv[j, j] = one(T) / L[j, j]
        for i = j+1:N
            sum_val = zero(T)
            for k = j:i-1
                sum_val += L[i, k] * L_inv[k, j]
            end
            L_inv[i, j] = -sum_val / L[i, i]
        end
    end

    # A_inv = L_inv^T * L_inv
    @inbounds for i = 1:N
        for j = i:N
            sum_val = zero(T)
            for k = j:N
                sum_val += L_inv[k, i] * L_inv[k, j]
            end
            A_inv[i, j] = sum_val
            if i != j
                A_inv[j, i] = sum_val
            end
        end
    end
    return true
end

# Test with a simple symmetric positive definite matrix
println("Testing Cholesky implementation")
A_test = Float32[
    4.0  1.0  2.0
    1.0  5.0  1.0
    2.0  1.0  6.0
]

# Reference: Julia's built-in
A_inv_ref = inv(A_test)
println("\nReference inverse (Julia inv()):")
display(A_inv_ref)

# Our implementation
A_mm = MMatrix{3,3,Float32}(A_test)
A_chol = MMatrix{3,3,Float32}(A_test)
A_inv_our = MMatrix{3,3,Float32}(undef)

if static_cholesky_decomposition!(A_chol) && static_cholesky_inverse!(A_inv_our, A_chol)
    println("\n\nOur inverse:")
    display(Matrix(A_inv_our))

    println("\n\nDifference:")
    diff = Matrix(A_inv_our) - A_inv_ref
    display(diff)

    println("\n\nMax error: $(maximum(abs.(diff)))")

    if maximum(abs.(diff)) < 1e-5
        println("✓ Cholesky implementation is CORRECT")
    else
        println("✗ Cholesky implementation has ERRORS")
    end
else
    println("✗ Cholesky decomposition FAILED")
end
