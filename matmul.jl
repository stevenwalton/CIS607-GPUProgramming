using CUDA

# Kernels that perform matrix multiplication for with matrix A
# For dense matrix first (heat equation is tri-diagonal though)

function matmul!(C, A, B)
    (M, N) = size(A)
    (P, Q) = size(B)
    @assert N == P
    @assert (M, Q) == size(C)

    for i = 1:M
        for j = 1:Q
            # Inner product which is totally another for loop
            # ' gives transpose
            C[i, j] = A[i, :]' * B[:, j] 
        end
    end

end

function fake_knl_matmul!(C, A, B, numx, numy, numblocks_x, numblocks_y)
    # numx = number of threadsd per block in x direction
    # numblocks_x = number of thread blocks in the x direction
    (M, N) = size(A)
    (P, Q) = size(B)
    @assert N == P
    @assert (M, Q) == size(C)

    for bidx = 1:numblocks_x
        for bidy = 1:numblocks_y
            for tidx = 1:numx
                for tidy = 1:numy
                    # Now on a single thread
                    # Unique global thread ID in {x,y} direction
                    i = numblocks_x * (bidx - 1) + tidx
                    j = numblocks_y * (bidy - 1) + tidy
                    # Make sure have data to work on 
                    #if i <= M && j <= Q
                        C[i, j] = A[i, :]' * B[:, j]
                    #end # If 
                end # tidy
            end # tidx
        end # bidy
    end # bidx
end # Function

M = 1000
N = 2000
P = N
Q = 500

A = rand(M, N)
B = rand(P, Q)
C = zeros(M, Q)

matmul!(C, A, B)

C0 = A * B
@assert isapprox(C0, C)

# Fake GPU
C1 = zeros(M, Q)
numx = 32 # some multiple of 32
numy = 32 # some multiple of 32
numblocks_x = cld(M, numx)
numblocks_y = cld(Q, numy)
fake_knl_matmul!(C1, A, B, numx, numy, numblocks_x, numblocks_y)
@assert isapprox(C0, C1)
