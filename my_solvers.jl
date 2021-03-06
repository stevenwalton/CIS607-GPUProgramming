using LinearAlgebra
using CUDA

function conj_grad(A, x, b, epsilon, iter_max)
    r = b - A*x
    err = norm(r)
    if err < epsilon * norm(x)
        #@show "Exiting too early"
        return x
    end
    p = r
    for k = 1:iter_max
        #@show x
        r_old = r
        alpha = (r' * r) / (p' * A * p)
        # .= so we redefine x
        x .= x + alpha * p
        r = r - alpha * A * p
        err = norm(r) 
        if err < epsilon * norm(x)
            #@show "Exiting early step" 
            #@show k
            #return #x
            return x
        end
        beta = (r' * r) / (r_old' * r_old)
        p = r + beta * p
    end
    return x
end

function norm_cuda!(err, x)
    for i = 1:length(x)
        err[1] = err[1] + (x[i] * x[i])
    end
    err = CUDA.sqrt(err[1])
    return nothing
end

function gemv!(y, A, x)

    N = length(y)

    for i = 1:N
        for k = 1:N
            y[i] += A[i, k]*x[k]
        end
    end
end

function knl_gemv!(y, A, x)

    N = length(y)

    bid = blockIdx().x  # get the thread's block ID
    tid = threadIdx().x # get my thread ID
    dim = blockDim().x  # how many threads in each block

    i = dim * (bid - 1) + tid #unique global thread ID
        if i <= N
            for k = 1:N
                y[i] += A[i, k]*x[k]
            end
        end
    return nothing
end

function conj_grad_cuda!(A, x, b, epsilon, iter_max, dummy)
    threads_pb=32
    num_blocks = 5#cld(N, threads)
    @cuda threads=threads_pb blocks=num_blocks knl_gemv!(dummy, A, x)
    #r = b - A * x
    r = b - dummy
    err = CuArray([0.])
    @cuda norm_cuda!(err, r)
    err2 = CuArray([0.])
    @cuda norm_cuda!(err2, x)
    if any(err .< epsilon .* err2)
        return #x
    end
    p = copy(r)
    for k = 1:iter_max
        r_old = copy(r)
        #alpha = (r' * r) / (p' * A * p)
        @cuda threads=threads_pb blocks=num_blocks knl_gemv!(dummy, A, p)
        alpha = (r' * r) / (p' * dummy)
        x .= x + alpha * p
        @cuda norm_cuda!(err, r)
        @cuda norm_cuda!(err2, x)
        if any(err .< epsilon .* err2)
            return #x
        end
        beta = (r' * r) / (r_old' * r_old)
        p = r + beta * p
    end
    ##return x
    return 
end

#let 
#    N = 20
#    B = rand(N, N)
#    A = Matrix(I, N, N) + B'*B
#    e = eigen(A)
#
#    b = rand(N);
#    x = zeros(N);
#    xexact = A\b # LU decomp
#    #@show xexact
#
#    #x = conj_grad(A, x, b, 1e-9, N^2)
#    A_cuda = CUDA.zeros(N, N)
#    for i = 1:N
#        A_cuda[i, i] = 2/0.01
#    end
#    x_cuda = CUDA.zeros(N)
#    b_cuda = CUDA.ones(N)
#    x = conj_grad_cuda(A_cuda, x_cuda, b_cuda, 1e-9, N^2)
#    #@show x
#    @show norm(x - xexact)
#end
#
