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

function norm_cuda(x)
    sum = CUDA.zeros(length(x))
    for i = 1:length(x)
        sum[i] = x[i] * x[i]
    end
    return sqrt(reduce(+, sum))
end

function conj_grad_cuda(A, x, b, epsilon, iter_max)
    r = b - A * x
    err = norm_cuda(r)
    if err < epsilon * norm_cuda(x)
        return x
    end
    p = copy(r)
    for k = 1:iter_max
        r_old = copy(r)
        alpha = (r' * r) / (p' * A * p)
        x .= x + alpha * p
        err = norm_cuda(r)
        if err < epsilon * norm_cuda(x)
            return x
        end
        beta = (r' * r) / (r_old' * r_old)
        p = r + beta * p
    end
    return x
end

let 
    N = 20
    B = rand(N, N)
    A = Matrix(I, N, N) + B'*B
    e = eigen(A)

    b = rand(N);
    x = zeros(N);
    xexact = A\b # LU decomp
    #@show xexact

    #x = conj_grad(A, x, b, 1e-9, N^2)
    A_cuda = CUDA.zeros(N, N)
    for i = 1:N
        A_cuda[i, i] = 2/0.01
    end
    x_cuda = CUDA.zeros(N)
    b_cuda = CUDA.ones(N)
    x = conj_grad_cuda(A_cuda, x_cuda, b_cuda, 1e-9, N^2)
    #@show x
    @show norm(x - xexact)
end

