using LinearAlgebra

function conj_grad(A, x, b, epsilon, iter_max)
    r = b - A*x
    err = norm(r)
    if err < epsilon * norm(x)
        @show "Exiting early"
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
            @show "Exiting early step" 
            @show k
            #return #x
            return x
        end
        beta = (r' * r) / (r_old' * r_old)
        p = r + beta * p
    end
    return x
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
#    x = conj_grad(A, x, b, 1e-9, N^2)
#    #@show x
#    @show norm(x - xexact)
#end
