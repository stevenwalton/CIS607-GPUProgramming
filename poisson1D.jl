include("my_solvers.jl")

using LinearAlgebra
using SparseArrays
#using Plots
using Printf
using CUDA

function get_error(solution, input, step)
    u_total = [0; input; 0]
    return sqrt((u_total - solution)' * (u_total - solution)) * sqrt(step)
end

function poisson(h, eps)
    N = Integer(1/h + 1) # Total number of nodes
    m = N - 2 # total interior nodes
    x0 = zeros(m)
    x = 0:h:1
    exact = sin.(pi*x)
    A = zeros(m, m)
    for i = 1:m
        A[i, i] = 2/h^2
    end

    for i = 1:m-1
        A[i, i+1] = -1/h^2
        A[i+1, i] = -1/h^2
    end

    # make vector b
    b = Array{Float32}(undef, m)
    for i = 1:m
        b[i] =  pi^2 * sin(pi * x[i]) 
    end

    u_int = conj_grad(A, x0, b, eps, N^2)
    error = get_error(exact, u_int, h)

    # Half
    hd2 = h/2
    N_half = Integer(1 / hd2 + 1)
    m_half = N_half - 2
    x0_half = zeros(m_half)
    x_half = 0:hd2:1
    exact_half = sin.(pi * x_half)
    A_half = zeros(m_half, m_half)
    for i = 1:m_half
        A_half[i, i] = 2/hd2^2
    end
    for i = 1:m_half-1
        A_half[i, i+1] = -1/hd2^2
        A_half[i+1, i] = -1/hd2^2
    end

    # make vector b
    b_half = Array{Float32}(undef, m_half)
    for i = 1:m_half
        b_half[i] =  pi^2 * sin(pi * x_half[i]) 
    end

    u_int_half = conj_grad(A_half, x0_half, b_half, eps, N_half^2)
    error_half = get_error(exact_half, u_int_half, hd2)

    @printf "%f\t %f\t %f\t %f\n" h error (error/error_half) log2(error)
end

function fillA!(A, h)
    m = size(A)[1]
    for i = 1:m
        A[i, i] = 2 / h^2
    end
    for i = 1:m-1
        A[i, i+1] = -1 / h^2
        A[i+1, i] = -1 / h^2
    end
    return nothing
end

function get_error_cuda!(error, solution, input, step)
    u_total = CuArray([0; input; 0])
    error = CUDA.sqrt((u_total - solution)' * (u_total - solution)) * sqrt(step) 
    return nothing
end

function test_cuda!(h, x0, eps)
    N = Integer(1 / h + 1)
    m = N - 2
    #x0 = zeros(m)
    #x_host = 0:h:1
    #x = x_host |> cu
    #exact = sin.(pi * (x_host))
    return nothing
end


function cuda_poisson!(h, eps)
    @show "Entering cuda poisson"
    N = Integer(1/h + 1) # Total number of nodes
    m = N - 2 # total interior nodes
    x0 = CUDA.zeros(m)
    x_host = 0:h:1
    #x = x_host |> cu
    exact = sin.(pi* (0:h:1))

    A = CUDA.zeros(m, m)
    @cuda threads=256 fillA!(A, h)
    #@cuda fillA!(A, h)
    #@show "Filled A"

    ## make vector b
    #b_host = zeros(m)
    #for i = 1:m
    #    b_host[i] =  pi^2 * sin(pi * x_host[i]) 
    #end
    #b = b_host |> cu
    #@show "Made b"
    #

    ## Make sure everything is caught up
    #synchronize()

    #@show "Made it to conj grad"
    #u_int = conj_grad_cuda(A, x0, b, eps, N^2)

    ## Make sure everything is caught up
    #synchronize()

    #error = get_error(exact, u_int, h)

    ### Half
    ##hd2 = h/2
    ##N_half = Integer(1 / hd2 + 1)
    ##m_half = N_half - 2
    ##x0_half = CUDA.zeros(m_half)
    ##x_half = CuArray{Float64}(0:hd2:1)
    ##exact_half = sin.(pi * (0:hd2:1))
    ##A_half = CUDA.zeros(m_half, m_half)
    ###@cuda threads=256 fillA!(A_half, m)
    ##@cuda threads=256 fillA!(m_half, h_half)

    ### make vector b
    ###b_half = CuArray{Float64}(undef, m_half)
    ##b_host = zeros(m_half)
    ##for i = 1:m_half
    ##    b_host[i] =  pi^2 * sin(pi * x_half[i]) 
    ##end
    ##b_half = b_host |> cu
    ### Make sure everything is caught up
    ##synchronize()

    ##u_int_half = conj_grad_cuda(A_half, x0_half, b_half, eps, N_half^2)

    ### Make sure everything is caught up
    ##synchronize()

    ##error_half = get_error(exact_half, u_int_half, hd2)

    ##@printf "%f\t %f\t %f\t %f\n" h error (error/error_half) log2(error)
    return nothing
end

eps = 1e-9
h_list = [0.1, 0.05, 0.025, 0.0125]

@printf "dx\t\t error_dx\t ratio\t\t rate\n"
@time begin
    for i = 1:length(h_list)
        h = h_list[i]
        poisson(h, eps)
    end
end

@time begin
    for i = 1:length(h_list)
        h = h_list[i]
        N = Integer(1 / h + 1)
        m = N-2
        x0 = CUDA.zeros(m)
        x_host = 0:h:1
        #x = x_host |> cu
        exact = sin.(pi * x_host)
        A = CUDA.zeros(m, m)
        @cuda fillA!(A, h)
        b_host = zeros(m)
        for j = 1:m
            b_host[j] = pi^2 * sin(pi * x_host[j])
        end
        b = b_host |> cu
        synchronize()
        conj_grad_cuda!(A, x0, b, CuArray([eps]), N^2)
        error = get_error(exact, Array(x0), h)

        # half calculation
        h_half = h / 2
        N = Integer(1 / h_half + 1)
        m = N-2
        x0_half = CUDA.zeros(m)
        x_host = 0:h_half:1
        x = x_host |> cu
        exact = sin.(pi * (0:h_half:1))
        A = CUDA.zeros(m, m)
        @cuda fillA!(A, h_half)
        b_host = zeros(m)
        for j = 1:m
            b_host[j] = pi^2 * sin(pi * x_host[j])
        end
        b = b_host |> cu
        synchronize()
        conj_grad_cuda!(A, x0_half, b, CuArray([eps]), N^2)
        error_half = get_error(exact, Array(x0_half), h_half)
        @printf "%f\t %f\t %f\t %f\n" h error (error/error_half) log2(error)
    end
end
