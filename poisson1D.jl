include("my_solvers.jl")

using LinearAlgebra
using SparseArrays
using Plots

h = 0.1
N = Integer(1/h + 1) # Total number of nodes
m = N - 2 # total interior nodes

x = 0:h:1
A = zeros(m, m)
for i = 1:m
    A[i, i] = 2/h^2
end

for i = 1:m-1
    A[i, i+1] = -1/h^2
    A[i+1, i] = -1/h^2
end

# make vector b
b = Array{Float64}(undef, m)
for i = 1:m
    b[i] =  pi^2 * sin(pi * x[i]) 
end
# form x0
x0 = zeros(m);

#(u_int, no_iters) = conj_grad(A, x0, b, 1e-6, N^2)
u_int = conj_grad(A, x0, b, 1e-9, N^2)
#@show u_int

exact = sin.(pi*x)
u_total = [0;u_int;0] 
#plot(x, u_total)
#plot!(x, exact)
error = sqrt((u_total - exact)'*(u_total - exact)) * sqrt(h)
@show error
