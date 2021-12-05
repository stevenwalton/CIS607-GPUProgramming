using CUDA

function add_vecs!(z, x, y)

	N = length(x)
	M = length(y)

	@assert N == M
	
	for i = 1:N
		z[i] = x[i] + y[i]
	end
end

function fake_knl_addvecs!(z, x, y, num_threads_per_block, num_blocks)

	dim = num_threads_per_block
	N = length(x)
	for bid = 1:num_blocks
		for tid = 1:num_threads_per_block
			#Specify the work that thread tid on block bid is 
			#going to do. 

			i = dim * (bid-1) + tid #unique globalk thread index
			
			if i <= N 
				z[i] = x[i] + y[i]
			end
		end
	end
end

function knl_addvecs!(z, x, y)
	N = length(x)

	bid = blockIdx().x  # get the thread's block ID
	tid = threadIdx().x # get my thread ID
	dim = blockDim().x  # how many threads in each block
	
	i = dim * (bid-1) + tid #unique globalk thread index

	if i <= N
		z[i] = x[i] + y[i]
	end

	return nothing
end






# 1e9 crashes
N = Integer(1e8)
# Host arrays
h_z = zeros(N)
h_x = rand(N)
h_y = rand(N)
z_cpu = zeros(N)


num_threads_per_block = 64
num_blocks = cld(N, num_threads_per_block)


#fake_knl_addvecs!(z, x, y, num_threads_per_block, num_blocks)

println("Addvecs CPU")
@time add_vecs!(h_z, h_x, h_y)

println("Fake kernel Addvecs")
@time fake_knl_addvecs!(h_z, h_x, h_y, num_threads_per_block, num_blocks)


println("GPU Addvecs")
@time begin
    # Device arrays
    d_z = CuArray(h_z)
    d_x = CuArray(h_x)
    d_y = CuArray(h_y)
    @cuda threads=num_threads_per_block blocks=num_blocks knl_addvecs!(d_z, d_x, d_y)
    synchronize()
end

println("Native CUDA version")
@time begin
    d_z = CuArray(h_z)
    d_x = CuArray(h_x)
    d_y = CuArray(h_y)
    d_z .= d_x .+ d_y
    synchronize()
end

z_cpu .= d_z
add_vecs!(h_z, h_x, h_y)
@assert isapprox(h_z, z_cpu)
