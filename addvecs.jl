

function add_vecs!(z, x, y)
	N = length(x)
	M = length(y)
	@assert N == M
	for i = 1:N
		z[i] = x[i] + y[i]
        end
end

function fake_kn1_addvecs!(z, x, y, num_threads_per_block, num_blocks)
    dim = num_threads_per_block
    N = length(x)
    for bid = 1:num_blocks
        for tid = 1:num_threads_per_block
            # Specify work that thread tid on block bid does
            i = dim * (bid - 1) + tid
            if i <= N
                z[i] = x[i] + y[i]
            end
        end
    end
end

function kn1_addvecs!(z, x, y)
    N = length(x)
    bid = blockIdx().x # get thread's block ID
    tid = threadIdx().x # get thread ID
    dim = blockDim().x # How many threads per block

    i = dim * (bid - 1) + tid
    if i <= N
        z[i] = x[i] + y[i]
    end
    return nothing
end

N = 1000
z = zeros(N)
x = rand(N)
y = rand(N)

#add_vecs!(z, x, y)

num_threads_per_block = 64
num_blocks = cld(N, num_threads_per_block)

fake_kn1_addvecs!(z, x, y, num_threads_per_block, num_blocks)
@show z
