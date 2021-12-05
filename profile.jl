using CUDA
using Test
using BenchmarkTools

N = 2^20;
x_d = CUDA.fill(1.0f0, N);
y_d = CUDA.fill(2.0f0, N);

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

# bench_gpu1!(y_d, x_d)
# CUDA.@profile bench_gpu1!(y_d, x_d)

function gpu_add2!(y, x)
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

# fill!(y_d, 2)
# @cuda threads=256 gpu_add2!(y_d, x_d)
# @test all(Array(y_d) .== 3.0f0)

function bench_gpu2!(y, x)
    CUDA.@sync begin
        @cuda threads=256 gpu_add2!(y, x)
    end
end

# @btime bench_gpu2!($y_d, $x_d)
#CUDA.@profile bench_gpu2!(y_d, x_d)

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds y[index] += x[index]
    return
end

numblocks = ceil(Int, N/256)
fill!(y_d, 2)

function bench_gpu3!(y, x)
    numblocks = ceil(Int, length(y) / 256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks gpu_add3!(y, x)
    end
end

CUDA.@profile bench_gpu3!(y_d, x_d)
