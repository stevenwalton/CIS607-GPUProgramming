using CUDA

function fillA!(A)
    len = size(A)[1]
    for i = 1:len
        A[i, i] = 1
    end
    @cuprint(A)
    return nothing
end

m = 5
#A = zeros(m, m)
A = CUDA.zeros(m, m)
display(A)
@cuda fillA!(A)
@show "Modified"
display(A)
