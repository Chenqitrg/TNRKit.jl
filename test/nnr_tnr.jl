@info "LoopTNR ising CFT data"
T = classical_ising_symmetric()
scheme = LoopTNR(T)
run!(scheme, truncdim(16), maxiter(15); nuclear_norm_regularization=true, verbosity = 3)

for shape in [[1, 4, 1], [sqrt(2), 2 * sqrt(2), 0]]
    cft = cft_data(scheme, shape)
    d1, d2 = real(cft[Z2Irrep(1)][1:10]), real(cft[Z2Irrep(0)][1:10])
    @info "Obtained lowest scaling dimensions:\n$(d1), $(d2)."
end
