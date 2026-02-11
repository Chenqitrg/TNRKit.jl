@info "LoopTNR ising CFT data"

using JLD2
T = classical_ising_symmetric()
scheme = LoopTNR(T)

function cft_data_different_geometry(scheme::LoopTNR)
    cft22 = cft_data(scheme, [2, 2, 0]; Nh = 40)
    cft24 = cft_data(scheme, [sqrt(2), 2 * sqrt(2), 0]; Nh = 40)
    return cft22, cft24
end

data = run!(scheme, truncdim(16), truncbelow(1e-14), maxiter(50), maxiter(100), maxiter(50), Finalizer(cft_data_different_geometry);; nuclear_norm_regularization=true, verbosity = 4, tol_loop = 1.0e-14)

# for shape in [[2, 2, 0]]
#     cft = cft_data(scheme, shape)
#     d1, d2 = real(cft[Z2Irrep(1)][1:23]), real(cft[Z2Irrep(0)][1:23])
#     @info "Obtained lowest scaling dimensions:\n$(d1), $(d2)."
# end

@save "/Users/chenqimeng/Downloads/very_high_scaling_dimension.jld2" data

