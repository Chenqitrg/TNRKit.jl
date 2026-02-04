function weighted_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, Y::TensorMap, trunc_dim::Int; method="svt", maximum_steps=100, rtol=1.0e-12, verbosity=1)
    # if method == "svt"
    #     spec, _ = eigsolve(
    #         PhidPhi, X0, 1, :LM; krylovdim=5, maxiter=100,
    #         tol=1.0e-12,
    #         verbosity=0
    #     )
    #     Lipschitz_const = real(spec[1])
    #     eta = 1 / Lipschitz_const * 0.9

    #     Ss, _ = eigsolve(x -> X0' * X0 * x, ones(domain(X0)), 1, :LR)

    #     return svt_low_rank(PhidPhi, PhidY, X0, Y, trunc_dim, maximum_steps, rtol, verbosity, eta, sqrt(abs(Ss[1])))
    if method == "svt"
        return svt_low_rank(PhidPhi, PhidY, X0, Y, trunc_dim, maximum_steps, rtol, verbosity; soft = true)
    elseif method == "svd"
        return svt_low_rank(PhidPhi, PhidY, X0, Y, trunc_dim, maximum_steps, rtol, verbosity; soft = false)
    elseif method == "fact"
        return factorized_low_rank(PhidPhi, PhidY, X0, Y, trunc_dim, maximum_steps, rtol, verbosity)
    end
end

function rank_tuning_function(trunc_dim::Int, step::Int; A=-1.0)
    return Int64(floor(A / (step - A / trunc_dim) + trunc_dim) + 1)
end

# function svt_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, Y::TensorMap, trunc_dim::Int, maximum_steps::Int, rtol::Float64, verbosity::Int, eta::Float64, tau::Float64)
#     X_update = copy(X0)
#     rank_this = 0
#     cost_const = tr(PhidY * Y')
#     error = (tr(X0' * PhidPhi(X0)) - 2 * real(tr(PhidY * X0')) + cost_const) / cost_const
#     @show error
#     for step in 1:maximum_steps
#         gradient_step = X_update - 2 * eta * (PhidPhi(X_update) - PhidY)
#         U, S_new, V, rank_this, tau = svt(gradient_step, trunc_dim)
#         X_new = U * S_new * V
#         error = (tr(X_new' * PhidPhi(X_new)) - 2 * real(tr(PhidY * X_new')) + cost_const) / cost_const

#         (rank_this > trunc_dim) && return X_update, rank_this, error

#         if verbosity > 1
#             @infov 3 "Step $step: rank = $rank_this, error = $error, tau = $tau"
#         end

#         (error < rtol) && return X_new, rank_this, error
#         X_update = X_new
#     end

#     return X_update, rank_this, error
# end

function svt_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, Y::TensorMap, trunc_dim::Int, maximum_steps::Int, rtol::Float64, verbosity::Int; soft = true)
    X_update = copy(X0)
    cost_const = tr(PhidY * Y')
    error = Inf
    rank_this = trunc_dim
    for step in 1:maximum_steps
        U1, S1, V1 = svt(X_update, trunc_dim; soft = soft)
        US_new, info = linsolve(x -> (PhidPhi(x * V1) * V1'), PhidY * V1', U1 * S1; krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)
        U2, S2, V2 = svt(US_new * V1, trunc_dim; soft = soft)
        SV_new, info = linsolve(x -> (U2' * PhidPhi(U2 * x)), U2' * PhidY, S2 * V2; krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)

        rank_this = dim(domain(S2))

        X_update = U2 * SV_new

        error = (tr(X_update' * PhidPhi(X_update)) - 2 * real(tr(PhidY * X_update')) + cost_const) / cost_const

        if verbosity > 1
            @infov 3 "Step $step: error = $error"
        end

        (error < rtol) && return X_update, rank_this, error
    end

    return X_update, rank_this, error
end

function factorized_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, Y::TensorMap, trunc_dim::Int, maximum_steps::Int, rtol::Float64, verbosity::Int)
    rank_this = 0
    cost_const = tr(PhidY * Y')
    U, V = SVD12(X0, truncdim(trunc_dim))
    X_update = U * V
    error = (tr(X_update' * PhidPhi(X_update)) - 2 * real(tr(PhidY * X_update')) + cost_const) / cost_const
    if verbosity > 1
        @infov 3 "Initial SVD cost = $error"
    end
    for step in 1:maximum_steps
        rank_this = dim(domain(U))
        U_new, info = linsolve(x -> (PhidPhi(x * V) * V'), PhidY * V', U; krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)
        V_new, info = linsolve(x -> (U_new' * PhidPhi(U_new * x)), U_new' * PhidY, V; krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)

        X_update = U_new * V_new

        error = (tr(X_update' * PhidPhi(X_update)) - 2 * real(tr(PhidY * X_update')) + cost_const) / cost_const

        if verbosity > 1
            @infov 3 "Step $step: error = $error"
        end

        (error < rtol) && return X_update, rank_this, error

        U = U_new
        V = V_new
    end

    return X_update, rank_this, error
end

# function svt(T::TensorMap, tau::Float64)
#     U, S, V = tsvd(T; alg=TensorKit.SVD())

#     thresholded_S = map(s -> max(s - tau, 0), S.data)
#     rank_reduced = count(s -> s > 0, thresholded_S)
#     new_S = DiagonalTensorMap(thresholded_S, domain(S, 1))
#     return U, new_S, V, rank_reduced, tau
# end

function svt(T::TensorMap, rank_reduced::Int; soft = true)
    if !soft
        U, S, V = tsvd(T; alg=TensorKit.SVD(), trunc=truncdim(rank_reduced) & truncbelow(1e-14))
        return U, S, V, rank_reduced, Inf
    end
    U, S, V = tsvd(T; alg=TensorKit.SVD(), trunc=truncdim(rank_reduced + 1) & truncbelow(1e-14))
    tau = S.data[end]
    thresholded_S = map(s -> max(s - tau, 0), S.data)
    rank_reduced = count(s -> s > 0, thresholded_S)
    new_S = DiagonalTensorMap(thresholded_S, domain(S, 1))
    return U, new_S, V, rank_reduced, tau
end