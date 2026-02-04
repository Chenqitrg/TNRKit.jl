function weighted_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc_dim::Int; method="svt", maximum_steps=100, rtol=1.0e-12, verbosity=1)
    if method == "svt"
        spec, _ = eigsolve(
            PhidPhi, X0, 1, :LM; krylovdim=5, maxiter=100,
            tol=1.0e-12,
            verbosity=0
        )
        Lipschitz_const = real(spec[1])
        eta = 1 / Lipschitz_const * 0.99

        Ss, _ = eigsolve(x -> X0' * X0 * x, ones(domain(X0)), 1, :LR)

        return svt_low_rank(PhidPhi, PhidY, X0, trunc_dim, maximum_steps, rtol, verbosity, eta, sqrt(abs(Ss[1])))
    elseif method == "fact"
        return factorized_low_rank(PhidPhi, PhidY, X0, trunc_dim, maximum_steps, rtol, verbosity)
    end
end

function svt_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc_dim::Int, maximum_steps::Int, rtol::Float64, verbosity::Int, eta::Float64, tau::Float64)
    X_update = copy(X0)
    rank_this = 0
    cost_const = tr(PhidY * X0')
    for step in 1:maximum_steps
        gradient_step = X_update - eta * (PhidPhi(X_update) - PhidY)
        X_new, rank_this = svt(gradient_step, tau)
        error = (tr(X_new' * PhidPhi(X_new)) - 2 * real(tr(PhidY * X_new')) + cost_const) / cost_const

        (rank_this > trunc_dim) && return X_update, rank_this, error

        if verbosity > 1
            @infov 3 "Step $step: rank = $rank_this, error = $error, tau = $tau"
        end

        (error < rtol) && return X_new, rank_this, error
        (rank_this < trunc_dim) && (tau *= (1.0 - 0.2 * (trunc_dim - rank_this + 0.001) / trunc_dim))
        X_update = X_new
    end

    return X_update, rank_this, error
end

function factorized_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc_dim::Int, maximum_steps::Int, rtol::Float64, verbosity::Int)
    # monitor cost function return X_fix 
    rank_this = 0
    cost_const = tr(PhidY * X0')
    X_update = copy(X0)
    error = (tr(X_update' * PhidPhi(X_update)) - 2 * real(tr(PhidY * X_update')) + cost_const) / cost_const
    @show error
    U, V = SVD12(X_update, truncdim(trunc_dim))
    X_update = U * V
    error = (tr(X_update' * PhidPhi(X_update)) - 2 * real(tr(PhidY * X_update')) + cost_const) / cost_const
    if verbosity > 1
        @infov 3 "Initial SVD cost = $error"
    end
    for step in 1:maximum_steps
        # U1, S1, V1 = tsvd(X_update; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim))
        rank_this = dim(domain(U))
        U_new, info = linsolve(x -> (PhidPhi(x * V) * V'), PhidY * V', U; krylovdim=20, maxiter=20, tol=1.0e-12, verbosity=0)

        # U2, S2, V2 = tsvd(US_new * V1; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim))


        V_new, info = linsolve(x -> (U_new' * PhidPhi(U_new * x)), U_new' * PhidY, V; krylovdim=20, maxiter=20, tol=1.0e-12,
            verbosity=0)

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

_threshold(S::Vector, tau::Float64) = map(s -> max(s - tau, 0), S)

function svt(T::TensorMap, tau::Float64)
    U, S, V = tsvd(T; alg=TensorKit.SVD())

    thresholded_S = map(s -> max(s - tau, 0), S.data)
    rank_reduced = count(s -> s > 0, thresholded_S)
    new_S = DiagonalTensorMap(thresholded_S, domain(S, 1))
    return U * new_S * V, rank_reduced
end