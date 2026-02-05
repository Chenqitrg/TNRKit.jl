function gradient_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, Y::TensorMap, trunc_dim::Int, maximum_steps::Int, rtol::Float64, verbosity::Int)
    spec, _ = eigsolve(
        PhidPhi, ones(space(X0)), 1, :LM; krylovdim=5, maxiter=100,
        tol=1.0e-12,
        verbosity=0
    )
    Lipschitz_const = real(spec[1])
    eta = 1 / Lipschitz_const * 0.9

    X_update = copy(X0)
    cost_const = tr(PhidY * Y')
    error = (tr(X0' * PhidPhi(X0)) - 2 * real(tr(PhidY * X0')) + cost_const) / cost_const

    for step in 1:maximum_steps
        X_gradient_step = X_update - 2 * eta * (PhidPhi(X_update) - PhidY)
        U, S_new, V, _, tau = svt(X_gradient_step, trunc_dim)
        X_new = U * S_new * V
        error_new = (tr(X_new' * PhidPhi(X_new)) - 2 * real(tr(PhidY * X_new')) + cost_const) / cost_const

        if verbosity == 4
            @infov 4 "Step $step: rank = $trunc_dim, error = $error_new, tau = $tau"
        end

        (error_new < rtol) && return X_new, trunc_dim, error_new
        (abs(error - error_new) / error < 1e-9) && return X_new, trunc_dim, error_new
        X_update = X_new
        error = error_new
    end

    return X_update, trunc_dim, error
end

function svd_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, Y::TensorMap, trunc_dim::Int; maximum_steps=30, rtol=1e-12, verbosity=3, k=0.0)
    X_update = copy(X0)
    cost_const = tr(PhidY * Y')
    error = Inf

    U, S, V = tsvd(X_update; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim) & truncbelow(1e-14))
    for step in 1:maximum_steps
        SV = pseudopow(S, k) * V
        US_new, info = linsolve(x -> (PhidPhi(x * SV) * SV'), PhidY * SV', U * pseudopow(S, 1 - k); krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)

        U2, S2, V2 = tsvd(US_new; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim) & truncbelow(1e-14))
        U3, S3, V3 = tsvd(S2 * V2 * SV; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim) & truncbelow(1e-14))
        US2 = U2 * U3 * pseudopow(S3, k)
        SV_new, info = linsolve(x -> (US2' * PhidPhi(US2 * x)), US2' * PhidY, pseudopow(S3, 1 - k) * V3; krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)

        X_update = US2 * SV_new

        error_this = (tr(X_update' * PhidPhi(X_update)) - 2 * real(tr(PhidY * X_update')) + cost_const) / cost_const

        if verbosity > 1
            @infov 3 "Step $step: k = $k, error = $error_this"
        end

        (error_this < rtol) && return X_update, error_this
        (abs(error - error_this) / error < 1e-3) && return X_update, error_this
        error = error_this

        U4, S4, V4 = tsvd(SV_new; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim) & truncbelow(1e-14))
        U, S, V5 = tsvd(US2 * U4 * S4; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim) & truncbelow(1e-14))
        V = V5 * V4
    end

    return X_update, error
end

function svt(T::TensorMap, tau::Float64)
    U, S, V = tsvd(T; alg=TensorKit.SVD())

    thresholded_S = map(s -> max(s - tau, 0), S.data)
    rank_reduced = count(s -> s > 0, thresholded_S)
    new_S = DiagonalTensorMap(thresholded_S, domain(S, 1))
    return U, new_S, V, rank_reduced, tau
end

function svt(T::TensorMap, rank_reduced::Int; soft=true, extra_dim=1)
    if !soft
        U, S, V = tsvd(T; alg=TensorKit.SVD(), trunc=truncdim(rank_reduced) & truncbelow(1e-14))
        return U, S, V, rank_reduced, Inf
    end

    U, S, V = tsvd(T; alg=TensorKit.SVD(), trunc=truncdim(rank_reduced + extra_dim) & truncbelow(1e-14))

    tau = S.data[rank_reduced+1]
    thresholded_S = map(s -> max(s - tau, 0), S.data)
    rank_reduced = count(s -> s > 0, thresholded_S)
    S = DiagonalTensorMap(thresholded_S, domain(S, 1))

    return U, S, V, rank_reduced, tau
end