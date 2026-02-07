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
    X_init = U * S * V
    if verbosity > 1
        @infov 3 "Initial truncation: kept rank = $(length(S.data)), error = $((tr(X_init' * PhidPhi(X_init)) - 2 * real(tr(PhidY * X_init')) + cost_const) / cost_const)"
    end
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

function smart_tunning_increasing()

end

function TR_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, Y::TensorMap, trunc_dim::Int; ξ=1e-4, ρ=0.99, ξ_min=1e-7, maximum_steps=10000, verbosity=3)
    X_update, _, _ = svt(X0, trunc_dim)
    Λ = zeros(eltype(X0), space(X0))
    M = copy(X_update)

    cost_const = tr(PhidY * Y')
    error = Inf

    for i in 1:maximum_steps
        X_new, info = linsolve(x -> (PhidPhi(x) + ξ * x), PhidY + ξ * M + Λ, X_update; krylovdim=5, maxiter=20, tol=1.0e-12, verbosity=0)
        M_new, rank, _ = svt(X_new + (-Λ / ξ), ξ)
        Λ += ξ * (M_new - X_update)

        error_X = (tr(X_new' * PhidPhi(X_new)) - 2 * real(tr(PhidY * X_new')) + cost_const) / cost_const
        error_M = (tr(M_new' * PhidPhi(M_new)) - 2 * real(tr(PhidY * M_new')) + cost_const) / cost_const
        residue = norm(X_new - M_new, Inf)
        relative_change = norm(X_update - X_new) / norm(X_update)

        if verbosity > 1
            @infov 3 "Iteration $i: rank = $rank, error_X = $(round(error_X, digits=10)), error_M = $(round(error_M, digits=10)), residue = $(round(residue, digits=5)), ξ = $(round(ξ, digits=7)), relative_change = $(round(relative_change, digits=3))"
        end

        if rank > trunc_dim
            ξ = min(ξ / ρ, 1.0)
        else
            ξ = max(ξ * ρ, ξ_min)
        end
        
        X_update = X_new
        M = M_new
    end

    return X_update, error
end

function svt(T::TensorMap, tau::Float64)
    U, S, V = tsvd(T; alg=TensorKit.SVD())

    thresholded_S = map(s -> max(s - tau, 0), S.data)
    rank_reduced = count(s -> s > 0, thresholded_S)
    new_S = DiagonalTensorMap(thresholded_S, domain(S, 1))
    return U * new_S * V, rank_reduced, tau
end

function svt(T::TensorMap, trunc_dim::Int)
    U, S, V = tsvd(T; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim + 1))
    tau = S.data[trunc_dim+1]
    thresholded_S = map(s -> max(s - tau, 0), S.data)
    rank_reduced = count(s -> s > 0, thresholded_S)
    new_S = DiagonalTensorMap(thresholded_S, domain(S, 1))
    return U * new_S * V, rank_reduced, tau
end