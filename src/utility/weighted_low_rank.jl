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

function smart_tunning_function_increasing()
    
end

function TR_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, Y::TensorMap, trunc_dim::Int; μ=10, τ=100.0, ρ=1.01, μ_max=1e10, τ_max=1e10, maximum_steps=100, verbosity=3, initial_dim=1)
    U, S, V = tsvd(X0; alg=TensorKit.SVD(), trunc=truncdim(initial_dim) & truncbelow(1e-14))
    X_update = U * S * V
    Λ = zeros(eltype(X0), space(X0))
    M = X_update
    dim_this = initial_dim

    cost_const = tr(PhidY * Y')
    error = Inf
    for i in 1:maximum_steps
        # μ_last = μ
        X_new, info = linsolve(x -> (τ * PhidPhi(x) + μ * x), τ * PhidY + μ * M + Λ, X_update; krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)
        M_new, rank, _ = svt(X_new + (-Λ / μ), 1 / μ)
        Λ += μ * (M - X_update)
        # μ = min(μ * ρ, μ_max)
        # τ = min(μ^2, τ_max)

        error = (tr(X_new' * PhidPhi(X_new)) - 2 * real(tr(PhidY * X_new')) + cost_const) / cost_const

        residue = norm(X_new - M_new, Inf)
        relative_change = norm(X_update - X_new) / norm(X_update)
        @infov 3 "Iteration $i: rank = $rank, error = $(round(error, digits=10)), residue = $(round(residue, digits=5)), μ = $(round(μ, digits=10)), τ = $(round(τ, digits=3)), relative_change = $(round(relative_change, digits=3))"

        if relative_change < 0.01 && dim_this <= trunc_dim
            dim_this += 1
        end

        if rank < trunc_dim
            μ = min(μ + 20, μ_max)
        elseif rank == trunc_dim
            μ += 10
        elseif rank > trunc_dim
            μ -= 1
        end

        τ = min(μ^2, τ_max)


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