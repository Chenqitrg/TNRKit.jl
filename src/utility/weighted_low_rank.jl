function gradient_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, Y::TensorMap, trunc_dim::Int; μ=1e-8, maximum_steps=20000, verbosity=3)
    spec, _ = eigsolve(
        PhidPhi, ones(space(X0)), 1, :LM; krylovdim=5, maxiter=100,
        tol=1.0e-12,
        verbosity=0
    )
    Lipschitz_const = real(spec[1])
    eta = 1 / Lipschitz_const * 2

    X_update = copy(X0)
    cost_const = tr(PhidY * Y')
    error = (tr(X0' * PhidPhi(X0)) - 2 * real(tr(PhidY * X0')) + cost_const) / cost_const

    α_up = 1e-3
    α_down = 1e-2
    ε = 1e-4

    for i in 1:maximum_steps
        X_gradient_step = X_update - eta * (PhidPhi(X_update) - PhidY)
        X_new, rank, _, _ = svt(X_gradient_step, eta * μ)
        error_new = (tr(X_new' * PhidPhi(X_new)) - 2 * real(tr(PhidY * X_new')) + cost_const) / cost_const

        if verbosity > 1
            @infov 3 "Iteration $i: rank = $rank, error = $(round(error_new, digits=10)), ξ = $(round(sqrt(μ), digits=9))"
        end

        if rank > trunc_dim
            μ *= exp(α_up * (rank - trunc_dim))
        elseif rank < trunc_dim
            μ *= exp(α_down * (rank - trunc_dim))
        else
            μ *= exp(+ε)
        end

        X_update = X_new
        error = error_new
    end

    return X_update, error
end

function one_loop_reduction(PhidPhi::Function, X0::TensorMap, trunc::TruncationScheme)
    U, V = SVD12(X0, truncbelow(1e-14))
    UVPhiPhi = U' * PhidPhi(X0) * V'
    U1, S1, V1 = tsvd(UVPhiPhi)
    U2, S2, V2 = tsvd(sqrt(S1) * V1 * U1 * sqrt(S1); trunc=trunc & truncbelow(1e-14))
    P = V1' * pseudopow(S1, -1 / 2) * U2 * S2 * V2 * pseudopow(S1, -1 / 2) * U1'
    X_filtered = U * P' * V
    U, S, V = tsvd(X_filtered; alg=TensorKit.SVD(), trunc=trunc & truncbelow(1e-14))
    return U, S, V
end

function svd_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, YdY::Float64, trunc_dim::Int; maximum_steps=30, rtol=1e-12, verbosity=3, k=0.0, one_loop=true)
    X_update = copy(X0)
    error = Inf

    if one_loop
        U, S, V = one_loop_reduction(PhidPhi, X0, truncdim(trunc_dim))
    else
        U, S, V = tsvd(X_update; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim) & truncbelow(1e-14))
    end

    X_init = U * S * V
    if verbosity > 1
        @infov 3 "Initial truncation: kept rank = $(length(S.data)), error = $((tr(X_init' * PhidPhi(X_init)) - 2 * real(tr(PhidY * X_init')) + YdY) / YdY)"
    end
    for step in 1:maximum_steps
        SV = pseudopow(S, k) * V
        US_new, info = linsolve(x -> (PhidPhi(x * SV) * SV'), PhidY * SV', U * pseudopow(S, 1 - k); krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)

        U2, S2, V2 = tsvd(US_new; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim) & truncbelow(1e-14))
        U3, S3, V3 = tsvd(S2 * V2 * SV; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim) & truncbelow(1e-14))
        US2 = U2 * U3 * pseudopow(S3, k)
        SV_new, info = linsolve(x -> (US2' * PhidPhi(US2 * x)), US2' * PhidY, pseudopow(S3, 1 - k) * V3; krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)

        X_update = US2 * SV_new
        S4 = tsvd(SV_new; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim) & truncbelow(1e-14))[2]

        error_this = (tr(X_update' * PhidPhi(X_update)) - 2 * real(tr(PhidY * X_update')) + YdY) / YdY
        nuclear_norm = tr(S4)

        if verbosity > 1
            @infov 3 "Step $step: k = $k, error = $error_this, nuclear norm = $nuclear_norm"
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

function admm_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, YdY::Float64, trunc_dim::Int; ξ=1e-4, maximum_steps=10000, verbosity=3)
    X_update, _, _ = svt(X0, trunc_dim)
    Λ = zeros(eltype(X0), space(X0))
    M = copy(X_update)

    error = Inf

    α_up = 1e-3
    α_down = 1e-2
    ε = 1e-4

    for i in 1:maximum_steps
        X_new, info = linsolve(x -> (PhidPhi(x) + ξ * x), PhidY + ξ * M + Λ, X_update; krylovdim=5, maxiter=20, tol=1.0e-12, verbosity=0)
        M_new, rank, _, nuclear_norm = svt(X_new + (-Λ / ξ), ξ)
        Λ += ξ * (M_new - X_new)

        error_X = (tr(X_new' * PhidPhi(X_new)) - 2 * real(tr(PhidY * X_new')) + YdY) / YdY
        error_M = (tr(M_new' * PhidPhi(M_new)) - 2 * real(tr(PhidY * M_new')) + YdY) / YdY
        objective_function = error_X / 2 + ξ^2 * nuclear_norm
        residue = norm(X_new - M_new, Inf)
        relative_change = norm(X_update - X_new) / norm(X_update)

        if verbosity > 1
            @infov 3 "Iteration $i: rank = $rank, error_X = $(round(error_X, digits=10)), error_M = $(round(error_M, digits=10)), objective function = $(round(objective_function, digits=10)), residue = $(round(residue, digits=5)), ξ = $(round(ξ, digits=7)), relative_change = $(round(relative_change, digits=5))"
        end

        if rank > trunc_dim
            ξ *= exp(α_up * (rank - trunc_dim))
        elseif rank < trunc_dim
            ξ *= exp(α_down * (rank - trunc_dim))
        else
            ξ *= exp(-ε)
        end

        error = error_X
        X_update = X_new
        M = M_new
    end

    return X_update, error
end

function tr_low_rank_factor(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, YdY::Float64, trunc_dim::Int; ξ=1e-4, ρ=0.85, tol = 1e-12, ξ_min=1e-7, maximum_steps=40, verbosity=3, one_loop=true)
    if one_loop
        U, S, V = one_loop_reduction(PhidPhi, X0, truncdim(trunc_dim))
    else
        U, S, V = tsvd(X_update; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim) & truncbelow(1e-14))
    end

    US = U * sqrt(S)
    M_U = copy(US)
    Λ_U = zeros(eltype(US), space(US))

    SV = sqrt(S) * V
    M_V = copy(SV)
    Λ_V = zeros(eltype(SV), space(SV))

    error = Inf

    X = copy(X0)

    for i in 1:maximum_steps
        US_new, info = linsolve(x -> (PhidPhi(x * SV) * SV' + ξ * x), PhidY * SV' + ξ * M_U + Λ_U, US; krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)
        M_U_new, rank, _, nuclear_norm = svt(US_new + (-Λ_U / ξ), ξ)
        Λ_U += ξ * (M_U_new - US_new)

        SV_new, info = linsolve(x -> (US_new' * PhidPhi(US_new * x) + ξ * x), US_new' * PhidY + ξ * M_V + Λ_V, SV; krylovdim=20, maxiter=100, tol=1.0e-12, verbosity=0)
        M_V_new, rank, _, nuclear_norm = svt(SV_new + (-Λ_V / ξ), ξ)
        Λ_V += ξ * (M_V_new - SV_new)

        X_new = US_new * SV_new

        error_X = (tr(X_new' * PhidPhi(X_new)) - 2 * real(tr(PhidY * X_new')) + YdY) / YdY
        objective_function = error_X / 2 + ξ^2 * nuclear_norm
        relative_change = norm(X - X_new) / norm(X)

        if verbosity > 1
            @infov 3 "Iteration $i: rank = $rank, error_X = $(round(error_X, digits=10)), objective function = $(round(objective_function, digits=10)), ξ = $(round(ξ, digits=7)), relative_change = $(round(relative_change, digits = 3))"
        end

        last_error = error
        error = error_X
        US = US_new
        M_U = M_U_new
        SV = SV_new
        M_V = M_V_new

        if error < tol || abs(last_error - error) / error < 1e-2
            break
        end

        ξ = max(ρ * ξ, ξ_min)
    end

    return US * SV, error
end

function svt(T::TensorMap, tau::Float64)
    U, S, V = tsvd(T; alg=TensorKit.SVD())

    thresholded_S = map(s -> max(s - tau, 0), S.data)
    rank_reduced = count(s -> s > 0, thresholded_S)
    nuclear_norm = sum(thresholded_S)
    new_S = DiagonalTensorMap(thresholded_S, domain(S, 1))
    return U * new_S * V, rank_reduced, tau, nuclear_norm
end

function svt(T::TensorMap, trunc_dim::Int)
    U, S, V = tsvd(T; alg=TensorKit.SVD(), trunc=truncdim(trunc_dim + 1))
    tau = S.data[trunc_dim+1]
    thresholded_S = map(s -> max(s - tau, 0), S.data)
    nuclear_norm = sum(thresholded_S)
    rank_reduced = count(s -> s > 0, thresholded_S)
    new_S = DiagonalTensorMap(thresholded_S, domain(S, 1))
    return U * new_S * V, rank_reduced, tau, nuclear_norm
end