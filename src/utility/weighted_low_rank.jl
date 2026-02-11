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

function svd_low_rank(PhidPhi::Function, PhidY::TensorMap, YdY::Float64, X0::TensorMap, trunc_dim::Int; maximum_steps=40, rtol=1e-12, verbosity=3, k=0.0, one_loop=true)
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

function tr_low_rank_factor!(PhidPhi::Function, PhidY::TensorMap, YdY::Float64, S1::TensorMap{E,S,1,2}, S2::TensorMap{E,S,2,1}, M1::TensorMap{E,S,1,2}, M2::TensorMap{E,S,2,1}, Λ1::TensorMap{E,S,1,2}, Λ2::TensorMap{E,S,2,1}, trunc_dim::Int, ξ::Float64; tol=1e-12, verbosity=3) where {E,S}
    t1 = time()
    b1 = S2' * PhidY + ξ * M1 + Λ1
    S1, info = linsolve(x -> (S2' * PhidPhi(S2 * x) + ξ * x), b1, S1; krylovdim=20, maxiter=20, tol=1.0e-12, verbosity=0)
    t1_lin = time()
    M1, rank, _, nuclear_norm1 = svt(S1 + (-Λ1 / ξ), ξ)
    Λ1 += ξ * (M1 - S1)
    t1_later = time()

    @infov 4 "      Update S1 time = $(t1_lin - t1), later = $(t1_later - t1_lin)"


    t2 = time()
    b2 = PhidY * S1' + ξ * M2 + Λ2
    S2, info = linsolve(x -> (PhidPhi(x * S1) * S1' + ξ * x), b2, S2; krylovdim=20, maxiter=20, tol=1.0e-12, verbosity=0)
    t2_lin = time()
    M2, rank, _, nuclear_norm2 = svt(S2 + (-Λ2 / ξ), ξ)
    Λ2 += ξ * (M2 - S2)
    t2_later = time()

    @infov 4 "      Update S2 time = $(t2_lin - t2), later = $(t2_later - t2_lin)"

    X_new = S2 * S1

    cost = (tr(X_new' * PhidPhi(X_new)) - 2 * real(tr(PhidY * X_new')) + YdY) / YdY

    if verbosity > 1
        @infov 3 "cost = $(round(cost, digits=10)), nuclear_norm_S1S1 = $(round(nuclear_norm1 + nuclear_norm2, digits=10)), ξ = $(round(ξ, digits=7))"
    end

    return S1, S2, M1, M2, Λ1, Λ2, cost
end

