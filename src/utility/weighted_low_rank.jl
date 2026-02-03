function weighted_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc::TNRKit.TruncationScheme; method="svt")
    if method == "svt"
        return svt_low_rank(PhidPhi, PhidY, X0, trunc)
    elseif method == "fact"
        return factorized_low_rank(PhidPhi, PhidY, X0, trunc)
    end
end

function svt_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc_dim::Int; tau0=100, maximum_steps=maxiter(100))
    spec, _ = eigsolve(
        PhidPhi, X0, 1, :LM; krylovdim=5, maxiter=100,
        tol=1.0e-12,
        verbosity=0
    )
    Lipschitz_const = real(spec[1])
    eta = 1 / Lipschitz_const * 0.99
    tau = tau0 * eta

    rank_this = 0
    for step in 1:maximum_steps
        gradient_step = X0 - eta * (PhidPhi(X0) - PhidY)
        X_new, rank_this = svt(gradient_step, tau)
        error = norm(X_new - X0) / norm(X0)

        rank_this > trunc_dim && return X_0, rank_this, error

        X0 = X_new
        tau *= 0.95

        @info "Step $step: rank = $rank_this, error = $error"
    end

    return X0, rank_this, error
end

function factorized_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc::TNRKit.TruncationScheme; maximum_steps=maxiter(100))
    # monitor cost function return X_fix 
end

_threshold(S::Vector, tau::Float64) = map(s -> max(s - tau, 0), S)

function svt(T::TensorMap, tau::Float64)
    U, S, V = tsvd(T; alg=TensorKit.SVD())

    thresholded_S = map(s -> max(s - tau, 0), S.data)
    rank_reduced = count(s -> s > 0, thresholded_S)
    new_S = DiagonalTensorMap(thresholded_S, domain(S, 1))
    return U * new_S * V, rank_reduced
end