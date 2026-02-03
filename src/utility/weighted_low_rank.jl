function weighted_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc::TNRKit.TruncationScheme; method="svt")
    if method == "svt"
        return svt_low_rank(PhidPhi, PhidY, X0, trunc)
    elseif method == "fact"
        return factorized_low_rank(PhidPhi, PhidY, X0, trunc)
    end
end

function svt_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc::TNRKit.TruncationScheme; tau0=100, maximum_steps=maxiter(100))
    spec, vector = eigsolve(
                    PhidPhi, X0, 1, :LM; krylovdim = 5, maxiter = 100,
                    tol = 1.0e-12,
                    verbosity = 0
                )
    Lipschitz_const = real(spec[1])
    eta = 1 / Lipschitz_const
    tau = tau0 * eta

    for step in 1:maximum_steps
        gradient_step = X0 - eta * (PhidPhi(X0) - PhidY)
        X_new, rank_this = svt(gradient_step, tau)
        norm(X_new - X0) / norm(X0) < 1e-6 && return X_new
        rank_this > trunc.trunc && return X_0

        X0 = X_new
        tau *= 0.9
    end
    # monitor rank and cost function return X_fix
end

function factorized_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc::TNRKit.TruncationScheme; maximum_steps=maxiter(100))
    # monitor cost function return X_fix 
end

_threshold(S::Vector, tau::Float64) = map(s -> max(s - tau, 0), S)

function svt(T::TensorMap, tau::Float64)
    U, S, V = tsvd(T; alg=TensorKit.SVD())

    rank_reduced = 0
    for t in fusiontrees(S)
        truncated = _threshold(diag(S[t...]), tau)
        rank_sector = count(s -> s > 0, truncated)
        rank_reduced += rank_sector
        S[t...] .= diagm(truncated)
    end
    return U * S * V, rank_reduced
end