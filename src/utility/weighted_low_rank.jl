function weighted_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc::TNRKit.TruncationScheme; method="svt")
    if method == "svt"
        return svt_low_rank(PhidPhi, PhidY, X0, trunc)
    elseif method == "fact"
        return factorized_low_rank(PhidPhi, PhidY, X0, trunc)
    end
end

function svt_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc::TNRKit.TruncationScheme; tau0=100, maximum_steps=maxiter(100))
    # monitor rank and cost function return X_fix
end

function factorized_low_rank(PhidPhi::Function, PhidY::TensorMap, X0::TensorMap, trunc::TNRKit.TruncationScheme; maximum_steps=maxiter(100))
    # monitor cost function return X_fix 
end

function svt(T::TensorMap, tau::Float64)
    return T_low
end