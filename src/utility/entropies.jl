function VN_entropy(M::TensorMap)
    _, S, _ = svd_trunc(M)
    S_vec_norm = S.data / S.data[1]
    S_von = - sum(S_vec_norm .* log.(S_vec_norm))
    return S_von
end