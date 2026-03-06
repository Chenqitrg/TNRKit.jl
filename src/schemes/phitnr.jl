function ΦΨAΨA(TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    TA_tr = transpose(TA, ((2, 4), (1, 3)); copy = true)
    Aa = TA_tr * TA_tr'
    aA = TA_tr' * TA_tr
    Bb = TB * TB'
    bB = TB' * TB
    return Aa, aA, Bb, bB
end

function env(Aa::AbstractTensorMap{E, S, 2, 2}, aA::AbstractTensorMap{E, S, 2, 2}, Bb::AbstractTensorMap{E, S, 2, 2}, bB::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    @plansor opt = true contractcheck = true BAB[-1 -2; -3 -4] := bB[-2 6; -1 5] * aA[6 8; 5 7] * Bb[-3 7; -4 8]
    @plansor opt = true ABA[-1 -2; -3 -4] := aA[5 -1; 6 -2] * bB[7 5; 8 6] * Aa[8 -4; 7 -3]
    return BAB, ABA
end

function env_conj(Aa::AbstractTensorMap{E, S, 2, 2}, aA::AbstractTensorMap{E, S, 2, 2}, Bb::AbstractTensorMap{E, S, 2, 2}, bB::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    @plansor opt = true BAB[-1 -2; -3 -4] := bB[5 -1; 6 -2] * Aa[6 8; 5 7] * Bb[8 -4; 7 -3]
    @plansor opt = true ABA[-1 -2; -3 -4] := aA[-2 6; -1 5] * Bb[7 5; 8 6] * Aa[-3 7; -4 8]
    return BAB, ABA
end

function ΦΨB(trunc::TruncationStrategy, T::AbstractTensorMap{E, S, 2, 2}, left::AbstractTensorMap{E, S, 2, 2}, right::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    @plansor opt = true Γ[-1 -2; -3 -4] := left[3 -3; 1 -1] * T'[1 2; 3 4] * right[-4 4; -2 2]
    F, G = SVD12(T, truncrank(trunc.howmany * 2))
    T_edge = G * Γ * F
    U, Σ, V = svd_compact(T_edge)
    T_rev = transpose(sqrt(Σ)) * transpose(U) * transpose(V) * transpose(sqrt(Σ))
    F_trunc, G_trunc = SVD12(T_rev, trunc)
    V_second_trunc = F * V' * pseudopow(Σ, -0.5) * transpose(G_trunc)
    U_second_trunc = transpose(F_trunc) * pseudopow(Σ, -0.5) * U' * G
    return V_second_trunc, U_second_trunc
end

function ΦΨB(trunc::TruncationStrategy, TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}, BAB_env::AbstractTensorMap{E, S, 2, 2}, ABA_env::AbstractTensorMap{E, S, 2, 2}, BAB_conj::AbstractTensorMap{E, S, 2, 2}, ABA_conj::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    VA, UA = ΦΨB(trunc, TA, BAB_conj, BAB_env)
    VB, UB = ΦΨB(trunc, transpose(TB, ((2, 4), (1, 3))), ABA_env, ABA_conj)
    return VA, UA, VB, UB
end

function right_cache_two_sites(transfer_mats::Vector{T}) where {T <: AbstractTensorMap{E, S, 2, 2}} where {E, S}
    n = length(transfer_mats)
    @assert iseven(n)

    m = n ÷ 2
    cache = similar(transfer_mats[1:4])
    cache[end] = id(E, domain(transfer_mats[end]))

    for i in (m - 1):-1:1
        cache[i] = transfer_mats[2 * i - 1] * transfer_mats[2 * i] * cache[i + 1]
    end

    return cache
end

function Φ_opt(TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}, trunc::TruncationStrategy) where {E, S}
    ΦΨA = Ψ_A(TA, TB)
    Aa, aA, Bb, bB = ΦΨAΨA(TA, TB)
    BAB_env, ABA_env = env(Aa, aA, Bb, bB)
    BAB_conj, ABA_conj = env_conj(Aa, aA, Bb, bB)
    VA, UA, VB, UB = ΦΨB(trunc, TA, TB, BAB_env, ABA_env, BAB_conj, ABA_conj)
    Ψ_B = [transpose(VA, ((2,), (1, 3)); copy = true), copy(UA), transpose(UB, ((2,), (3, 1)); copy = true), transpose(VB, ((3,), (2, 1)); copy = true), transpose(UA, ((2,), (3, 1)); copy = true), transpose(VA, ((3,), (2, 1)); copy = true), transpose(UB, ((2,), (1, 3)); copy = true), copy(VB)]
    Ψ_B_conj = copy(Ψ_B)

    Ψ_B_Ψ_B = ΨBΨB(Ψ_B)
    Ψ_B_Ψ_B_conj = ΨBΨB(Ψ_B_conj)

    Ψ_B_Ψ_A = ΨBΨA(Ψ_B, ΦΨA)
    Ψ_B_Ψ_A_conj = ΨBΨA(Ψ_B_conj, ΦΨA)

    while crit
        right_cache_BB = right_cache_two_sites(Ψ_B_Ψ_B)
        right_cache_BA = right_cache_two_sites(Ψ_B_Ψ_A)
        right_cache_BB_conj = right_cache_two_sites(Ψ_B_Ψ_B_conj)
        right_cache_BA_conj = right_cache_two_sites(Ψ_B_Ψ_A_conj)
        left_BB = id(E, codomain(Ψ_B_Ψ_B[1])) # Initialize the left transfer matrix for ΨBΨB
        left_BA = id(E, codomain(Ψ_B_Ψ_A[1]))
        for i in 1:4

        end
    end
end
