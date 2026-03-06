function ΦΨ_A(TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}) where {E, S}
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

function ΦΨ_B(trunc::TruncationStrategy, TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}, BAB_env::AbstractTensorMap{E, S, 2, 2}, ABA_env::AbstractTensorMap{E, S, 2, 2}, BAB_conj::AbstractTensorMap{E, S, 2, 2}, ABA_conj::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    @plansor opt = true A_env[-1 -2; -3 -4] := BAB_conj[3 -3; 1 -1] * TA'[1 2; 3 4] * BAB_env[-4 4; -2 2] # ⟨A|env|A⟩ = Tr(A * A_env)
    @plansor opt = true B_env[-1 -2; -3 -4] := ABA_env[3 -3; 1 -1] * TB'[2 4; 1 3] * ABA_conj[-4 4; -2 2] # ⟨B|env|B⟩ = Tr(B * B_env)
    FA, GA = SVD12(TA, truncrank(trunc.howmany * 2))
    FB, GB = SVD12(transpose(TB, ((2, 4), (1, 3))), truncrank(trunc.howmany * 2))

    A_edge = GA * A_env * FA
    B_edge = GB * B_env * FB

    UA, ΣA, VA = svd_compact(A_edge)
    UB, ΣB, VB = svd_compact(B_edge)

    A_reversal = transpose(sqrt(ΣA)) * transpose(UA) * transpose(VA) * transpose(sqrt(ΣA))
    B_reversal = transpose(sqrt(ΣB)) * transpose(UB) * transpose(VB) * transpose(sqrt(ΣB))

    FA_trunc, GA_trunc = SVD12(A_reversal, trunc)
    FB_trunc, GB_trunc = SVD12(B_reversal, trunc)

    VA_second_trunc = FA * VA' * pseudopow(ΣA, -0.5) * transpose(GA_trunc)
    UA_second_trunc = transpose(FA_trunc) * pseudopow(ΣA, -0.5) * UA' * GA

    VB_second_trunc = FB * VB' * pseudopow(ΣB, -0.5) * transpose(GB_trunc)
    UB_second_trunc = transpose(FB_trunc) * pseudopow(ΣB, -0.5) * UB' * GB
    return VA_second_trunc, UA_second_trunc, VB_second_trunc, UB_second_trunc
end

function Φ_opt()
    
end
