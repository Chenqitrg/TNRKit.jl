mutable struct ΦTRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "central tensor"
    T::TT

    function ΦTRG(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(T)
    end
end


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
    @plansor opt = true Γ[4 7; 2 5] := left[1 2; 3 4] * T'[3 8; 1 6] * right[5 6; 7 8]
    
    F, G = SVD12(T, truncrank(trunc.howmany * 2))

    T_edge = G * Γ * F
    U, Σ, V = svd_compact(T_edge)
    T_rev = transpose(sqrt(Σ)) * transpose(U) * transpose(V) * transpose(sqrt(Σ))
    F_trunc, G_trunc = SVD12(T_rev, trunc)

    V_second_trunc = F * V' * pseudopow(Σ, -0.5) * transpose(G_trunc)
    U_second_trunc = transpose(F_trunc) * pseudopow(Σ, -0.5) * U' * G

    T_trunc = V_second_trunc * U_second_trunc
    @plansor opt = true Γ_trunc[-1 -2; -3 -4] := left[3 -3; 1 -1] * T_trunc'[1 2; 3 4] * right[-4 4; -2 2]

    AA = tr(T * Γ)
    AB = tr(T_trunc * Γ)
    BB = tr(T_trunc * Γ_trunc)
    cost = (AA + BB - 2 * real(AB)) / AA

    return V_second_trunc, U_second_trunc, T_trunc, AA, cost
end

function ΨBΨA(ΨB::Vector{<:AbstractTensorMap{E, S, 1, 3}}, ΨA::Vector{<:AbstractTensorMap{E, S, 1, 3}}) where {E, S}
    return map(zip(ΨB, ΨA)) do (B, A)
        return @plansor BA[-1 -2; -3 -4] := A[-2; 1 2 -4] * conj(B[-1; 1 2 -3])
    end
end

function ΨB_conj(ΨB::Vector{<:AbstractTensorMap{E, S, 1, 2}}) where {E, S}
    ΨB_conj_vect = [transpose(ΨB[2], ((2,), (3, 1))), transpose(ΨB[1], ((3,), (1, 2))), transpose(ΨB[4], ((2,), (3, 1))), transpose(ΨB[3], ((3,), (1, 2))), transpose(ΨB[6], ((2,), (3, 1))), transpose(ΨB[5], ((3,), (1, 2))), transpose(ΨB[8], ((2,), (3, 1))), transpose(ΨB[7], ((3,), (1, 2)))]
    return ΨB_conj_vect
end

function opt_B_down(left_B::AbstractTensorMap{E, S, 2, 2}, right_B::AbstractTensorMap{E, S, 2, 2}, left_A::AbstractTensorMap{E, S, 2, 2}, right_A::AbstractTensorMap{E, S, 2, 2}, down::AbstractTensorMap{E, S, 2, 1}, up::AbstractTensorMap{E, S, 1, 2}, TA::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    function A(x::AbstractTensorMap{E, S, 2, 1}) where {E, S}
        T_approx = x * up
        @plansor opt = true Ax[1 6; 0] := left_B[1 2; 3 4] * T_approx[2 5; 4 7] * right_B[5 6; 7 8] * up'[3 8; 0]
        return Ax
    end
    @plansor contractcheck = true b[1 6; 0] := left_A[1 2; 3 4] * TA[2 5; 4 7] * right_A[5 6; 7 8] * up'[3 8; 0]
    new_down, info = lssolve((A, A), b, 1e-5; krylovdim = 50, maxiter = 150, tol = 1.0e-12,
        verbosity = 0)
    if info.converged == 0
        @warn "Residual = $(info.normres)."
    end
    return new_down, norm(new_down - down) / norm(down), info.normres
end

function opt_B_up(left_B::AbstractTensorMap{E, S, 2, 2}, right_B::AbstractTensorMap{E, S, 2, 2}, left_A::AbstractTensorMap{E, S, 2, 2}, right_A::AbstractTensorMap{E, S, 2, 2}, down::AbstractTensorMap{E, S, 2, 1}, up::AbstractTensorMap{E, S, 1, 2}, TA::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    function A(x::AbstractTensorMap{E, S, 1, 2}) where {E, S}
        T_approx = down * x
        @plansor opt = true Ax[0; 3 8] := left_B[1 2; 3 4] * T_approx[2 5; 4 7] * right_B[5 6; 7 8] * down'[0; 1 6]
        return Ax
    end
    @plansor opt = true b[0; 3 8] := left_A[1 2; 3 4] * TA[2 5; 4 7] * right_A[5 6; 7 8] * down'[0; 1 6]
    new_up, info = lssolve((A, A), b, 1e-5; krylovdim = 50, maxiter = 150, tol = 1.0e-12,
        verbosity = 0)
    if info.converged == 0
        @warn "Residual = $(info.normres)."
    end
    return new_up, norm(new_up - up) / norm(up), info.normres
end

function to_cost(left_BB::AbstractTensorMap{E, S, 2, 2}, right_BB::AbstractTensorMap{E, S, 2, 2}, left_BA::AbstractTensorMap{E, S, 2, 2}, right_BA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}, TA::AbstractTensorMap{E, S, 2, 2}, AA::Float64) where {E, S}
    @plansor opt = true BB_const = left_BB[1 2; 3 4] * TB[2 5; 4 7] * transpose(right_BB)[5 6; 7 8] * TB'[3 8; 1 6]
    @plansor opt = true BA_const = left_BA[1 2; 3 4] * TA[2 5; 4 7] * transpose(right_BA)[5 6; 7 8] * TB'[3 8; 1 6]
    return BB_const + AA - 2 * real(BA_const)
end

function Φ_init(TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}, trunc::TruncationStrategy) where {E, S}
    Aa, aA, Bb, bB = ΦΨAΨA(TA, TB)
    BAB_env, ABA_env = env(Aa, aA, Bb, bB)
    BAB_conj, ABA_conj = env_conj(Aa, aA, Bb, bB)
    VA, UA, TA_trunc, const_AA, costA_0 = ΦΨB(trunc, TA, BAB_conj, BAB_env)
    VB, UB, TB_trunc, const_BB, costB_0 = ΦΨB(trunc, transpose(TB, ((2, 4), (1, 3))), ABA_env, ABA_conj)
    return VA, UA, VB, UB, TA_trunc, TB_trunc, const_AA, const_BB
end

function Φ_opt(TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}, trunc::TruncationStrategy) where {E, S}
    VA, UA, VB, UB, TA_trunc, TB_trunc, const_AA, const_BB = Φ_init(TA, TB, trunc)

    Ψ_A_env = Ψ_A(TA, TB)
    Ψ_B_env = [transpose(VA, ((2,), (1, 3)); copy = true), copy(UA), transpose(UB, ((2,), (3, 1)); copy = true), transpose(VB, ((3,), (2, 1)); copy = true), transpose(UA, ((2,), (3, 1)); copy = true), transpose(VA, ((3,), (2, 1)); copy = true), transpose(VB, ((2,), (1, 3)); copy = true), copy(UB)]
    Ψ_B_two_sites_env = Ψ_A(TA_trunc, transpose(TB_trunc, ((3, 1), (4, 2))))
    Ψ_B_Ψ_B_two_sites_env = ΨAΨA(Ψ_B_two_sites_env)
    Ψ_B_Ψ_A_env = ΨBΨA(Ψ_B_two_sites_env, Ψ_A_env)

    perm4 = [3, 4, 1, 2]
    Ψ_B_Ψ_B_two_sites_conj = Ψ_B_Ψ_B_two_sites_env[perm4]
    Ψ_B_Ψ_A_conj = Ψ_B_Ψ_A_env[perm4]

    sweep = 1
    crit = true

    while sweep < 2
        right_cache_BB_env = right_cache(Ψ_B_Ψ_B_two_sites_env)
        right_cache_BA_env = right_cache(Ψ_B_Ψ_A_env)

        right_cache_BB_conj = right_cache(Ψ_B_Ψ_B_two_sites_conj)
        right_cache_BA_conj = right_cache(Ψ_B_Ψ_A_conj)

        left_BB_env = id(E, codomain(Ψ_B_Ψ_B_two_sites_env[1])) # Initialize the left transfer matrix for ΨBΨB
        left_BA_env = id(E, codomain(Ψ_B_Ψ_A_env[1]))

        left_BB_conj = id(E, codomain(Ψ_B_Ψ_B_two_sites_conj[1])) # Initialize the left transfer matrix for ΨBΨB
        left_BA_conj = id(E, codomain(Ψ_B_Ψ_A_conj[1]))
        
        @infov 2 "Sweep $sweep: optimizing ΨB tensors..."
        for i in 1:4
            BB_env = right_cache_BB_env[i] * left_BB_env
            BA_env = right_cache_BA_env[i] * left_BA_env

            BB_conj = right_cache_BB_conj[i] * left_BB_conj
            BA_conj = right_cache_BA_conj[i] * left_BA_conj

            down = transpose(Ψ_B_env[2 * i - 1], ((2, 1), (3,)))
            up = Ψ_B_env[2 * i]
            TA_temp = transpose(Ψ_A_env[i], ((2, 1), (3, 4)))

            if isodd(i)
                relative_cost_before = to_cost(BB_conj, BB_env, BA_conj, BA_env, down * up, TA_temp, const_AA) / const_AA
            else
                relative_cost_before = to_cost(BB_conj, BB_env, BA_conj, BA_env, down * up, TA_temp, const_BB) / const_BB
            end

            new_down, relative_change_down, res_down = opt_B_down(BB_conj, transpose(BB_env), BA_conj, transpose(BA_env), down, up, TA_temp)

            Ψ_B_env[2 * i - 1] = transpose(new_down, ((2,), (1, 3)))
            @plansor psiBpsiB_down[-1 -2; -3 -4] := Ψ_B_env[2 * i - 1][-2; m -4] * conj(Ψ_B_env[2 * i - 1][-1; m -3])

            Ψ_B_conj_temp = transpose(new_down, ((3,), (2, 1)))
            @plansor psiBpsiB_conj_down[-1 -2; -3 -4] := Ψ_B_conj_temp[-2; m -4] * conj(Ψ_B_conj_temp[-1; m -3])

            new_up, relative_change_up, res_up = opt_B_up(BB_conj, transpose(BB_env), BA_conj, transpose(BA_env), new_down, up, TA_temp)

            @infov 4 "      res down = $res_down, res_up = $res_up"

            Ψ_B_env[2 * i] = new_up
            @plansor psiBpsiB_up[-1 -2; -3 -4] := Ψ_B_env[2 * i][-2; m -4] * conj(Ψ_B_env[2 * i][-1; m -3])
            Ψ_B_Ψ_B_two_sites_env[i] = psiBpsiB_down * psiBpsiB_up

            Ψ_B_conj_temp = transpose(new_up, ((2,), (3, 1)))
            @plansor psiBpsiB_conj_up[-1 -2; -3 -4] := Ψ_B_conj_temp[-2; m -4] * conj(Ψ_B_conj_temp[-1; m -3])
            Ψ_B_Ψ_B_two_sites_conj[i] = psiBpsiB_conj_up * psiBpsiB_conj_down
            
            left_BB_env = left_BB_env * Ψ_B_Ψ_B_two_sites_env[i]
            left_BB_conj = left_BB_conj * Ψ_B_Ψ_B_two_sites_conj[i]

            BB_two_site = new_down * new_up

            @plansor BA_temp_env[-1 -2; -3 -4] := Ψ_A_env[i][-2; 1 2 -4] * BB_two_site'[2 -3; 1 -1]
            Ψ_B_Ψ_A_env[i] = BA_temp_env # Update the transfer matrix for ΨBΨA
            left_BA_env = left_BA_env * BA_temp_env

            @plansor BA_temp_conj[-1 -2; -3 -4] := Ψ_A_env[mod(i + 1, 4) + 1][-2; 1 2 -4] * BB_two_site'[-1 1; -3 2]
            Ψ_B_Ψ_A_conj[i] = BA_temp_conj # Update the transfer matrix for ΨBΨA
            left_BA_conj = left_BA_conj * BA_temp_conj

            if isodd(i)
                relative_cost_after = to_cost(BB_conj, BB_env, BA_conj, BA_env, BB_two_site, TA_temp, const_AA) / const_AA
            else
                relative_cost_after = to_cost(BB_conj, BB_env, BA_conj, BA_env, BB_two_site, TA_temp, const_BB) / const_BB
            end

            @infov 3 "  site $i: relative cost = $relative_cost_after (before: $relative_cost_before), Δdown = $relative_change_down, Δup = $relative_change_up"
        end

        sweep += 1

        crit = sweep < 20
    end

    return Ψ_B_env
end

function step!(scheme::ΦTRG, trunc::TruncationStrategy)
    VA, UA, VB, UB, TA_trunc, TB_trunc, const_AA, const_BB = Φ_init(scheme.T, scheme.T, trunc)
    scheme.T = step(VA, UA, VB, UB)
    @show "step"
    return scheme
end
