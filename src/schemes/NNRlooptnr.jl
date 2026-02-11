function ΨA_approx_ΨA(ΨA_approx::Vector{<:AbstractTensorMap{E,S,1,3}}, ΨA::Vector{<:AbstractTensorMap{E,S,1,3}}) where {E,S}
    return map(zip(ΨA_approx, ΨA)) do (A_approx, A)
        return @plansor A_approxA[-1 -2; -3 -4] := A[-2; 1 2 -4] * conj(A_approx[-1; 1 2 -3])
    end
end

function PhidPhi(T::AbstractTensorMap{E,S,2,2}, right_left::AbstractTensorMap{E,S,2,2}) where {E,S}
    @plansor T_new[-1 -2; -3 -4] := T[-1 2; -3 1] * right_left[-4 1; -2 2]
    return T_new
end


function nnr_loop_opt(
    psiA::Vector{T}, loop_criterion::stopcrit,
    trunc::TensorKit.TruncationScheme, verbosity::Int;
    ξ_min=1e-7, ξ_init=1e-4, ρ=0.85
) where {T<:AbstractTensorMap{E,S,1,3}} where {E,S}
    psiB = Ψ_B(psiA, trunc)
    M = to_M(psiB)
    Λ = map(x -> zeros(E, space(x)), M)

    NA = length(psiA) # Number of tensors in the MPS Ψ_A
    psiBpsiB = ΨBΨB(psiB)
    psiBpsiA = ΨBΨA(psiB, psiA)
    psiApsiA = ΨAΨA(psiA)
    YdY = to_number(psiApsiA) # Since C is not changed during the optimization, we can compute it once and use it in the cost function.
    cost = Float64[Inf]
    sweep = 0
    crit = true

    ξ = ξ_init
    while crit
        t0 = time()
        right_cache_BB = right_cache(psiBpsiB)
        right_cache_BA = right_cache(psiBpsiA)
        left_BB = id(E, codomain(psiBpsiB[1])) # Initialize the left transfer matrix for ΨBΨB
        left_BA = id(E, codomain(psiBpsiA[1])) # Initialize the left transfer matrix for ΨBΨA

        t1 = time()

        cost_this = Inf

        if sweep == 0
            tNt = tr(psiBpsiB[1] * right_cache_BB[1])
            tdw = tr(psiBpsiA[1] * right_cache_BA[1])
            wdt = conj(tdw)
            cost_this = real((YdY + tNt - wdt - tdw) / YdY)
            if verbosity > 1
                @infov 3 "Initial cost: $cost_this"
            end
            push!(cost, cost_this)
        end

        t2 = time()

        @infov 4 "  initialize time = $(t2 - t1)"

        for pos_psiA in 1:NA
            t3 = time()
            right_left = tN(left_BB, right_cache_BB[2*pos_psiA]) # Compute the half of the matrix N for the current position in the loop, right cache is used to minimize the number of multiplications
            PhidY = PhidPhi(transpose(psiA[pos_psiA], ((2, 1), (3, 4))), right_cache_BA[pos_psiA] * left_BA)

            S1 = psiB[2*pos_psiA]
            right_left_1 = right_left * 
            S2 = transpose(psiB[2*pos_psiA-1], ((2, 1), (3,)))

            t4 = time()

            S1, S2, M[2*pos_psiA], M[2*pos_psiA-1], Λ[2*pos_psiA], Λ[2*pos_psiA-1], cost_this = tr_low_rank_factor!(x -> PhidPhi(x, right_left), PhidY, YdY, S1, S2, M[2*pos_psiA], M[2*pos_psiA-1], Λ[2*pos_psiA], Λ[2*pos_psiA-1], trunc.dim, ξ)

            t5 = time()

            psiB[2*pos_psiA] = S1
            psiB[2*pos_psiA-1] = transpose(S2, ((2,), (1, 3)))

            @plansor BB_temp1[-1 -2; -3 -4] := psiB[2*pos_psiA][-2; 1 -4] * conj(psiB[2*pos_psiA][-1; 1 -3])
            @plansor BB_temp2[-1 -2; -3 -4] := psiB[2*pos_psiA-1][-2; 1 -4] * conj(psiB[2*pos_psiA-1][-1; 1 -3])
            psiBpsiB[2*pos_psiA] = BB_temp1 # Update the transfer matrix for ΨBΨB
            psiBpsiB[2*pos_psiA-1] = BB_temp2
            left_BB = left_BB * BB_temp2 * BB_temp1 # Update the left transfer matrix for ΨBΨB

            @plansor BA_temp[-1 -2; -3 -4] :=
                conj(psiB[2*pos_psiA-1][-1; 1 3]) *
                psiA[pos_psiA][-2; 1 2 -4] *
                conj(psiB[2*pos_psiA][3; 2 -3])
            psiBpsiA[pos_psiA] = BA_temp # Update the transfer matrix for ΨBΨA
            left_BA = left_BA * BA_temp # Update the left transfer matrix for ΨBΨA

            t6 = time()

            @infov 4 "  t_prepare = $(t4 - t3), t_optimization = $(t5 - t4), t_later = $(t6 - t5)"
        end
        sweep += 1
        ξ = max(ρ * ξ, ξ_min)

        push!(cost, cost_this)
        crit = loop_criterion(sweep, cost)
        if verbosity > 1
            @infov 3 "Sweep: $sweep, Cost: $(cost[end]), Time: $(time() - t0)s" # Included the time taken for the sweep
        end
    end
    return psiB
end

