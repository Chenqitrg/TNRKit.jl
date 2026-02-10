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
    ξ_min = 1e-7, ξ_init = 1e-4, ρ = 0.85
) where {T<:AbstractTensorMap{E,S,1,3}} where {E,S}
    psiA_approx = copy(psiA)
    NA = length(psiA) # Number of tensors in the MPS Ψ_A
    psiA_approx_psiA_approx = ΨAΨA(psiA_approx)
    psiA_approx_psiA = ΨA_approx_ΨA(psiA_approx, psiA)
    psiApsiA = ΨAΨA(psiA)
    YdY = to_number(psiApsiA) # Since C is not changed during the optimization, we can compute it once and use it in the cost function.
    cost = Float64[Inf]
    sweep = 0
    crit = true
    ξ = ξ_init

    while crit
        right_cache_A_approx_A_approx = right_cache(psiA_approx_psiA_approx)
        right_cache_A_approx_A = right_cache(psiA_approx_psiA)
        left_A_approx_A = id(E, codomain(psiApsiA[1])) # Initialize the left transfer matrix for ΨBΨB
        left_A_approx_A_approx = id(E, codomain(psiApsiA[1]))
        t_start = time()

        for pos_psiA in 1:NA
            TA = transpose(psiA[pos_psiA], ((2, 1), (3, 4)))
            TA_approx = transpose(psiA_approx[pos_psiA], ((2, 1), (3, 4)))
            PhidY = PhidPhi(TA, right_cache_A_approx_A[pos_psiA] * left_A_approx_A)
            right_left = right_cache_A_approx_A_approx[pos_psiA] * left_A_approx_A_approx
            new_psiA_approx_tr, cost_this = tr_low_rank_factor(x -> PhidPhi(x, right_left), PhidY, TA_approx, YdY, trunc.dim; verbosity = verbosity, ξ = ξ, ρ = ρ)

            new_psiA_approx = transpose(new_psiA_approx_tr, ((2,), (1, 3, 4)))
            psiA_approx[pos_psiA] = new_psiA_approx

            @plansor A_approx_A_approx_temp[-1 -2; -3 -4] := new_psiA_approx[-2; 1 2 -4] * conj(new_psiA_approx[-1; 1 2 -3])
            @plansor A_approx_A_temp[-1 -2; -3 -4] := psiA[pos_psiA][-2; 1 2 -4] * conj(new_psiA_approx[-1; 1 2 -3])

            psiA_approx_psiA_approx[pos_psiA] = A_approx_A_approx_temp
            psiA_approx_psiA[pos_psiA] = A_approx_A_temp

            left_A_approx_A_approx = left_A_approx_A_approx * A_approx_A_approx_temp
            left_A_approx_A = left_A_approx_A * A_approx_A_temp

            push!(cost, cost_this)
        end
        ξ = max(ρ * ξ, ξ_min)
        sweep += 1
        
        crit = loop_criterion(sweep, cost)
        if verbosity > 1
            @infov 3 "Sweep: $sweep, Cost: $(cost[end]), Time: $(time() - t_start)s" # Included the time taken for the sweep
        end
    end

    ΨB = [
        collect(SVD12(psiA_approx[1], trunc; reversed = true));
        collect(SVD12(psiA_approx[2], trunc; reversed = true));
        collect(SVD12(psiA_approx[3], trunc));
        collect(SVD12(psiA_approx[4], trunc));
    ]
    return ΨB
end

