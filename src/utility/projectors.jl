"""
Perform SVD of `t` in the "reversed" direction such that
`t = u * s * vh` and the arrow direction is `u → s → vh`.
"""
function svd_reversed(t::AbstractTensorMap; kwargs...)
    vh, s, u, ϵ = svd_trunc(transpose(t); kwargs...)
    u, s, vh = transpose(u), transpose(s), transpose(vh)
    @assert isdual(space(s, 1))
    return u, DiagonalTensorMap(s), vh, ϵ
end

#Utility functions for QR decomp
function QR_L(
        L::TT, T::AbstractTensorMap{E, S, M, N},
        in_ind::Int, out_ind::Int
    )::TT where {E, S, TT <: AbstractTensorMap{E, S, 1, 1}, M, N}
    permT = (
        (in_ind,),
        (
            reverse(collect(1:(in_ind - 1)))..., collect((M + 1):(M + N))...,
            reverse(collect((in_ind + 1):M))...,
        ),
    )
    permLT = (
        (
            reverse(collect(2:(in_ind + out_ind - 1)))..., 1,
            reverse(collect((in_ind + out_ind + 1):(M + N)))...,
        ), (in_ind + out_ind,),
    )
    LT = transpose(L * transpose(T, permT), permLT)
    _, Rt = left_orth(LT)
    return normalize!(Rt, Inf)
end

function QR_R(
        R::TT, T::AbstractTensorMap{E, S, M, N},
        in_ind::Int, out_ind::Int
    )::TT where {E, S, TT <: AbstractTensorMap{E, S, 1, 1}, M, N}
    permT = (
        (
            reverse(collect((M + 1):(M + in_ind - 1)))..., collect(1:M)...,
            reverse(collect((M + in_ind + 1):(M + N)))...,
        ), (M + in_ind,),
    )
    permTR = (
        (in_ind + out_ind - 1,),
        (
            reverse(collect(1:(in_ind + out_ind - 2)))..., M + N,
            reverse(collect((in_ind + out_ind):(M + N - 1)))...,
        ),
    )
    TR = transpose(transpose(T, permT) * R, permTR)
    Lt, _ = right_orth(TR)
    return normalize!(Lt, Inf)
end

# Functions to find the left and right projectors

# Function to find the list of left projectors L_list
function find_L(
        psi::Vector{<:AbstractTensorMap{E, S}}, site::Int, in_inds::Vector{Int}, out_inds::Vector{Int},
        entanglement_criterion::stopcrit
    ) where {E, S}
    L = id(E, codomain(psi[site])[in_inds[site]])
    error = [Inf]
    crit = true
    steps = 1
    while crit
        L_last_time = L
        for j in 0:(length(psi) - 1)
            running_pos = mod(site + j - 1, length(psi)) + 1
            L = QR_L(
                L, psi[running_pos], in_inds[running_pos],
                out_inds[running_pos]
            )
        end
        if space(L) == space(L_last_time)
            push!(error, abs(norm(L - L_last_time)))
        end
        crit = entanglement_criterion(steps, error)
        steps += 1
    end
    return L
end

# Function to find the list of left projectors L_list
function find_R(
        psi::Vector{<:AbstractTensorMap{E, S}}, site::Int, in_inds::Vector{Int}, out_inds::Vector{Int},
        entanglement_criterion::stopcrit
    ) where {E, S}
    R = id(E, domain(psi[site])[in_inds[site]])
    error = Float64[Inf]
    crit = true
    steps = 1
    while crit
        R_last_time = R
        for j in 0:(length(psi) - 1)
            running_pos = mod(site - j - 1, length(psi)) + 1
            R = QR_R(R, psi[running_pos], in_inds[running_pos], out_inds[running_pos])
        end
        if space(R) == space(R_last_time)
            push!(error, abs(norm(R - R_last_time)))
        end
        crit = entanglement_criterion(steps, error)
        steps += 1
    end
    return R
end

# Function to find the projector P_L and P_R
function P_decomp(
        R::TensorMap{E, S, 1, 1}, L::TensorMap{E, S, 1, 1},
        trunc::MatrixAlgebraKit.TruncationStrategy; reversed::Bool = false
    ) where {E, S}
    U, s, V, _ = if reversed
        svd_reversed(L * R; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
    else
        svd_trunc(L * R; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
    end
    re_sq = pseudopow(s, -0.5)
    PR = R * V' * re_sq
    PL = re_sq * U' * L
    return PR, PL
end

# Function to find the list of projectors
function find_projector(
        psi::Vector{T}, link::Tuple{Int, Int}, in_inds::Vector{Int}, out_inds::Vector{Int},
        entanglement_criterion::stopcrit, trunc::TruncationStrategy
    ) where {T <: AbstractTensorMap}
    left_site, right_site = link

    N = length(psi)
    @assert right_site == mod(left_site, N) + 1
    @assert length(in_inds) == N == length(out_inds)

    L = find_L(psi, right_site, in_inds, out_inds, entanglement_criterion)
    R = find_R(psi, left_site, out_inds, in_inds, entanglement_criterion)

    reversed = isdual(space(psi[right_site], in_inds[right_site]))

    PR_this, PL_next = P_decomp(R, L, trunc; reversed)
    return PR_this, PL_next
end

function MPO_disentangled!(
        psi::Vector{T}, in_inds::Vector{Int}, out_inds::Vector{Int}, entanglement_criterion::stopcrit, trunc::TruncationStrategy
    ) where {
        T <: AbstractTensorMap
    }
    n = length(psi)
    for i in 1:n
        link = (i, mod(i, n) + 1)
        PR_i, PL_ip1 = find_projector(psi, link, in_inds, out_inds, entanglement_criterion, trunc) 

        M_ip1 = length(codomain(psi[i+1]))
        N_ip1 = length(domain(psi[i+1]))

        M_i = length(codomain(psi[i]))
        N_i = length(domain(psi[i]))

        in_ind = in_inds[i+1]
        out_ind = out_inds[i]
        perm_T_ip1 = (
            (in_ind,),
            (
                reverse(collect(1 : (in_ind - 1)))..., collect((M + 1) : (M_ip1 + N_ip1))...,
                reverse(collect((in_ind + 1) : M_ip1))...,
            ),
        )
        perm_LT_ip1 = (
            (
                reverse(collect(2 : in_ind))..., 1,
                reverse(collect(in_ind + N_ip1 + 1 : (M_ip1 + N_ip1)))...,
            ), collect(in_ind + 1 : in_ind + N_ip1),
        )
        perm_T_i = (
            (
                reverse(collect(M_i + 1 : M_i + out_ind - 1))..., collect(1 : M_i)...,
                reverse(collect(M_i + out_ind + 1 : M_i + N_i)...)
            ), (M_i + out_ind,)
        )
        perm_TR_i = (
            collect(out_ind : out_ind + M_i - 1),
            (
                reverse(collect(1 : out_ind - 1))..., M_i + N_i,
                reverse(collect(out_ind + M_i : M_i + N_i - 1))...
            )
        )
        TR = transpose(transpose(psi[i], perm_T_i) * PR_i, perm_TR_i)
        LT = transpose(PL_ip1 * transpose(psi[mod(i, n) + 1], perm_T_ip1), perm_LT_ip1)
        @assert [isdual(space(psi[i], ax)) for ax in 1:numind(psi[i])] ==
            [isdual(space(LTR, ax)) for ax in 1:numind(LTR)]
        psi[i] = TR
        psi[mod(i, n) + 1] = LT
    end
    return
end

function SVD12(
        T::AbstractTensorMap{E, S, 1, 3}, trunc::MatrixAlgebraKit.TruncationStrategy;
        reversed::Bool = false
    ) where {E, S}
    T_trans = transpose(T, ((2, 1), (3, 4)); copy = true)
    U, s, V, e = if reversed
        svd_reversed(T_trans; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
    else
        svd_trunc(T_trans; trunc = trunc, alg = MatrixAlgebraKit.LAPACK_QRIteration())
    end
    @plansor S1[-1; -2 -3] := U[-2 -1; 1] * sqrt(s)[1; -3]
    @plansor S2[-1; -2 -3] := sqrt(s)[-1; 1] * V[1; -2 -3]
    return S1, S2
end

function SVD12(
        T::AbstractTensorMap{E, S, 2, 2}, trunc::MatrixAlgebraKit.TruncationStrategy;
        reversed::Bool = false
    ) where {E, S}
    U, s, V, e = reversed ? svd_reversed(T; trunc = trunc) : svd_trunc(T; trunc = trunc)
    return U * sqrt(s), sqrt(s) * V
end
