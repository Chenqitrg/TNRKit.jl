"""
$(TYPEDEF)

Tensor Renormalization Group

### Constructors
    $(FUNCTIONNAME)(T)

### Running the algorithm
    run!(::TRG, trunc::TruncationStrategy, stop::Stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of √2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step
    
### Fields

$(TYPEDFIELDS)

### References
* [Levin & Nave Phys. Rev. Letters 99(12) (2007)](@cite levin2007)
"""
mutable struct TRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "central tensor"
    T::TT

    function TRG(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(T)
    end
end

function step(VA::AbstractTensorMap{E, S, 2, 1}, UA::AbstractTensorMap{E, S, 1, 2}, VB::AbstractTensorMap{E, S, 2, 1}, UB::AbstractTensorMap{E, S, 1, 2}) where {E, S}
    @plansor opt = true T[-1 -2; -3 -4] := UA[-1; 1 2] * UB[-2; 2 3] * VA[4 3; -4] * VB[1 4; -3]
    return T
end

function step!(scheme::TRG, trunc::TruncationStrategy)
    A, B = SVD12(scheme.T, trunc)
    Tp = transpose(scheme.T, ((2, 4), (1, 3)))
    C, D = SVD12(Tp, trunc)
    scheme.T = step(A, B, C, D)
    return scheme
end

function Base.show(io::IO, scheme::TRG)
    println(io, "TRG - Tensor Renormalization Group")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
