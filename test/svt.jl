V = Z2Space(0=>3, 1 => 2)

T = rand(V ⊗ V ← V ⊗ V)

newT, my_rank = svt(T, 0.8)
@show my_rank