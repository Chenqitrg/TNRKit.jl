using KrylovKit

V = Z2Space(0 => 8, 1 => 8)

T = rand(V ⊗ V ← V ⊗ V)
T /= norm(T)

T1 = rand(V ← V ⊗ V)
T2 = rand(V ← V ⊗ V)
T3 = rand(V ← V ⊗ V)
T4 = rand(V ← V ⊗ V)
T1 /= norm(T1)
T2 /= norm(T2)
T3 /= norm(T3)
T4 /= norm(T4)

TT1 = T1' * T1
TT2 = T2' * T2
TT3 = T3' * T3
TT4 = T4' * T4

@tensor Φ_left[-1 -2; -3 -4] := TT1[6 -2; 5 -4] * TT2[5 -3; 6 -1]
@tensor Φ_right[-1 -2; -3 -4] := TT3[-4 5; -2 6] * TT4[-1 6; -3 5]

PhidPhi(X::TensorMap) = transpose(Φ_left * transpose(X, ((3, 1), (4, 2))) * Φ_right, ((2, 4), (1, 3)))
PhidY = PhidPhi(T)

# X_approx1, error1 = TR_low_rank(PhidPhi, PhidY, T, T, 16; maximum_steps = 10000)
# X_approx2, error2 = gradient_low_rank(PhidPhi, PhidY, T, T, 16; maximum_steps = 10000)

# println("error1 = $error1, error2 = $error2")
X_approx0, error0 = svd_low_rank(PhidPhi, PhidY, T, T, 16; maximum_steps = 20, k = 0.0)
X_tr, error_tr = tr_low_rank_factor(PhidPhi, PhidY, T, T, 16)
