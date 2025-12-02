f(x₁, x₂) = [(x₁ + x₂) * x₁^2, x₁ * x₂]
y, back = Zygote.pullback(f, 1.0, 2.0)