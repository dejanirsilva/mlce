using Zygote
f(x₁, x₂) = (x₁ + x₂) * x₁^2
y, back = Zygote.pullback(f, 1.0, 2.0)