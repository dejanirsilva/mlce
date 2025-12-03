using Plots, LaTeXStrings, ForwardDiff, Statistics, BenchmarkTools, LinearAlgebra, Zygote, Random
pgfplotsx()
default(legend_font_halign=:left)

############################################################
### Include helper functions ###
############################################################
for f in ("dual_struct.jl", "dual_implementation.jl", "dual_test_functions.jl", "dual_derivative.jl")
    include(joinpath(@__DIR__,"helper_functions/", f))
end

############################################################
### Auto-differentiation with dual numbers ###
############################################################

# Define the dual number type
struct D <: Number
    v::Real # primal part (value of the function)
    d::Real # dual part (derivative of the function)
end

# How to perform basic operations on dual numbers
import Base: +,-,*,/, convert, promote_rule, exp, sin, cos, log
+(x::D, y::D)  = D(x.v + y.v, x.d + y.d)
-(x::D, y::D)  = D(x.v - y.v, x.d - y.d)
*(x::D, y::D)  = D(x.v * y.v, x.d * y.v + x.v * y.d)
/(x::D, y::D)  = D(x.v / y.v, (x.v * y.d - y.v * x.d) / y.v^2)
exp(x::D)      = D(exp(x.v), exp(x.v) * x.d)
log(x::D)      = D(log(x.v), 1.0 / x.v * x.d)
sin(x::D)      = D(sin(x.v), cos(x.v) * x.d)
cos(x::D)      = D(cos(x.v), -sin(x.v) * x.d)
promote_rule(::Type{D}, ::Type{<:Number}) = D
Base.show(io::IO,x::D) = print(io,x.v," + ",x.d," ϵ")

# Define f and g functions and their exact derivatives
f(x) = exp(x^2)
g(x) = cos(x^3)
fprime_exact(x) = exp(x^2) * 2 * x
gprime_exact(x) = -sin(x^3) * 3 * x^2

# Define the dual number
u = D(1.0, 1.0)
v = D(2.0, 1.0)
D(1,1.0)

# Primal and exact derivative
f(1.0), fprime_exact(1.0)
f(u)

g(2.0), gprime_exact(2.0)
g(v)

# Derivative function
function derivative(f::Function, x::Real)
    u = D(x, 1.0) # construct the dual number u = x_0 + 1.0 * ϵ
    return f(u).d # extract the dual part of the result
end
derivative(f, 1.0), derivative(g, 2.0)

############################################################
### Dual numbers with multiple dimensions ###
############################################################

# Functions to test
f(x; n = 1) = n == 1 ? exp(-mean(sqrt.(x)))+1.0 : [exp(-mean(sqrt.(x)))+i for i = 1:n]

function jvp(f, x::AbstractVector, v::AbstractVector)
    xdual = ForwardDiff.Dual.(x, v)
    jvp = ForwardDiff.partials.(f(xdual))
    return collect(jvp)
end

i_range = 10:10:100
input_primal, input_jvp, input_gradient = [], [], []
output_primal, output_jvp, output_gradient = [], [], []
for i in i_range
    # f
    x = ones(i)
    v = ones(i)
    push!(input_primal, @benchmark f($x))
    push!(input_jvp, @benchmark jvp(f, $x, $v))
    push!(input_gradient, @benchmark ForwardDiff.gradient(f, $x))

    # g
    x = collect(1.0:10)
    v = x/10
    push!(output_primal, @benchmark f($x, n = $i))
    push!(output_jvp, @benchmark jvp(x->f(x, n = $i), $x, $v))
    push!(output_gradient, @benchmark ForwardDiff.jacobian(x->f(x, n = $i), $x))
end

function ei(i, n)
    e = zeros(n)
    e[i] = 1
    return e
end

function fd_jacobian(f, x0; m = 1, h = 1e-4)
    n = length(x0)
    return hcat([(f(x0+h*ei(i, n); n = m) - f(x0; n = m))/h for i in 1:n]...)
end

# ForwardDiff: Median time: input dimension
input_primal_median     = [median(benchmark).time * 1e-9 for benchmark in input_primal]
input_jvp_median        = [median(benchmark).time * 1e-9 for benchmark in input_jvp]
input_gradient_median   = [median(benchmark).time * 1e-9 for benchmark in input_gradient]

# ForwardDiff: Median time: output dimension
output_primal_median    = [median(benchmark).time * 1e-9 for benchmark in output_primal]
output_jvp_median       = [median(benchmark).time * 1e-9 for benchmark in output_jvp]
output_gradient_median  = [median(benchmark).time * 1e-9 for benchmark in output_gradient]

# Input dimension
p_input = plot(i_range, input_jvp_median./input_primal_median / (input_jvp_median[1]/input_primal_median[1]), line = 2, label="JVP", xlabel="input dimension (n)", ylabel="Median time (s)", legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(i_range, input_gradient_median./input_primal_median / (input_gradient_median[1]/input_primal_median[1]), line = 2, label="Gradient", ylims = (0,11.0))
hline!([1], label="", linestyle=:dash, color=:black)

# Output dimension
p_output = plot(i_range, output_jvp_median./output_primal_median / (output_jvp_median[1]/output_primal_median[1]), line = 2, label="JVP", xlabel="output dimension (m)", ylabel="Median time (s)", legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(i_range, output_gradient_median./output_primal_median / (output_gradient_median[1]/output_primal_median[1]), line = 2, label="Jacobian", ylims = (0,11.0))
hline!([1], label="", linestyle=:dash, color=:black)

plot(p_input, p_output, layout = (1,2), size = (1000, 500))

########################################################
### Reverse mode AD ###
########################################################

# Reverse mode AD with Zygote
f(x₁, x₂) = (x₁ + x₂) * x₁^2
y, back = Zygote.pullback(f, 1.0, 2.0)
back(1.0)

function pullback_manual(x1, x2)
    v1, v2 = x1, x2
    v3 = v1 + v2
    v4 = v1^2
    v5 = v3 * v4
    function back(v̅5)
        v̅3 = v̅5 * v4
        v̅4 = v̅5 * v3
        v̅1 = v̅3 * 1 + v̅4 * (2v1)
        v̅2 = v̅3 * 1
        return (v̅1, v̅2)
    end
    return v5, back
end
primal, back = pullback_manual(1.0, 2.0)
back(1.0)

########################################################
### Benchmarking reverse mode AD ###
########################################################

f(x; n = 1) = n == 1 ? exp(-mean(sqrt.(x)))+1.0 : [exp(-mean(sqrt.(x)))+i for i = 1:n]
# Varying output dimension (m)
function bench_output_scaling(i)
    x = collect(1.0:10)
    v̅ = ones(i)
    return @benchmark begin
        # redefine function in each run to force recompilation and new closure type
        f_local = let n = $i
            x -> [exp(-mean(sqrt.(x))) + j for j = 1:n]
        end
        y, back = Zygote.pullback(f_local, $x)
        back($v̅)
    end
end

# Varying input dimension (n)
function bench_input_scaling(i)
    x = ones(i)
    v̅ = 1.0
    return @benchmark begin
        f_local = x -> exp(-mean(sqrt.(x))) + 1.0
        y, back = Zygote.pullback(f_local, $x)
        back($v̅)
    end
end

# Full Jacobian benchmark (rebuilds pullback each time)
function bench_jacobian_scaling(i)
    x = collect(1.0:10)
    return @benchmark begin
        f_local = let n = $i
            x -> [exp(-mean(sqrt.(x))) + j for j = 1:n]
        end
        Zygote.jacobian(f_local, $x)
    end
end

# Benchmark
b_final     = bench_jacobian_scaling(1000)
b_initial   = bench_jacobian_scaling(10)
f_final     = @benchmark f(collect(1.0:10), n = 1000)
f_initial   = @benchmark f(collect(1.0:10), n = 10)
r_final     = median(b_final).time / median(f_final).time
r_initial   = median(b_initial).time / median(f_initial).time
r_final / r_initial / (1000/10)

i_range = 10:15:100
zygote_input_primal, zygote_input_vjp, zygote_input_gradient = [], [], []
zygote_output_primal, zygote_output_vjp, zygote_output_jacobian = [], [], []
for i in i_range
    # Vary input dimension, fixed output dimension
    x = ones(i)
    v = [1.0]
    push!(zygote_input_primal, @benchmark f($x))
    push!(zygote_input_vjp, bench_input_scaling(i))
    push!(zygote_input_gradient, @benchmark Zygote.gradient(f, $x))

    # Vary output dimension, fixed input dimension
    x = collect(1.0:10)
    v̅ = ones(i)
    push!(zygote_output_primal, @benchmark f($x, n = $i))
    push!(zygote_output_vjp, bench_output_scaling(i))
    push!(zygote_output_jacobian, bench_jacobian_scaling(i))
end

# Zygote: Median time: input dimension
zygote_input_primal_median      = [median(benchmark).time * 1e-9 for benchmark in zygote_input_primal]
zygote_input_vjp_median         = [median(benchmark).time * 1e-9 for benchmark in zygote_input_vjp]
zygote_input_gradient_median    = [median(benchmark).time * 1e-9 for benchmark in zygote_input_gradient]

# Zygote: Median time: output dimension
zygote_output_primal_median     = [median(benchmark).time * 1e-9 for benchmark in zygote_output_primal]
zygote_output_vjp_median        = [median(benchmark).time * 1e-9 for benchmark in zygote_output_vjp]
zygote_output_jacobian_median   = [median(benchmark).time * 1e-9 for benchmark in zygote_output_jacobian]

# Input dimension
p_input = plot(i_range, zygote_input_vjp_median./zygote_input_primal_median / (zygote_input_vjp_median[1]/zygote_input_primal_median[1]), line = 2, label="VJP", xlabel="input dimension (n)", ylabel="Median time (s)", legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(i_range, zygote_input_gradient_median./zygote_input_primal_median / (zygote_input_gradient_median[1]/zygote_input_primal_median[1]), line = 2, label="Gradient", ylims = (0,11))
hline!([1], label="", linestyle=:dash, color=:black)

# Output dimension
p_output = plot(i_range, zygote_output_vjp_median./zygote_output_primal_median / (zygote_output_vjp_median[1]/zygote_output_primal_median[1]), line = 2, label="VJP", xlabel="output dimension (m)", ylabel="Median time (s)", legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(i_range, zygote_output_jacobian_median./zygote_output_primal_median / (zygote_output_jacobian_median[1]/zygote_output_primal_median[1]), line = 2, label="Jacobian", ylims = (0,11.0))
hline!([1], label="", linestyle=:dash, color=:black)

plot(p_input, p_output, layout = (1,2), size = (1000, 500))

########################################################
# The order of operations matters
########################################################

using ForwardDiff, Zygote, BenchmarkTools, Random

# ---------------------------
# CASE A: n = 1000, m = 1
# ---------------------------
rng = Random.MersenneTwister(0)
xA = rand(rng, 1000)

# Common inner function: simple nonlinearity
h(x) = sin.(x)
g(y) = y .^ 2 .+ 2y
f(z) = sum(exp.(z))         # scalar output

# Compose them
h(xA)
g(h(xA))
f(g(h(xA)))
F(x) = f(g(h(x)))

forward_benchmark = @benchmark ForwardDiff.gradient(F, $xA)  # Forward mode
reverse_benchmark = @benchmark Zygote.gradient(F, $xA)       # Reverse mode

forward_time = median(forward_benchmark).time * 1e-9
reverse_time = median(reverse_benchmark).time * 1e-9

forward_time / reverse_time
