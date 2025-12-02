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