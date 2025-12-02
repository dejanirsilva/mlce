# Define the dual number type
struct D <: Number
    v::Real # primal part (value of the function)
    d::Real # dual part (derivative of the function)
end