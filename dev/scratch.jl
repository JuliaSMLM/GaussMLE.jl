using Revise
using CUDA
using StatsFuns
using BenchmarkTools
using Images
using Zygote
using Symbolics
using SpecialFunctions
using Latexify

##
@variables x y n b  σ 

z(t) = sin(t)
g = z(x) * z(y)
Dgx = Differential(x)
expand_derivatives(Dgx(g))
##


intgauss(x) = 1/2 * (erf((x + 1/2) * sqrt(1/(2*σ^2))) - erf((x - 1/2) * sqrt(1/(2*σ^2))))




f = erf( (x+1/2)/σ )-erf( (x-1/2)./σ )

Dx = Differential(x)
expand_derivatives(Dx(f))

##
function psf(θ)
    x,y,σ = θ
    return normpdf(0,σ,x)*normpdf(0,σ,y)        
end


##



f(x) = x^2

f''(10)

psf'([1,1,1])

hessian(x -> psf(x),[1,1,1])



p(x) = normpdf(0,1,x)

p(x) = x^2
pp(x) =  gradient(x -> p(x))
function calchess(x,out)
    out[1] = pp(x)
    return nothing
end
out = CuArray(zeros(1))
@cuda calchess(1,out)
out



## block 
p(x) = x^2
pp(x) =  gradient(x -> p(x))
function calchess(x,out; iterations = 10)
    out[1] = p(x)
    return nothing
end
out = CuArray(zeros(1))
@cuda calchess(1,out)
out



