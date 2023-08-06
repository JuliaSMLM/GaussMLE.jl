using Revise
using GaussMLE 
using BenchmarkTools
G = GaussMLE

sz = 8
roi = ones(Float32,sz,sz)
θ = ones(Float32,4)
Σ = ones(Float32,4)
logL = [1.0]

args = G.ArgsGMLE(; use_cuda = false)
fitargs = G.genmodel(:xynb, sz, args)

##

G.fitroi!(θ,Σ,logL,roi,fitargs)

display(θ)

@btime G.fitroi!(θ,Σ,logL,roi,fitargs)
@profview G.fitroi!(θ,Σ,logL,roi,fitargs)

