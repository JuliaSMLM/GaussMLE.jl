using Revise
using GaussMLE
using CUDA 
import Adapt
G = GaussMLE

sz = 8
roi = cu(ones(Float32,sz,sz))
θ = cu(ones(Float32,4))
Σ = cu(ones(Float32,4))
logL = cu([1.0f0])

args = G.ArgsGMLE()
args.use_cuda = true
fitargs = G.genmodel(:xynb, sz, args)

@cuda G.fitroi!(θ,Σ,logL,roi,fitargs)


##
break

isbits(fitgpu)
isbits(fitgpu.zfit)
isbits(fitgpu.maxjump)
isbits(fitgpu.constraints)
isbits(fitgpu.fittype)
isbits(fitgpu.niterations)
isbits(fitgpu.σ_PSF)

a = cudaconvert(fitargs)

isbits(cudaconvert("asdf"))

##

@cuda G.fitroi!(θ,Σ,logL,roi,fitargs)

display(θ)

##
struct MyStruct{A,B}
    a :: A
    b :: B
  end
  
  Adapt.@adapt_structure MyStruct
  
  s = MyStruct(CUDA.rand(1,100),Ref(20))
  
  isbits(s)

  t = cudaconvert(s)
  isbits(cudaconvert(s)) # true
  
  