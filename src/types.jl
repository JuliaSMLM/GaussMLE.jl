Base.@kwdef struct ZFitParams{T}
    σ_PSFx::T = 1f0
    σ_PSFy::T = 1f0
    ax::T = 1f0 
    ay::T = 1f0
    bx::T = 1f0
    by::T = 1f0
    d::T = 1f0
    γ::T = 1f0
end

Base.@kwdef mutable struct ArgsGMLE
    σ_PSF = 1           ## PSF Sigma
    fittype = :xysnb     ## Fit Type
    use_cuda = true     ## if true use cuda (if available)
    zfit::ZFitParams = ZFitParams()
end

struct ArgsFit{A,B,C}
    maxjump::A
    constraints::B
    niterations::Int
    fittype::Int
    σ_PSF::C
    zfit::ZFitParams{C}
end

# This is called when an ArgsFit type is passed to gpu kernel
function Adapt.adapt_structure(to, af::ArgsFit)
    maxjump = Adapt.adapt_structure(to, cu(af.maxjump))
    constraints = Adapt.adapt_structure(to, cu(af.constraints))
    σ_PSF = Adapt.adapt_structure(to, cu(af.σ_PSF))
    zfit = Adapt.adapt_structure(to, cu(af.zfit))
    return ArgsFit(maxjump,constraints,af.niterations,af.fittype,σ_PSF,zfit)
end
