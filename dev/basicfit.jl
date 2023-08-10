using GaussMLE

out, θ_true, = GaussMLE.GaussSim.genstack(5,5,:xynb; T= Float64)

θ = GaussMLE.GaussModel.θ_xynb()
Σ = GaussMLE.GaussModel.Σ_xynb()
args = GaussMLE.GaussModel.Args_xynb(1.3)

display(θ_true[1])
GaussMLE.GaussFit.fitbox!(θ, Σ, out[:,:,1], args)

