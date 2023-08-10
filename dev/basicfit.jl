using GaussMLE

out, θ_true, = GaussMLE.GaussSim.genstack(7,3,:xynb; T= Float64)
display(out[:,:,1])

θ = GaussMLE.GaussModel.θ_xynb()
Σ = GaussMLE.GaussModel.Σ_xynb()
args = GaussMLE.GaussModel.Args_xynb(1.3)

display(θ_true[1])
GaussMLE.GaussFit.fitbox!(θ, Σ, out[:,:,1], args)

