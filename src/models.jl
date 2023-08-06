# Define the fit models 





function genmodel(fittype::Symbol, sz::Int, args::ArgsGMLE)

    if fittype == :xynb
        fitargs = gen_xynb(sz,args)
    else
        @error "Unknown fittype"
    end

    return fitargs 
end