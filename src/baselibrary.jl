# Base Library for Gauss mLE 


function intGauss1D(ii::Int, x, sigma::Real) 
	norm = 0.5 / sigma^2
    return 0.5 * (erf((ii - x + 0.5) * sqrt(norm)) - erf((ii - x - 0.5) * sqrt(norm)))
end

function alpha(z::Real, Ax::Real, Bx::Real, d::Real)
	return 1.0 + (z / d)^2 + Ax * (z / d)^3 + Bx * (z / d)^4
end

function dalphadz(z::Real, Ax::Real, Bx::Real, d::Real)
    return 2.0 * z / d^2 + 3.0 * Ax * z^2 / d^3 + 4.0 * Bx * z^3 / d^4
end

function d2alphadz2(z::Real, Ax::Real, Bx::Real, d::Real)
    return 2.0 / d^2 + 6.0 * Ax * z / d^3 + 12.0 * Bx * z^2 / d^4
end

function derivativeIntGauss1D(ii::Int, x, sigma::Real, N, PSFy::Real)
    a = exp(-0.5 * ((ii + 1.0 - x) / sigma)^2)
    b = exp(-0.5 * ((ii - x) / sigma)^2)
    dudt = -N / sqrt(2.0 * pi) / sigma * (a - b) * PSFy
    d2udt2 = -N / sqrt(2.0 * pi) / sigma^3 * ((ii + 1.0 - x) * a - (ii - x) * b) * PSFy
    return (dudt, d2udt2)
end

function derivativeIntGauss1DSigma(ii::Int, x::Real,  Sx::Real, N::Real, PSFy::Real)    
    ax = exp(-0.5 * ((ii + 1.0 - x) / Sx)^2)
    bx = exp(-0.5 * ((ii - x) / Sx)^2) 
    dudt = -N / sqrt(2.0 * pi) / Sx / Sx * (ax * (ii - x + 1.0) - bx * (ii - x)) * PSFy
    d2udt2 = -2.0 / Sx * dudt[0] - N / sqrt(2.0 * pi) / pow(Sx, 5) * (ax * pow((ii - x + 1.0), 3) - bx * pow((ii - x), 3)) * PSFy
    return (dudt, d2udt2)
end

function derivativeIntGauss2DSigma(ii::Int, jj::Int, x::Real, y::Real,  S::Real, N::Real, 
    PSFx::Real, PSFy::Real) 

    (dSx, ddSx) = derivativeIntGauss1DSigma(ii, x, S, N, PSFy)
    (dSy, ddSy) = derivativeIntGauss1DSigma(jj, y, S, N, PSFx)
    dudt    = dSx + dSy
    d2udt2 = ddSx + ddSy
    return (dudt, d2udt2)
end

function derivativeIntGauss2Dz(ii::Int, jj::Int, theta, PSFSigma_x::Real, PSFSigma_y::Real, 
    Ax::Real, Ay::Real, Bx::Real, By::Real, gamma::Real, d::Real, dudt, d2udt2) 
 
    z = theta[5]
    alphax  = alpha(z - gamma, Ax, Bx, d)
    alphay  = alpha(z + gamma, Ay, By, d)
 
    Sx = PSFSigma_x * sqrt(alphax)
    Sy = PSFSigma_y * sqrt(alphay)
    
    PSFx = intGauss1D(ii, theta[1], Sx)
    PSFy = intGauss1D(jj, theta[2], Sy)
    
    (dudt[1], d2udt2[1]) = derivativeIntGauss1D(ii, theta[1], Sx, theta[3], PSFy)
    (dudt[2], d2udt2[2]) = derivativeIntGauss1D(jj, theta[2], Sy, theta[3], PSFx)
    (dSx, ddSx) = derivativeIntGauss1DSigma(ii, theta[1], Sx, theta[3], PSFy)
    (dSy, ddSy) = derivativeIntGauss1DSigma(jj, theta[2], Sy, theta[3], PSFx)

    dSdalpha_x = PSFSigma_x / 2.0 / sqrt(alphax)
    dSdalpha_y = PSFSigma_y / 2.0 / sqrt(alphay)
    
    dSdzx  = dSdalpha_x * dalphadz(z - gamma, Ax, Bx, d) 
    dSdzy  = dSdalpha_y * dalphadz(z + gamma, Ay, By, d)
    dudt[5] = dSx * dSdzx + dSy * dSdzy

    d2Sdalpha2_x = -PSFSigma_x / 4.0 / alphax^1.5
    d2Sdalpha2_y = -PSFSigma_y / 4.0 / alphay^1.5
    
    ddSddzx  = d2Sdalpha2_x * dalphadz(z - gamma, Ax, Bx, d)^2 + dSdalpha_x * d2alphadz2(z - gamma, Ax, Bx, d) 
    ddSddzy  = d2Sdalpha2_y * dalphadz(z + gamma, Ay, By, d)^2 + dSdalpha_y * d2alphadz2(z + gamma, Ay, By, d) 
    
    d2udt2[5] = ddSx * (dSdzx * dSdzx) + dSx * ddSddzx +
            ddSy * (dSdzy * dSdzy) + dSy * ddSddzy

    return (PSFx, PSFy)
end

function centerofMass2D(sz::Int, data) 
	
    tmpx = 0.0
    tmpy = 0.0
    tmpsum = 0.0
    
    for ii = 0:sz - 1 
        for jj = 0:sz - 1 
            tmpx += data[sz * jj + ii + 1] * ii 
            tmpy += data[sz * jj + ii + 1] * jj
            tmpsum += data[sz * jj + ii + 1]
        end
    end

    x = tmpx / tmpsum
    y = tmpy / tmpsum
    return (x, y)
end

function gaussFMaxMin2D(sz::Int, sigma::Real, data) 

    filteredpixel = 0
    sum = 0
    maxN = 0.0
    minBG = 1e10 # big
    norm = 0.5 / sigma^2

    for kk = 0:sz - 1
        for ll = 0:sz - 1
            filteredpixel = 0.0
            sum = 0.0
            for ii = 0:sz - 1 
                for jj = 0:sz - 1
                    filteredpixel += exp(-(ii - kk)^2 * norm) * exp(-(ll - jj)^2 * norm) * data[ii * sz + jj + 1]
                    sum += exp(-(ii - kk)^2 * norm) * exp(-(ll - jj)^2 * norm)
                end
            end

            filteredpixel /= sum
            maxN = max(maxN, filteredpixel)
            minBG = min(minBG, filteredpixel)

        end
    end
    return (maxN, minBG)
end

function mymax(x, y)
    if(x >= y)
        return x
    end

    return y
end

function mymin(x, y)
    if(x >= y)
        return y
    end

    return x
end

function matInv(m::Array{<:Real}, sz::Int)

    tmp1 = 0
    mtype = typeof(m[1])
    yy = zeros(mtype, 25)
    minv = zeros(mtype, sz, sz)
    diag = zeros(mtype, sz)

    for jj = 1:sz
        for ii = 1:jj
            if(ii > 1)
                for kk = 1:ii-1
                    tmp1 += m[kk, ii] * m[jj, kk]
                end
                m[jj, ii] -= tmp1
                tmp1 = 0.0
            end
        end
        for ii = jj + 1:sz
            if(jj > 1)
                for kk = 1:jj-1
                    tmp1 += m[kk, ii] * m[jj, kk]
                end
                m[jj, ii] = (1/m[jj, jj]) * (m[jj, ii] - tmp1)
                tmp1 = 0
            else
                m[jj, ii] = (1/m[jj, jj]) * m[jj, ii]
            end
        end
    end

    tmp1 = 0
    b = 0
    
    for num = 1:sz

        if(num == 1)
            yy[1] = 1
        else
            yy[1] = 0
        end

        for ii = 2:sz
            if(ii == num)
                b = 1
            else
                b = 0
            end
            for jj = 1:ii-1
                tmp1 += m[jj, ii] * yy[jj]
            end

            yy[ii] = b - tmp1
            tmp1 = 0.0
        end

        minv[num, sz - 1] = yy[sz - 1]/ m[sz - 1, sz - 1]

        for ii = sz - 1:-1:1
            for jj = ii + 2:sz
                tmp1 += m[jj, ii] * m[num, jj]
            end
            minv[num, ii] = (1/m[ii, ii]) * (yy[ii] - tmp1)
            tmp1 = 0.0
        end

    end

    for ii = 1:sz
        diag[ii] = minv[ii, ii]
    end

    return minv, diag
end
