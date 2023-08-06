using CUDA
import Adapt

struct TestStruct{A,B}
    a::A
    b::B
    c::Float32
end

function Adapt.adapt_structure(to, ts::TestStruct)
    a = Adapt.adapt_structure(to, cu(ts.a))
    b = Adapt.adapt_structure(to, cu(ts.b))
    c = Adapt.adapt_structure(to, cu(ts.c))
    display(a)
    return TestStruct(a, b, c)
end

function testcuda!(out,ts::TestStruct)
    out[1]= ts.a[1]
    return nothing
end

ts = TestStruct(ones(4),ones(4,4),4f0);
out = ones(1)
@cuda testcuda!(out,ts);

gout = cu(out)
testcuda!(gout,ts)

