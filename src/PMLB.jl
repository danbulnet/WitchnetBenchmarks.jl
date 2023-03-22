__precompile__()
module PMLB
using PyCall

const pmlb = PyNULL()

function __init__()
    copy!(pmlb, pyimport_conda("pmlb"))
end

end