export PMLB
module PMLB
using PyCall
using Conda
import GZip

const CLASSIFICATION_DIR = "data/PMLB/classification"
const REGRESSION_DIR = "data/PMLB/regression"

function __init__()
    Conda.pip_interop(true)
    Conda.pip("install", "pmlb")
end

function fetchdata()
    mkpath(CLASSIFICATION_DIR)
    mkpath(REGRESSION_DIR)

    run(`python src/PMLB/fetch_data.py $CLASSIFICATION_DIR $REGRESSION_DIR`)
end

function preparedata()
    for basedir in [CLASSIFICATION_DIR, REGRESSION_DIR]
        @info("preparing $basedir:")
        for (i, datadir) in enumerate(readdir(basedir))
            datadir = joinpath(basedir, datadir)
            if isdir(datadir)
                for file in readdir(datadir)
                    filepath = joinpath(datadir, file)
                    if isfile(filepath)
                        GZip.open(filepath) do zipfile
                            targetfile = chop(file, tail=3)
                            open(joinpath(basedir, targetfile), "w") do tsvfile
                                for line in readlines(zipfile)
                                    println(tsvfile, line)
                                end
                            end
                        end
                        @info("  $(i): $(chop(file, tail=3))")
                    else
                        @warn("$file in $datadir is not a file, skipping")
                    end
                end
                rm(datadir, force=true, recursive=true)
            else
                @warn("$datadir is not a directory, skipping")
            end
        end
    end
end

function updatedata()
    @info "updating PMLB data"
    rm(CLASSIFICATION_DIR, force=true, recursive=true)
    rm(REGRESSION_DIR, force=true, recursive=true)
    fetchdata()
    preparedata()
end

end