export PMLB
module PMLB
using PyCall
using Conda
using DataFrames
using CSV
import GZip

const DATA_DIR = "data/PMLB"
const CLASSIFICATION_DIR = joinpath(DATA_DIR, "classification")
const REGRESSION_DIR = joinpath(DATA_DIR, "regression")

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
                        @info("  $(i): $(tsv2name(file))")
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

function statistics()
    cdfs = map(readdir(CLASSIFICATION_DIR)) do filename
        df = CSV.File(joinpath(CLASSIFICATION_DIR, filename)) |> DataFrame
        if !(eltype(df.target) <: Integer)
            df[!, :target] = parse.(Int64, df.target)
        end
        Symbol(tsv2name(filename)) => df
    end |> Dict{Symbol, DataFrame}
    rdfs = map(readdir(REGRESSION_DIR)) do filename
        df = CSV.File(joinpath(REGRESSION_DIR, filename)) |> DataFrame
        if !(eltype(df.target) <: AbstractFloat)
            if eltype(df.target) <: Integer
                df[!, :target] = Float64.(df.target)
            else
                df[!, :target] = parse.(Float64, df.target)
            end
        end
        Symbol(tsv2name(filename)) => df
    end |> Dict{Symbol, DataFrame}

    @info("calculating statistics for classification datasets")
    cstatsdf = statsdf(keys(cdfs), values(cdfs))
    @info("calculating statistics for regression datasets")
    rstatsdf = statsdf(keys(rdfs), values(rdfs))
    allstatsdf = vcat(cstatsdf, rstatsdf)

    CSV.write(joinpath(DATA_DIR, "classification_statistics.csv"), cstatsdf)
    CSV.write(joinpath(DATA_DIR, "regression_statistics.csv"), rstatsdf)
    CSV.write(joinpath(DATA_DIR, "all_statistics.csv"), allstatsdf)

    allstatsdf, cstatsdf, rstatsdf
end

function statsdf(keys, values)
	DataFrame(
		"name" => Symbol.(keys), 
		"features" => map(ncol, values), 
		"records" => map(nrow, values),
		"unique_target_features" => map(x -> length(unique(x.target)), values),
		"binary_features" => map(values) do df
			sum(
				map(eachcol(df[!, Not(:target)])) do col
					eltype(col) <: Integer && length(unique(col)) == 2
				end
			)
		end,
		"discrete_nonbinary_features" => map(values) do df
			sum(
				map(eachcol(df[!, Not(:target)])) do col
					eltype(col) <: Integer && length(unique(col)) != 2
				end
			)
		end,
		"continuous_features" => map(values) do df
			sum(map(x -> eltype(x) <: AbstractFloat, eachcol(df[!, Not(:target)])))
		end,
		"class_imbalance" => map(values) do df
			x = df.target
			if eltype(x) <: Real # Integer
				N = length(x)
				classes = unique(x)
				K = length(classes)
				imbalance = (K / (K - 1)) * sum(
					map(classes) do class
						n = sum(x .== class)
						(n / N - 1 / K)^2
					end
				)
			else
				missing
			end
		end
	)
end

function updatedata()
    @info "updating PMLB data"
    rm(CLASSIFICATION_DIR, force=true, recursive=true)
    rm(REGRESSION_DIR, force=true, recursive=true)
    fetchdata()
    preparedata()
    statistics()
end

function loaddf(name::String)::Union{DataFrame, Nothing}
    for dir in [CLASSIFICATION_DIR, REGRESSION_DIR]
        for file in readdir(dir)
            if tsv2name(file) == name
                return CSV.File(joinpath(dir, file)) |> DataFrame
            end
        end
    end
    nothing
end

"""
Load Penn Machine Learning Benchmark datasets.
# Arguments:
- `task::Symbol`: one of `[:all, :classification, :regression]`.
- `cluster::Symbol`: one of `[:all]`.
- `limit::Union{Int, Nothing}=nothing`: pick datasets with number of records 
    less than or equal to `limit`, no limit by default.
"""
function loaddata(;
    task::Symbol=:all,
    _cluster::Symbol=:all,
    limit::Union{Int, Nothing}=nothing
)::Dict{Symbol, DataFrame}
    dirs = if task == :classification
        [CLASSIFICATION_DIR]
    elseif task == :regression
        [REGRESSION_DIR]
    else
        [CLASSIFICATION_DIR, REGRESSION_DIR]
    end

    stats = loadstats(task)
    toload = if isnothing(limit)
        stats.name
    else
        stats[stats.records .<= limit, :name]
    end

    result = Dict()
    for dir in dirs
        for file in readdir(dir)
            setname = tsv2name(file)
            if setname in toload
                result[Symbol(setname)] = CSV.File(joinpath(dir, file)) |> DataFrame
            end
        end
    end
    result
end

function loadstats(task=:all)::DataFrame
    if task == :classification
        CSV.File(joinpath(DATA_DIR, "classification_statistics.csv")) |> DataFrame
    elseif task == :regression
        CSV.File(joinpath(DATA_DIR, "regression_statistics.csv")) |> DataFrame
    else
        CSV.File(joinpath(DATA_DIR, "all_statistics.csv")) |> DataFrame
    end
end

tsv2name(file::String)::String = chop(file, tail=4)

end