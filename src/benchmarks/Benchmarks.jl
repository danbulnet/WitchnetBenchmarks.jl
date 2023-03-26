export warmup, predictall, classifyall, estimateall, 
    summarizeall, collectbenchmarks, summarizeclassification

using Statistics

import Logging

include("Iris.jl")
include("Penguin.jl")
include("Star.jl")
include("WhiteWine.jl")
include("RedWine.jl")
include("BostonHousing.jl")
include("Diamond.jl")
include("Penn.jl")

ALL_DATASETS = [
    Iris,
    Penguin,
    Star,
    WhiteWine,
    RedWine,
    BostonHousing,
    Diamond
]

function warmup(dataset::Module=Iris)
    for _ in 1:5
        try
            Base.@invokelatest dataset.classify()
        catch
            continue
        end
    end
end

function predictall(
    classifymodels=fast_classification_models(),
    estimatemodels=fast_regression_models()
)::Dict{Symbol, Dict{Symbol, DataFrame}}
    Dict(
        :classification => classifyall(classifymodels),
        :regression => estimateall(estimatemodels)
    )
end

function classifyall(
    classifymodels=fast_classification_models(),
    datasets=ALL_DATASETS
)::Dict{Symbol, DataFrame}
    results = Dict{Symbol, DataFrame}()
    for dataset in datasets
        key = Symbol(lowercase(string(dataset)))
        
        Logging.disable_logging(Logging.Debug)
        @info "$key classification"
        Logging.disable_logging(Logging.Warn)

        redirect_stdout(devnull) do
            results[key] = dataset.classify(models=classifymodels)
        end
    end
    results
end

function estimateall(
    estimatemodels=fast_classification_models(),
    datasets=ALL_DATASETS
)::Dict{Symbol, DataFrame}    
    results = Dict{Symbol, DataFrame}()
    for dataset in datasets
        key = Symbol(lowercase(string(dataset)))

        Logging.disable_logging(Logging.Debug)
        @info "$key regression"
        Logging.disable_logging(Logging.Warn)

        redirect_stdout(devnull) do
            results[key] = dataset.estimate(models=estimatemodels)
        end
    end
    results
end

function collectbenchmarks(dataset::String="penn")::Dict{Symbol, DataFrame}
    results = Dict{Symbol, DataFrame}()
    benchmarkdir = joinpath(Utils.BENCHMARK_DIR, dataset)
    for file in readdir(benchmarkdir)
        df = CSV.File(joinpath(benchmarkdir, file)) |> DataFrame
        results[Symbol(chop(file, tail=4))] = df
    end
    results
end

function summarizeclassification(
    results::Dict{Symbol, DataFrame};
    save2csv::Union{String, Nothing}=nothing
)::DataFrame
    accuracy = Dict{String, Vector{Float64}}()
    time = Dict{String, Vector{Float64}}()
    memory = Dict{String, Vector{Float64}}()
    for benchmark in values(results)
        for row in eachrow(benchmark)
            if haskey(accuracy, row.model)
                push!(accuracy[row.model], row.accuracy)
                push!(time[row.model], row.time)
                push!(memory[row.model], row.memory)
            else
                accuracy[row.model] = [row.accuracy]
                time[row.model] = [row.time]
                memory[row.model] = [row.memory]
            end
        end
    end
    models = sort!(collect(keys(accuracy)))
    df = DataFrame(
        :model => models,
        :accuracy => map(model -> mean(accuracy[model]), models),
        :time => map(model -> mean(time[model]), models),
        :memory => map(model -> mean(memory[model]), models),
        :evaluated_datasets => map(model -> length(accuracy[model]), models),
    )
    sort!(df, [:accuracy, :time, :memory], rev=true)
    if !isnothing(save2csv)
        CSV.write(save2csv, df)
    end
    df
end

function summarizeall(results=predictall())::Dict{Symbol, DataFrame}
    classification = results[:classification]
    regression = results[:regression]

    classificationlen = length(values(classification))
    regressionlen = length(values(regression))

    accuracy = sum(getproperty.(values(classification), :accuracy)) / classificationlen
    time = sum(getproperty.(values(classification), :time)) / classificationlen
    memory = sum(getproperty.(values(classification), :memory)) / classificationlen
    classificationmean = DataFrame(
        :model => first(values(classification)).model,
        :accuracy => accuracy,
        :time => time,
        :memory => memory
    )

    nrmse = sum(getproperty.(values(regression), :nrmse)) / regressionlen
    time = sum(getproperty.(values(regression), :time)) / regressionlen
    memory = sum(getproperty.(values(regression), :memory)) / regressionlen
    regressionmean = DataFrame(
        :model => first(values(regression)).model,
        :nrmse => nrmse,
        :time => time,
        :memory => memory
    )

    Utils.writecsv(classificationmean, "summary", "classification", :mean)    
    title = string("classification accuracy mean of ", classificationlen, " datasets")
    plot = percent_barplot(classificationmean, :model, :accuracy, title)
    Utils.writeimg(plot, "summary", "accuracy", :mean)

    Utils.writecsv(regressionmean, "summary", "regression", :mean)
    title = string("regression nrmse mean of ", regressionlen, " datasets")
    plot = percent_barplot(regressionmean, :model, :nrmse, title)
    Utils.writeimg(plot, "summary", "nrmse", :mean)

    Dict(
        :classificationmean => classificationmean,
        :regressionmean => regressionmean
    )
end