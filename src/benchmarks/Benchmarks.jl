export warmup, predictall, classifyall, estimateall, 
    summarizeall, summarizeclassification, summarizeregression, 
    collectbenchmarks, filterbest, dropmissing, cleanbenchmark

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

function collectbenchmarks(
    benchmarkdir::String=joinpath(Utils.BENCHMARK_DIR, dataset)
)::Dict{Symbol, DataFrame}
    results = Dict{Symbol, DataFrame}()
    for file in readdir(benchmarkdir)
        df = CSV.File(joinpath(benchmarkdir, file)) |> DataFrame
        results[Symbol(chop(file, tail=4))] = df
    end
    results
end

function summarizeclassification(
    results::Dict{Symbol, DataFrame};
    save2csv::Union{String, Nothing}=nothing,
    dataset_filter::Vector{Symbol}=Symbol[]
)::DataFrame
    accuracy = Dict{Symbol, Vector{Float64}}()
    time = Dict{Symbol, Vector{Float64}}()
    memory = Dict{Symbol, Vector{Float64}}()
    for (name, benchmark) in results
        if isempty(dataset_filter) || name in dataset_filter
            for row in eachrow(benchmark)
                model = Symbol(row.model)
                accuracyvalue = if typeof(row.accuracy) <: Float64
                    row.accuracy
                else
                    parse(Float64, row.accuracy)
                end
                timevalue = if typeof(row.time) <: Float64
                    row.time
                else
                    parse(Float64, row.time)
                end
                memoryvalue = if typeof(row.memory) <: Float64
                    row.memory
                else
                    parse(Float64, row.memory)
                end
                if haskey(accuracy, model)
                    push!(accuracy[model], accuracyvalue)
                    push!(time[model], timevalue)
                    push!(memory[model], memoryvalue)
                else
                    accuracy[model] = [accuracyvalue]
                    time[model] = [timevalue]
                    memory[model] = [memoryvalue]
                end
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

function summarizeregression(
    results::Dict{Symbol, DataFrame};
    save2csv::Union{String, Nothing}=nothing,
    dataset_filter::Vector{Symbol}=Symbol[]
)::DataFrame
    nrmse = Dict{Symbol, Vector{Float64}}()
    time = Dict{Symbol, Vector{Float64}}()
    memory = Dict{Symbol, Vector{Float64}}()
    for (name, benchmark) in results
        if isempty(dataset_filter) || name in dataset_filter
            for row in eachrow(benchmark)
                model = Symbol(row.model)
                nrmsevalue = if typeof(row.nrmse) <: Float64
                    row.nrmse
                else
                    parse(Float64, row.nrmse)
                end
                timevalue = if typeof(row.time) <: Float64
                    row.time
                else
                    parse(Float64, row.time)
                end
                memoryvalue = if typeof(row.memory) <: Float64
                    row.memory
                else
                    parse(Float64, row.memory)
                end
                if haskey(nrmse, model)
                    push!(nrmse[model], nrmsevalue)
                    push!(time[model], timevalue)
                    push!(memory[model], memoryvalue)
                else
                    nrmse[model] = [nrmsevalue]
                    time[model] = [timevalue]
                    memory[model] = [memoryvalue]
                end
            end
        end
    end
    models = sort!(collect(keys(nrmse)))
    df = DataFrame(
        :model => models,
        :nrmse => map(model -> mean(nrmse[model]), models),
        :time => map(model -> mean(time[model]), models),
        :memory => map(model -> mean(memory[model]), models),
        :evaluated_datasets => map(model -> length(nrmse[model]), models),
    )
    sort!(df, [:nrmse, :time, :memory], rev=true)
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

function filterbest(
    df::DataFrame, prefix::String="MAGDS"; measure::Symbol=:accuracy, simplify=true
)::DataFrame
    df = sort(df, [measure], rev=true)
    result = df[(!).(startswith.(string.(df.model), prefix)), :]
    winner = df[startswith.(string.(df.model), prefix), :][1, :]
    if simplify
        winner.model = prefix
    end
    push!(result, winner)
    sort!(result, [measure], rev=true)
end

function dropmissing(results::Dict{Symbol, DataFrame})::Dict{Symbol, DataFrame}
    nomodels = maximum(unique(size.(values(results), 1)))
    ret = Dict{Symbol, DataFrame}()
    for (dataset, df) in results
        if size(df, 1) == nomodels
            ret[dataset] = df
        else
            println("dropping $dataset")
        end
    end
    ret
end

function cleanbenchmark(dir, results::Dict{Symbol, DataFrame})
    counter = 0
    @info keys(results)
    for file in readdir(dir)
        dataset = lowercase(split(chop(file, tail=4), "penn_regression_")[2])
        if !(dataset in lowercase.(string.(keys(results))))
            @info "removing", file, "from directory", dir
            counter += 1
            rm(joinpath(dir, file); force=true)
        end
    end
    @info "removed", counter, "files"
end