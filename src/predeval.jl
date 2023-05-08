export pred, predeval, evalmodels, ttindices

using DataFrames
using CSV

import Random
import MLJ
import CategoricalArrays
import CategoricalArrays.CategoricalValue
import StatsBase

mutable struct ModelBenchmark
    modelname::Symbol
    result::Float64
    time::Float64
    memory::Float64
end

mutable struct ModelsBenchmark
    metric::Symbol
    data::Vector{ModelBenchmark}
end

ModelsBenchmark(metric) = ModelsBenchmark(metric, [])

function add(benchmarks::ModelsBenchmark, benchmark::ModelBenchmark)
    push!(benchmarks.data, benchmark)
end

function dataframe(benchmarks::ModelsBenchmark; sort=true)::DataFrame
    df = DataFrame(
        :model => collect(map(x -> x.modelname, benchmarks.data)), 
        benchmarks.metric => collect(map(x -> x.result, benchmarks.data)), 
        :time => collect(map(x -> x.time, benchmarks.data)),
        :memory => collect(map(x -> x.memory, benchmarks.data))
    )
    sort && sort!(df, benchmarks.metric, rev=true)
    df
end

benchmarkslock = ReentrantLock()

function asyncadd(benchmarks::ModelsBenchmark, benchmark::ModelBenchmark)
    lock(benchmarkslock)
    try
        add(benchmarks, benchmark)
    finally
        unlock(benchmarkslock)
    end
end

"""
    example: 
        modelfactory = @load RandomForestClassifier pkg = ScikitLearn
        predictmlj(modelfactory, X, y)
"""
function pred(modelfactory, X, y, measure; ttratio=0.7, seed=58)
    outs = measure in [:accuracy] ? length(unique(y)) : 1
    model = modelfactory(ncol(X), outs)
    mach = MLJ.machine(model, X, y)

    train, test = ttindices(y, ttratio; seed=seed)
    MLJ.fit!(mach; rows=train, verbosity=0)
    y[test], MLJ.predict_mode(mach, X[test, :])
end

function ttindices(y, ttratio=0.7; seed=58)
    train, test = MLJ.partition(
        eachindex(y), ttratio;
        shuffle=true, rng=Random.MersenneTwister(seed)
    )
    train, test
end

function predeval(modelfactory, X, y, measure::Symbol; ttratio=0.7, seed=58)
    ytest, ŷtest = pred(modelfactory, X, y, measure; ttratio=ttratio, seed=seed)
    result = if measure == :nrmse
        nrmse(ŷtest, ytest)
    else
        getproperty(MLJ, measure)(ŷtest, ytest)
    end
    @info string(measure, ": ", result)
    result
end

nrmse(ŷtest, ytest) = MLJ.rmse(ŷtest, ytest) / StatsBase.iqr(ytest)

function evalmodels(
    data::DataFrame, target::Symbol, models::Dict, metric::Symbol;
    ttratio=0.7, seed=58, standarize=true, onehot=true
)::DataFrame
    MLJ.default_resource(CPUProcesses())

    y, X = MLJ.unpack(data, ==(target), colname -> true)

    if metric == :accuracy
        y = CategoricalArrays.categorical(y)
    end

    if standarize
        standarizer = MLJ.Standardizer(count=false)
        X = MLJ.transform(MLJ.fit!(MLJ.machine(standarizer, X)), X)
    end

    if onehot
        X = MLJ.coerce(X, Count => Multiclass)
        # X = MLJ.coerce(X, Count => OrderedFactor)
        mach = MLJ.fit!(MLJ.machine(MLJ.OneHotEncoder(ordered_factor=false, drop_last=false), X))
        X = MLJ.transform(mach, X)
    end

    benchmarks = ModelsBenchmark(metric)

    Threads.@threads for (name, model) in collect(models)
        if name == :MAGDS
            try
                Logging.disable_logging(Logging.Debug)
                magds_grid(
                    data, target, metric, benchmarks, ttratio, seed;
                    weightingstrategy=["ConstantOneWeight", "OneOverOutsUpperHalf"],
                    fuzzy=[true, false],
                    weighted=[true, false],
                    ieth=[0.00001],
                    iee=[0, 1],
                    winnerslimit=[500],
                    weightratio=[1.1, 1.5],
                    alpha=[0.1, 0.01, 0.001],
                    epoch=[0, 1, 3],
                    include_input_sensor_priority=[true, false],
                    signal_similarity_threshold=[0.0, 0.97]
                )
                Logging.disable_logging(Logging.Warn)
            catch e
                @error "error predicting $name, skipping"
            end  
        elseif name == :MAGDS_one
            Logging.disable_logging(Logging.Debug)
            magds_grid(
                data, target, metric, benchmarks, ttratio, seed;
                weightingstrategy=["OneOverOutsUpperHalf"],
                fuzzy=[true],
                weighted=[true],
                ieth=[0.00001],
                iee=[0],
                winnerslimit=[500],
                weightratio=[1.1],
                alpha=[0.01],
                epoch=[3],
                include_input_sensor_priority=[true],
                signal_similarity_threshold=[0.97]
            )
            Logging.disable_logging(Logging.Warn)
        elseif name == :MAGDS_one_regression
            Logging.disable_logging(Logging.Debug)
            magds_grid(
                data, target, metric, benchmarks, ttratio, seed;
                weightingstrategy=["OneOverOutsUpperQuarter"],
                fuzzy=[true],
                weighted=[true],
                ieth=[0.00001],
                iee=[1],
                winnerslimit=[500],
                weightratio=[1.5],
                alpha=[1.0e-5],
                epoch=[8],
                include_input_sensor_priority=[false],
                signal_similarity_threshold=[0.0]
            )
            Logging.disable_logging(Logging.Warn)
        elseif name == :MAGDS_gridsearch
            Logging.disable_logging(Logging.Debug)
            magds_grid(data, target, metric, benchmarks, ttratio, seed)
            Logging.disable_logging(Logging.Warn)
        else
            try
                time = @elapsed begin
                    mem = @allocated result = predeval(
                        model, X, y, metric; ttratio=ttratio, seed=seed
                    )
                end
                asyncadd(benchmarks, ModelBenchmark(name, result, time, mem))
            catch e
                @error "error predicting $name, skipping"
            end
        end
    end

    dataframe(benchmarks)
end

function magds_grid(
    data::DataFrame, 
    target::Symbol,
    metric::Symbol,
    benchmarks::ModelsBenchmark,
    ttratio=0.7, 
    seed=58;
    weightingstrategy=["ConstantOneWeight", "OneOverOuts", "OneOverOutsUpperHalf", "OneOverOutsUpperQuarter"],
    fuzzy=[true, false],
    weighted=[true, false],
    ieth=[0.00001, 0.99],
    iee=[1, 3, 5],
    winnerslimit=[100, 500],
    weightratio=[1.0, 1.1, 1.25, 1.5, 2.5],
    alpha=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
    epoch=[0, 3, 8],
    include_input_sensor_priority=[true, false],
    signal_similarity_threshold=[0.0, 0.97]
)
    datav2 = mapcols(col -> eltype(col) <: CategoricalValue ? string.(col) : col , data)
    if metric == :accuracy
        datav2[!, :target] = string.(datav2.target) .* "_class"
    end

    tmpdir = "evalmodels_temp"
    mkpath(tmpdir)

    train, test = ttindices(data[!, target], ttratio; seed=seed)

    trainpath = joinpath(tmpdir, "train.csv")
    testpath = joinpath(tmpdir, "test.csv")

    CSV.write(trainpath, datav2[train, :])
    CSV.write(testpath, datav2[test, :])

    # magds_predict = getproperty(@__MODULE__, Symbol("magds_" * string(metric)))
    magds_predict = getproperty(@__MODULE__, Symbol("asyncmagds_" * string(metric)))

    counterlock = ReentrantLock()
    i = 1
    Threads.@threads for weightingstrategy in weightingstrategy
        Threads.@threads for fuzzy in fuzzy
            Threads.@threads for weighted in weighted
                Threads.@threads for ieth in ieth
                    Threads.@threads for iee in iee
                        Threads.@threads for winnerslimit in winnerslimit
                            Threads.@threads for weightratio in weightratio
                                Threads.@threads for alpha in alpha
                                    Threads.@threads for epoch in epoch
                                        Threads.@threads for iisp in include_input_sensor_priority
                                            Threads.@threads for ssth in signal_similarity_threshold
                                                modelname = Symbol(string(
                                                    "MAGDS_",
                                                    "weighting[", weightingstrategy, "]_",
                                                    "fuzzy[", fuzzy, "]_",
                                                    "weighted[", weighted, "]_",
                                                    "ieth[", ieth, "]_",
                                                    "iee[", iee, "]_",
                                                    "winnerslimit[", winnerslimit, "]_",
                                                    "weightratio[", weightratio, "]_",
                                                    "alpha[", alpha, "]_",
                                                    "epoch[", epoch, "]_",
                                                    "iisp[", iisp, "]_",
                                                    "ssth[", ssth, "]"
                                                ))
                                                time = @elapsed begin
                                                    mem = @allocated result = magds_predict(
                                                        trainpath, 
                                                        testpath, 
                                                        string(target), 
                                                        weightingstrategy,
                                                        fuzzy,
                                                        weighted,
                                                        Float32(ieth), 
                                                        Int32(iee),
                                                        UInt(winnerslimit),
                                                        Float32(weightratio),
                                                        Float32(alpha),
                                                        UInt(epoch),
                                                        Bool(iisp),
                                                        Float32(ssth)   
                                                    )
                                                end
                                                if result >= 0
                                                    asyncadd(benchmarks, ModelBenchmark(modelname, result, time, mem))
                                                else
                                                    @error modelname, "result ==", result
                                                end
                                                lock(counterlock)
                                                try
                                                    if i == 1 || i % 100 == 0
                                                        @info i
                                                    end
                                                    i += 1
                                                finally
                                                    unlock(counterlock)
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    rm(tmpdir; recursive=true)
end