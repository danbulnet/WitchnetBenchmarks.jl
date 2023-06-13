export Penn

module Penn

using DataFrames
using CSV
using Logging

using WitchnetBenchmarks
using WitchnetBenchmarks.Utils
import WitchnetBenchmarks.PMLB

const TRAIN_TEST_DATA_DIR = "data/PMLB_train_test"

"classification task on the boston housing dataset"
function classify(;
    cluster::Symbol=:all,
    limit::Union{Int, Nothing}=1000, filter::Vector{String}=[],
    measure::Symbol=:accuracy, 
    models=fast_classification_models(),
    ttratio=0.7, seed=58, standarize=true, onehot=false, savett=false
)::Dict{Symbol, DataFrame}
    data = PMLB.loaddata(task=:classification, cluster=cluster, limit=limit)
    resultslock = ReentrantLock()
    # results = Dict{Symbol, DataFrame}()
    results = []
    # Threads.@threads for (name, df) in collect(data)
    for (name, df) in collect(data)
        if !(name in filter)
            @info "skipping", name
            continue
        end
        Logging.disable_logging(Logging.Debug)
        @info "$name classification"
        Logging.disable_logging(Logging.Warn)

        redirect_stdout(devnull) do
            if savett
                train, test = ttindices(df[:, :target], ttratio; seed=seed)

                lname = lowercase(string(name))
                ttdir = normpath(joinpath(
                    TRAIN_TEST_DATA_DIR, "classification", lname
                ))
                mkpath(ttdir)
                
                trainpath = joinpath(ttdir, "$(lname)_train.csv")
                testpath = joinpath(ttdir, "$(lname)_test.csv")

                CSV.write(trainpath, df[train, :])
                CSV.write(testpath, df[test, :])
            end
            result = evalmodels(
                df, :target, models, measure; 
                ttratio=ttratio, seed=seed, 
                standarize=standarize, onehot=onehot
            )
            Utils.writecsv(result, "penn", "classification", name)
            # lock(resultslock) do
            #     push!(results, (name => result))
            # end
            push!(results, (name => result))
            # results[name] = result
        end
    end
    # results
    Dict{Symbol, DataFrame}(results...)
end

"regression task on the boston housing dataset"
function estimate(;
    cluster::Symbol=:all,
    limit::Union{Int, Nothing}=1000, filter::Vector{String}=[],
    measure::Symbol=:nrmse,
    models=fast_regression_models(),
    ttratio=0.7, seed=58, standarize=true, onehot=false, savett=false
)::Dict{Symbol, DataFrame}
    data = PMLB.loaddata(task=:regression, cluster=cluster, limit=limit)
    results = []
    for (name, df) in data
        if !(name in filter)
            @info "skipping", name
            continue
        end
        Logging.disable_logging(Logging.Debug)
        @info "$name regression"
        Logging.disable_logging(Logging.Warn)

        redirect_stdout(devnull) do
            if savett
                train, test = ttindices(df[:, :target], ttratio; seed=seed)

                lname = lowercase(string(name))
                ttdir = normpath(joinpath(
                    TRAIN_TEST_DATA_DIR, "regression", lname
                ))
                mkpath(ttdir)
                
                trainpath = joinpath(ttdir, "$(lname)_train.csv")
                testpath = joinpath(ttdir, "$(lname)_test.csv")

                CSV.write(trainpath, df[train, :])
                CSV.write(testpath, df[test, :])
            end
            result = evalmodels(
                df, :target, models, measure; 
                ttratio=ttratio, seed=seed, 
                standarize=standarize, onehot=onehot
            )
            Utils.writecsv(result, "penn", "regression", name)
            push!(results, (name => result))
        end
    end
    Dict{Symbol, DataFrame}(results...)
end

end