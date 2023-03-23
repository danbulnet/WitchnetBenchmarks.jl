export Diamond

module Diamond

using WitchnetBenchmarks
using WitchnetBenchmarks.Utils
using RDatasets
using DataFrames
using MLJ
using CSV
using Gadfly

"classification task on the diamond dataset"
function classify(;
    target::Symbol=:Clarity, 
    measure::Symbol=:accuracy, 
    models=fast_classification_models()
)::DataFrame
    data = dataset()
    result = evalmodels(data, target, models, measure)
    
    Utils.writecsv(result, "diamond", "classify", target)
    
    title = string("diamond ", lowercase(string(target)), " classification ", measure)
    plot = percent_barplot(result, :model, measure, title)
    Utils.writeimg(plot, "diamond", "classify", target)

    result
end

"regression task on the diamond dataset"
function estimate(;
    target::Symbol=:Price,
    measure::Symbol=:nrmse,
    models=fast_regression_models()
)::DataFrame
    data = dataset()
    result = evalmodels(data, target, models, measure)
    
    Utils.writecsv(result, "diamond", "estimate", target)
    
    title = string("diamond ", lowercase(string(target)), " estimation ", measure)
    plot = percent_barplot(result, :model, measure, title)
    Utils.writeimg(plot, "diamond", "estimate", target)
    
    result
end

"load ready-to-use diamond data"
dataset()::DataFrame = RDatasets.dataset("ggplot2", "diamonds")

end