export 
    fast_classification_models, classification_models,
    fast_regression_models, regression_models,
    magds_model, one_magds_model, magds_gridsearch_models

using MLJ
using MLJFlux
using Flux

MLJ.default_resource(CPUProcesses())

fast_classification_models() = Dict(
    :MAGDS => nothing,
	:DecisionTreeClassifier_DecisionTree => 
		(ins, outs) -> @load(DecisionTreeClassifier, pkg=DecisionTree, verbosity=false)(),
	:RandomForestClassifier_DecisionTree => 
		(ins, outs) -> @load(RandomForestClassifier, pkg=DecisionTree, verbosity=false)(),
	:XGBoostClassifier_XGBoost => 
		(ins, outs) -> @load(XGBoostClassifier, pkg=XGBoost, verbosity=false)(),
	:EvoTreeClassifier_EvoTrees => 
        (ins, outs) -> @load(EvoTreeClassifier, pkg=EvoTrees, verbosity=false)(),
    :KNNClassifier_NearestNeighborModels =>
        (ins, outs) -> @load(KNNClassifier, pkg=NearestNeighborModels, verbosity=false)(),
    :LogisticClassifier_MLJLinearModels =>
        (ins, outs) -> @load(LogisticClassifier, pkg=MLJLinearModels, verbosity=false)(),
    :LGBMClassifier_LightGBM =>
        (ins, outs) -> @load(LGBMClassifier, pkg=LightGBM, verbosity=false)()
)

fast_regression_models() = Dict(
    :MAGDS => nothing,
	:DecisionTreeRegressor_DecisionTree => 
		(ins, outs) -> @load(DecisionTreeRegressor, pkg=DecisionTree, verbosity=false)(),
	:RandomForestRegressor_DecisionTree => 
		(ins, outs) -> @load(RandomForestRegressor, pkg=DecisionTree, verbosity=false)(),
	:XGBoostRegressor_XGBoost => 
		(ins, outs) -> @load(XGBoostRegressor, pkg=XGBoost, verbosity=false)(),
    :EvoTreeRegressor_EvoTrees => 
        (ins, outs) -> @load(EvoTreeRegressor, pkg=EvoTrees, verbosity=false)(),
    :KNNRegressor_NearestNeighborModels =>
        (ins, outs) -> @load(KNNRegressor, pkg=NearestNeighborModels, verbosity=false)(),
    :LinearRegressor_MLJLinearModels =>
        (ins, outs) -> @load(LinearRegressor, pkg=MLJLinearModels, verbosity=false)(),
    :LGBMRegressor_LightGBM =>
        (ins, outs) -> @load(LGBMRegressor, pkg=LightGBM, verbosity=false)()
)

nothread_classification_models() = Dict(
	:RandomForestClassifier_ScikitLearn => 
		(ins, outs) -> @load(RandomForestClassifier, pkg=ScikitLearn, verbosity=false)(),
	:XGBoostClassifier_XGBoost =>
        (ins, outs) -> @load(AdaBoostClassifier, pkg=ScikitLearn, verbosity=false)(),
    :KNeighborsClassifier_ScikitLearn =>
        (ins, outs) -> @load(KNeighborsClassifier, pkg=ScikitLearn, verbosity=false)(),
    :LogisticClassifier_ScikitLearn =>
        (ins, outs) -> @load(LogisticClassifier, pkg=ScikitLearn, verbosity=false)()
)

nothread_regression_models() = Dict(
	:RandomForestRegressor_ScikitLearn => 
		(ins, outs) -> @load(RandomForestRegressor, pkg=ScikitLearn, verbosity=false)(),
	:XGBoostRegressor_XGBoost => 
        (ins, outs) -> @load(AdaBoostRegressor, pkg=ScikitLearn, verbosity=false)(),
    :KNeighborsRegressor_ScikitLearn =>
        (ins, outs) -> @load(KNeighborsRegressor, pkg=ScikitLearn, verbosity=false)(),
    :LinearRegressor_ScikitLearn =>
        (ins, outs) -> @load(LinearRegressor, pkg=ScikitLearn, verbosity=false)()
)

function classification_models()
    models = fast_classification_models()
    models[:NeuralNetworkClassifier_MLJFlux] = (ins, outs) -> begin
        builder = MLJFlux.@builder begin
            Chain(
                Dense(ins => 64, relu),
                Dense(64 => 32, relu),
                Dense(32 => outs),
                softmax
            )
        end
        @load(NeuralNetworkClassifier, pkg=MLJFlux, verbosity=true)(
            builder=builder, rng=123, epochs=50, acceleration=CUDALibs()
        )
    end
    models
end

function regression_models()
    models = fast_regression_models()
    models[:NeuralNetworkRegressor_MLJFlux] = (ins, outs) -> begin
        builder = MLJFlux.@builder begin
            Chain(
                Dense(ins, 64, relu),
                Dense(64, 32, relu),
                Dense(32, outs)
            )
        end
        @load(NeuralNetworkRegressor, pkg=MLJFlux, verbosity=false)(
            builder=builder, rng=123, epochs=50, acceleration=CUDALibs()
        )
    end
    models
end

magds_model() = Dict(
    :MAGDS => nothing
)

one_magds_model() = Dict(
    :MAGDS_one => nothing
)

magds_gridsearch_models() = Dict(
    :MAGDS_gridsearch => nothing
)
