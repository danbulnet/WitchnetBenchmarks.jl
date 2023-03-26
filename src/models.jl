export 
    fast_classification_models, classification_models,
    fast_classification_models_magds, classification_models_magds,
    fast_regression_models, regression_models,
    fast_regression_models_magds, regression_models_magds,
    magds_model, one_magds_model, magds_gridsearch_models

using MLJ
using MLJFlux
using Flux

MLJ.default_resource(CPUProcesses())

###############################################################################
# classification models
###############################################################################

fast_classification_models() = Dict(
	:DecisionTreeClassifier_DecisionTree => 
		(ins, outs) -> @load(DecisionTreeClassifier, pkg=DecisionTree, verbosity=false)(),
	:RandomForestClassifier_DecisionTree => 
		(ins, outs) -> @load(RandomForestClassifier, pkg=DecisionTree, verbosity=false)(),
	# :XGBoostClassifier_XGBoost => 
	# 	(ins, outs) -> @load(XGBoostClassifier, pkg=XGBoost, verbosity=false)(),
	:EvoTreeClassifier_EvoTrees => 
        (ins, outs) -> @load(EvoTreeClassifier, pkg=EvoTrees, verbosity=false)(),
    :KNNClassifier_NearestNeighborModels =>
        (ins, outs) -> @load(KNNClassifier, pkg=NearestNeighborModels, verbosity=false)(),
    :LogisticClassifier_MLJLinearModels =>
        (ins, outs) -> @load(LogisticClassifier, pkg=MLJLinearModels, verbosity=false)(),
    :LGBMClassifier_LightGBM =>
        (ins, outs) -> @load(LGBMClassifier, pkg=LightGBM, verbosity=false)(),
    :GaussianNBClassifier_NaiveBayes =>
        (ins, outs) -> @load(GaussianNBClassifier, pkg=NaiveBayes, verbosity=false)()
)

function fast_classification_models_magds()
    models = fast_classification_models()
    models[:MAGDS] = () -> nothing
    models
end

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

function classification_models_magds()
    models = classification_models()
    models[:MAGDS] = () -> nothing
    models
end

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

###############################################################################
# regression models
###############################################################################

fast_regression_models() = Dict(
	:DecisionTreeRegressor_DecisionTree => 
		(ins, outs) -> @load(DecisionTreeRegressor, pkg=DecisionTree, verbosity=false)(),
	:RandomForestRegressor_DecisionTree => 
		(ins, outs) -> @load(RandomForestRegressor, pkg=DecisionTree, verbosity=false)(),
	# :XGBoostRegressor_XGBoost => 
	# 	(ins, outs) -> @load(XGBoostRegressor, pkg=XGBoost, verbosity=false)(),
    :EvoTreeRegressor_EvoTrees => 
        (ins, outs) -> @load(EvoTreeRegressor, pkg=EvoTrees, verbosity=false)(),
    :KNNRegressor_NearestNeighborModels =>
        (ins, outs) -> @load(KNNRegressor, pkg=NearestNeighborModels, verbosity=false)(),
    :LinearRegressor_MLJLinearModels =>
        (ins, outs) -> @load(LinearRegressor, pkg=MLJLinearModels, verbosity=false)(),
    :LGBMRegressor_LightGBM =>
        (ins, outs) -> @load(LGBMRegressor, pkg=LightGBM, verbosity=false)()
)

function fast_regression_models_magds()
    models = fast_regression_models()
    models[:MAGDS] = () -> nothing
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

function regression_models_magds()
    models = regression_models()
    models[:MAGDS] = () -> nothing
    models
end

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

###############################################################################
# MAGDS models
###############################################################################

magds_model() = Dict(
    :MAGDS => nothing
)

one_magds_model() = Dict(
    :MAGDS_one => nothing
)

magds_gridsearch_models() = Dict(
    :MAGDS_gridsearch => nothing
)
