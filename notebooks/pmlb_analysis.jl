### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 476df2b0-c9a8-11ed-13a0-5355d5f6d33c
begin
	rootdir = normpath(joinpath(@__DIR__, ".."))
	if dirname(rootdir) != dirname(Base.current_project())
		import Pkg
		Pkg.activate(rootdir)
	end
	
	using WitchnetBenchmarks
	using WitchnetBenchmarks.PMLB

	using DataFrames
	using CSV

	using Gadfly
end

# ╔═╡ 79d64207-25ea-4266-b1d0-fde6e8b575de
html"""
<style>
	main {
	    max-width: 850px;
	}
	body {
		background: #080808;
	}
	iframe > body { background-color: orange; }
</style>
<script>
	var intervalId = window.setInterval(function(){
		var selection = document.getElementsByTagName("iframe");
		var iframes = Array.prototype.slice.call(selection);
		iframes.forEach(function(x) {
		    var y = (x.contentWindow || x.contentDocument);
		    if (y.document)y = y.document;
		    y.body.style.backgroundColor = "#080808";
		});
	}, 5000);
</script>
"""

# ╔═╡ e7f345fc-0b81-4596-ab67-79e2b9bdff2f
set_default_plot_size(21cm, 18cm)

# ╔═╡ 121b1e18-4ac7-4f76-b1ce-d2d3c00c556f
begin
	cdir = normpath(joinpath(rootdir, PMLB.CLASSIFICATION_DIR))
	rdir = normpath(joinpath(rootdir, PMLB.REGRESSION_DIR))
	cdir, rdir
end

# ╔═╡ f7f3e821-745e-4cc6-91f2-56904fe63d5d
cdfs = map(readdir(cdir)) do filename
	df = CSV.File(joinpath(cdir, filename)) |> DataFrame
	Symbol(chop(filename, tail=4)) => df
end |> Dict{Symbol, DataFrame};

# ╔═╡ ab02a3b9-f3d1-45e6-9594-9dfc1912294e
rdfs = map(readdir(rdir)) do filename
	df = CSV.File(joinpath(rdir, filename)) |> DataFrame
	Symbol(chop(filename, tail=4)) => df
end |> Dict{Symbol, DataFrame};

# ╔═╡ 0e92d095-ebda-4d31-8755-ccacc4606cff
length(cdfs), length(rdfs), length(cdfs) + length(rdfs)

# ╔═╡ bdd57492-a9fd-4f01-a307-4617b6155a3f
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
				nothing
			end
		end
	)
end

# ╔═╡ b59a6490-5265-4d00-a1dc-40f117a3d88a
cstatsdf = statsdf(keys(cdfs), values(cdfs));

# ╔═╡ 497148b8-fdc8-4bf6-b234-210b1ad29ad0
rstatsdf = statsdf(keys(rdfs), values(rdfs));

# ╔═╡ 0a3dffd4-58ab-41a8-89bf-f89150e57012
allstatsdf = vcat(cstatsdf, rstatsdf);

# ╔═╡ d6b93c92-5e01-4b0f-9ab0-9d139f8aceaf
plot(
	Scale.x_log10, Scale.y_log10,
	layer(
		cstatsdf, x=:features, y=:records,
		Geom.point, color=[colorant"orange"]
	),
	layer(
		rstatsdf, x=:features, y=:records,
		Geom.point, color=[colorant"green"]
	)
)

# ╔═╡ 5f40c5d1-2acf-4949-9f2c-bf93792499e6
allstatsdf

# ╔═╡ Cell order:
# ╠═476df2b0-c9a8-11ed-13a0-5355d5f6d33c
# ╠═79d64207-25ea-4266-b1d0-fde6e8b575de
# ╠═e7f345fc-0b81-4596-ab67-79e2b9bdff2f
# ╠═121b1e18-4ac7-4f76-b1ce-d2d3c00c556f
# ╠═f7f3e821-745e-4cc6-91f2-56904fe63d5d
# ╠═ab02a3b9-f3d1-45e6-9594-9dfc1912294e
# ╠═0e92d095-ebda-4d31-8755-ccacc4606cff
# ╠═bdd57492-a9fd-4f01-a307-4617b6155a3f
# ╠═b59a6490-5265-4d00-a1dc-40f117a3d88a
# ╠═497148b8-fdc8-4bf6-b234-210b1ad29ad0
# ╠═0a3dffd4-58ab-41a8-89bf-f89150e57012
# ╠═d6b93c92-5e01-4b0f-9ab0-9d139f8aceaf
# ╠═5f40c5d1-2acf-4949-9f2c-bf93792499e6
