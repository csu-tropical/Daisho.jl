using Documenter
using Documenter.Remotes: GitHub
using Daisho

makedocs(
    sitename = "Daisho.jl",
    modules = [Daisho],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://csu-tropical.github.io/Daisho.jl",
        edit_link = "main",
    ),
    repo = GitHub("csu-tropical", "Daisho.jl"),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "guide/radar_io.md",
            "guide/quality_control.md",
            "guide/gridding.md",
            "guide/spectral_gridding.md",
            "guide/wind_synthesis.md",
            "guide/echo_products.md",
            "guide/hybrid_scan.md",
            "guide/srtm.md",
        ],
        "Theory" => [
            "theory/beam_geometry.md",
            "theory/gridding_algo.md",
            "theory/wind_synthesis.md",
        ],
        "API Reference" => [
            "api/parameters.md",
            "api/types.md",
            "api/radar_io.md",
            "api/quality_control.md",
            "api/gridding.md",
            "api/spectral_gridding.md",
            "api/wind_synthesis.md",
            "api/echo_products.md",
            "api/hybrid_scan.md",
            "api/srtm.md",
        ],
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/csu-tropical/Daisho.jl.git",
    devbranch = "main",
    push_preview = true,
)
