using Documenter
using Daisho

makedocs(
    sitename = "Daisho.jl",
    modules = [Daisho],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://mmbell.github.io/Daisho.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "guide/radar_io.md",
            "guide/quality_control.md",
            "guide/gridding.md",
            "guide/spectral_gridding.md",
            "guide/wind_synthesis.md",
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
            "api/srtm.md",
        ],
    ],
    checkdocs = :none,
)

deploydocs(
    repo = "github.com/csu-tropical/Daisho.jl.git",
    devbranch = "main",
    push_preview = true,
)
