# CfRadial 2.1 spec validator. Returns a `ValidationReport` with errors (must
# fix) and warnings (informational). In `strict=true` mode, presence of any
# error raises ArgumentError; warnings still pass.

"""
    ValidationReport(errors::Vector{String}, warnings::Vector{String})

Outcome of `validate_spec`. `errors` lists spec violations; `warnings` lists
informational deviations.
"""
struct ValidationReport
    errors::Vector{String}
    warnings::Vector{String}
end

ValidationReport() = ValidationReport(String[], String[])

Base.isempty(r::ValidationReport) = isempty(r.errors) && isempty(r.warnings)

function Base.show(io::IO, r::ValidationReport)
    print(io, "ValidationReport(errors=$(length(r.errors)), warnings=$(length(r.warnings)))")
end

"""
    validate_spec(volume::Volume; strict::Bool=false) -> ValidationReport

Validate `volume` against the CfRadial 2.1 spec. With `strict=true`, raises
`ArgumentError` if any errors are found.
"""
function validate_spec(volume::Volume; strict::Bool=false)
    r = ValidationReport()
    _validate_globals!(r, volume)
    _validate_sweeps!(r, volume)
    if strict && !isempty(r.errors)
        throw(ArgumentError("validate_spec: " * join(r.errors, "; ")))
    end
    return r
end

function _validate_globals!(r::ValidationReport, v::Volume)
    isempty(v.title) && push!(r.warnings, "global attr `title` is empty")
    isempty(v.institution) && push!(r.warnings, "global attr `institution` is empty")
    isempty(v.source) && push!(r.warnings, "global attr `source` is empty")
    isempty(v.history) && push!(r.warnings, "global attr `history` is empty")
    isempty(v.instrument_name) && push!(r.errors, "global attr `instrument_name` is empty")
    isempty(v.sweeps) && push!(r.errors, "volume has no sweeps")
    return nothing
end

function _validate_sweeps!(r::ValidationReport, v::Volume)
    for (i, s) in enumerate(v.sweeps)
        prefix = "sweep $(i) (sweep_number=$(s.sweep_number))"
        isempty(s.time) && push!(r.errors, "$prefix: empty time axis")
        isempty(s.range) && push!(r.errors, "$prefix: empty range axis")
        n = length(s.time)
        if length(s.azimuth) != n
            push!(r.errors, "$prefix: azimuth length $(length(s.azimuth)) ≠ time length $(n)")
        end
        if length(s.elevation) != n
            push!(r.errors, "$prefix: elevation length $(length(s.elevation)) ≠ time length $(n)")
        end
        for (name, fld) in s.fields
            sz = size(fld.data)
            if sz != (n, length(s.range))
                push!(r.errors, "$prefix field `$(name)`: data shape $(sz) ≠ (time=$(n), range=$(length(s.range)))")
            end
            fld.metadata.units === nothing && push!(r.warnings,
                "$prefix field `$(name)`: missing `units` attribute")
            fld.metadata.fill_value === nothing && push!(r.warnings,
                "$prefix field `$(name)`: missing `_FillValue` attribute")
        end
        if v.platform_is_mobile && s.georeference === nothing
            push!(r.warnings, "$prefix: platform is mobile but no georeference subgroup present")
        end
    end
    return nothing
end
