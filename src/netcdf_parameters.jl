# NetCDF parameters for common variables
#
# This file defines standard NetCDF attribute dictionaries for radar moment variables
# used when writing CfRadial-format NetCDF files. Each variable (e.g., DBZ, VEL, ZDR)
# has an associated OrderedDict containing its standard_name, long_name, and units.
# A common attribute dictionary provides shared metadata such as valid range, fill values,
# coordinate references, and grid mapping. A master lookup dictionary
# (`variable_attrib_dict`) maps variable name strings to their corresponding attribute
# dictionaries for convenient access during NetCDF file construction.

common_attrib = OrderedDict(
        "valid_min"                 => Float32(-3.4028e38),
        "valid_max"                 => Float32(3.4028e38),
        "_FillValue"                => Float32(-32768.0),
        "missing_value"             => Float32(-32768.0),
        "coordinates"               => "longitude latitude",
        "grid_mapping"              => "grid_mapping",
    )

dbz_attrib = OrderedDict(
        "standard_name"             => "DBZ",
        "long_name"                 => "Radar Reflectivity",
        "units"                     => "dBZ",
    )

zdr_attrib = OrderedDict(
        "standard_name"             => "ZDR",
        "long_name"                 => "Differential Reflectivity",
        "units"                     => "dB",
    )

vel_attrib =  OrderedDict(
        "standard_name"             => "VEL",
        "long_name"                 => "Doppler Velocity",
        "units"                     => "m/s",
    )

width_attrib = OrderedDict(
        "standard_name"             => "WIDTH",
        "long_name"                 => "Spectrum width",
        "units"                     => "m/s",
    )

kdp_attrib = OrderedDict(
        "standard_name"             => "KDP",
        "long_name"                 => "Specific Differential Phase",
        "units"                     => "deg/km",
    )

rhohv_attrib = OrderedDict(
        "standard_name"             => "RHOHV",
        "long_name"                 => "Cross-correlation coefficient",
        "units"                     => "unitless",
    )

phidp_attrib = OrderedDict(
        "standard_name"             => "PHIDP",
        "long_name"                 => "Differential Phase",
        "units"                     => "deg",
    )

sqi_attrib = OrderedDict(
        "standard_name"             => "SQI",
        "long_name"                 => "Signal Quality Index",
        "units"                     => "unitless",
    )

snr_attrib = OrderedDict(
        "standard_name"             => "SNR",
        "long_name"                 => "Signal-to-noise ratio",
        "units"                     => "dB",
    )

rainrate_attrib = OrderedDict(
    "standard_name"             => "RAINRATE",
    "long_name"                 => "CSU Blended hybrid rainrate",
    "units"                     => "mm/hr",
)

pid_attrib = OrderedDict(
    "standard_name"             => "PID",
    "long_name"                 => "particle_id",
    "units"                     => "none",
)

pid_qc_attrib = OrderedDict(
    "standard_name"             => "PID_FOR_QC",
    "long_name"                 => "particle_id",
    "units"                     => "none",
)

hid_attrib = OrderedDict(
    "standard_name"             => "HID_CSU",
    "long_name"                 => "particle_id",
    "units"                     => "none",
)

# Individual polarimetric rain-rate components (CSU_RadarTools port). Each is
# computed directly from the gridded variables and can be written alongside (or
# instead of) the blended product.
rate_z_attrib = OrderedDict(
    "standard_name"             => "RATE_Z",
    "long_name"                 => "Rain rate from R(Z)",
    "units"                     => "mm/hr",
)
rate_z_conv_attrib = OrderedDict(
    "standard_name"             => "RATE_Z_CONV",
    "long_name"                 => "Rain rate from convective R(Z)",
    "units"                     => "mm/hr",
)
rate_z_strat_attrib = OrderedDict(
    "standard_name"             => "RATE_Z_STRAT",
    "long_name"                 => "Rain rate from stratiform R(Z)",
    "units"                     => "mm/hr",
)
rate_kdp_attrib = OrderedDict(
    "standard_name"             => "RATE_KDP",
    "long_name"                 => "Rain rate from R(Kdp)",
    "units"                     => "mm/hr",
)
rate_z_zdr_attrib = OrderedDict(
    "standard_name"             => "RATE_Z_ZDR",
    "long_name"                 => "Rain rate from R(Z, Zdr)",
    "units"                     => "mm/hr",
)
rate_kdp_zdr_attrib = OrderedDict(
    "standard_name"             => "RATE_KDP_ZDR",
    "long_name"                 => "Rain rate from R(Kdp, Zdr)",
    "units"                     => "mm/hr",
)
rain_method_attrib = OrderedDict(
    "standard_name"             => "RATE_CSU_BLENDED_METHOD",
    "long_name"                 => "CSU blended rain method code",
    "units"                     => "none",
)

# Level 1 variables
dbz_l1_attrib = OrderedDict(
        "standard_name"             => "DBZ_L1",
        "long_name"                 => "Level 1 Radar Reflectivity",
        "units"                     => "dBZ",
    )

zdr_l1_attrib = OrderedDict(
        "standard_name"             => "ZDR_L1",
        "long_name"                 => "Level 1 Differential Reflectivity",
        "units"                     => "dB",
    )

vel_l1_attrib =  OrderedDict(
        "standard_name"             => "VEL_L1",
        "long_name"                 => "Level 1 Doppler Velocity",
        "units"                     => "m/s",
    )

width_l1_attrib = OrderedDict(
        "standard_name"             => "WIDTH_L1",
        "long_name"                 => "Level 1 Spectrum width",
        "units"                     => "m/s",
    )

kdp_l1_attrib = OrderedDict(
        "standard_name"             => "KDP_L1",
        "long_name"                 => "Level 1 Specific Differential Phase",
        "units"                     => "deg/km",
    )

rhohv_l1_attrib = OrderedDict(
        "standard_name"             => "RHOHV_L1",
        "long_name"                 => "Level 1 Cross-correlation coefficient",
        "units"                     => "unitless",
    )

phidp_l1_attrib = OrderedDict(
        "standard_name"             => "PHIDP_L1",
        "long_name"                 => "Level 1 Differential Phase",
        "units"                     => "deg",
    )

unknown_attrib = OrderedDict(
    "standard_name"             => "UNKNOWN",
    "long_name"                 => "unknown",
    "units"                     => "unknown",
)

variable_attrib_dict = Dict()
variable_attrib_dict["DBZ"] = dbz_attrib
variable_attrib_dict["ZDR"] = zdr_attrib
variable_attrib_dict["RHOHV"] = rhohv_attrib
variable_attrib_dict["KDP"] = kdp_attrib
variable_attrib_dict["WIDTH"] = width_attrib
variable_attrib_dict["VEL"] = vel_attrib
variable_attrib_dict["PHIDP"] = phidp_attrib
variable_attrib_dict["SNR"] = snr_attrib
variable_attrib_dict["SQI"] = sqi_attrib
variable_attrib_dict["DBZ_L1"] = dbz_l1_attrib
variable_attrib_dict["ZDR_L1"] = zdr_l1_attrib
variable_attrib_dict["RHOHV_L1"] = rhohv_l1_attrib
variable_attrib_dict["KDP_L1"] = kdp_l1_attrib
variable_attrib_dict["WIDTH_L1"] = width_l1_attrib
variable_attrib_dict["VEL_L1"] = vel_l1_attrib
variable_attrib_dict["RATE_CSU_BLENDED"] = rainrate_attrib
variable_attrib_dict["RATE_CSU_BLENDED_METHOD"] = rain_method_attrib
variable_attrib_dict["RATE_Z"] = rate_z_attrib
variable_attrib_dict["RATE_Z_CONV"] = rate_z_conv_attrib
variable_attrib_dict["RATE_Z_STRAT"] = rate_z_strat_attrib
variable_attrib_dict["RATE_KDP"] = rate_kdp_attrib
variable_attrib_dict["RATE_Z_ZDR"] = rate_z_zdr_attrib
variable_attrib_dict["RATE_KDP_ZDR"] = rate_kdp_zdr_attrib
variable_attrib_dict["PID"] = pid_attrib
variable_attrib_dict["PID_FOR_QC"] = pid_qc_attrib
variable_attrib_dict["HID_CSU"] = hid_attrib
variable_attrib_dict["UNKNOWN"] = unknown_attrib

