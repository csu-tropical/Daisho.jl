function initialize_working_dirs(date, working_base_dir, plot_base_dir)

    println("Initializing...")
    
    # Set up the working directories directories
    # Make sure the directory exists
    temp_dir = working_base_dir * "/proc_temp"
    mkpath(temp_dir)

    # Check to see if anything is left over from previous run
    # This directory holds raw data in SIGMET format
    sigmet_raw = temp_dir * "/sigmet_raw/" * date
    mkpath(sigmet_raw)
    sigmet_files = readdir(sigmet_raw; join=true)
    filter!(!isdir,sigmet_files)

    # This directory holds raw data converted to CfRadial format
    cfrad_raw = temp_dir * "/cfrad_raw/" * date
    mkpath(cfrad_raw)
    cfrad_files = readdir(cfrad_raw; join=true)
    filter!(!isdir,cfrad_files)

    # This directory holds raw data merged into volumes
    cfrad_merge = temp_dir * "/cfrad_merge/" * date
    mkpath(cfrad_merge)
    merge_files = readdir(cfrad_merge; join=true)
    filter!(!isdir,merge_files)

    # This directory holds QCed data merged into volumes
    cfrad_qc = temp_dir * "/cfrad_qc/" * date
    mkpath(cfrad_qc)
    qc_files = readdir(cfrad_qc; join=true)
    filter!(!isdir,qc_files)

    # This directory holds gridded data
    cart_grid = temp_dir * "/gridded_data/cartesian/" * date
    mkpath(cart_grid)
    cart_grid_files = readdir(cart_grid; join=true)
    filter!(!isdir,cart_grid_files)

    # This directory archives all the files
    archive_dir = working_base_dir * "/archive"
    mkpath(archive_dir * "/sigmet_raw/" * date)
    mkpath(archive_dir * "/cfrad_raw/" * date)
    mkpath(archive_dir * "/cfrad_merge/" * date)
    mkpath(archive_dir * "/cfrad_qc/" * date)
    mkpath(archive_dir * "/gridded_data/cartesian/" * date)

    # Move any left over files from a previous run
    archive_files(temp_dir, archive_dir, collect([sigmet_files; cfrad_files; merge_files; qc_files; cart_grid_files]))

    # Make sure the plot directory exists
    # No need to clean this one out
    plot_dir = plot_base_dir * "/" * date
    mkpath(plot_dir)
    
    return temp_dir, archive_dir, plot_dir

end

function fetch_sigmet_data(date, time, dir)

    files = date * "/SEA" * date * "_" * chop(time) * "*"
    run(`rsync -avz -e 'ssh -p 20801' --progress radarproc@192.168.111.59:"/shares/radar_data/projects/piccolo/$files" $dir`)

end

function archive_files(temp_dir, archive_dir, files)
    for file in files
        newfile = replace(file, temp_dir => archive_dir)
        println("Archiving $file -> $newfile")
        mv(file, newfile, force=true)
    end
end


function process_volumes(date, time, temp_dir, archive_dir, plot_dir, raw_moment_dict, qc_moment_dict)

    # Get the sigmet data from the SEA-POL server
    sigmet_raw = temp_dir * "/sigmet_raw/" * date
    fetch_sigmet_data(date, time, sigmet_raw)

    # Convert the Sigmet files to CfRadial
    println("Converting Sigmet data to ..")
    cfrad_raw = temp_dir * "/cfrad_raw"
    sigmet_files = readdir(sigmet_raw; join=true)
    filter!(!isdir,sigmet_files)
    run(`RadxConvert -outdir $cfrad_raw -f $(sigmet_files)`)

    # Check the CfRadial files for scan mode
    # Merge the different volumes together based on scan mode
    # Allow up to 2 volumes and rhis in 10 minutes
    volume1 = Set{String}()
    volume2 = Set{String}()
    rhis = Set{String}()
    scan_dict = Dict()
    
    cfrad_files = readdir("$(cfrad_raw)/$(date)"; join=true)
    filter!(!isdir,cfrad_files)
    
    # Files should be in chronological order
    for file in cfrad_files

        cfrad_ds = Dataset(file);
        scan_name = cfrad_ds.attrib["scan_name"]

        # Put scan names in a Dictionary in case we need it for later
        scan_dict[file] = scan_name
        
        if contains(scan_name, "RHI")
            # All RHIs go together
            push!(rhis,file)
        elseif contains(scan_name, "PICCOLO_LONG")
            # PICCOLO_LONG starts a volume
            push!(volume1,file)
        elseif contains(scan_name, "CIRL") && isempty(volume1)
            # The first circle long range starts a volume
            push!(volume1,file)
        elseif contains(scan_name, "CIRC") && isempty(volume2)
            # First CIRC volume
            push!(volume1,file)
        elseif contains(scan_name, "PICO_LONGVOL")
            # Single 10 minute long-range volume
            push!(volume1,file)
        elseif contains(scan_name, "VOL2")
            # Secondary volume
            push!(volume2,file)
        elseif contains(scan_name, "CIRL") && isempty(volume2)
            # The second circle long range starts a volume
            push!(volume2,file)
        elseif contains(scan_name, "CIRC") && !isempty(volume2)
            push!(volume2,file)
        else # VOL1, NEAR, FAR, HILO
            push!(volume1,file)
        end
    end

    # Aggregate the volumes into the merge directory
    println("Merging volumes...")
    cfrad_merge = temp_dir * "/cfrad_merge"

    if !isempty(volume1)
        run(`RadxConvert -ag_all -sort_rays_by_time -outdir $cfrad_merge -const_ngates -f $volume1`)
    end

    if !isempty(volume2)
        run(`RadxConvert -ag_all -sort_rays_by_time -outdir $cfrad_merge -const_ngates -f $volume2`)
    end

    if !isempty(rhis)
        run(`RadxConvert -ag_all -sort_rays_by_time -outdir $cfrad_merge -const_ngates -f $rhis`)
    end

    # QC the data
    println("Quality controlling the data...")
    cfrad_qc = temp_dir * "/cfrad_qc"
    merge_files = readdir("$(cfrad_merge)/$(date)"; join=true)
    filter!(!isdir,merge_files)
    for file in merge_files
        seapol_vol = read_cfradial(file, raw_moment_dict)
        fix_SEAPOL_RHOHV!(seapol_vol, raw_moment_dict)
        qc_moments = initialize_qc_fields(seapol_vol, raw_moment_dict, qc_moment_dict)
        qc_moments = threshold_qc(seapol_vol.moments, raw_moment_dict, qc_moments, qc_moment_dict)
        #qc_moments = despeckle(4, qc_moments, qc_moment_dict, length(seapol_vol.range), length(seapol_vol.azimuth) )
        qc_file = replace(file, "cfrad_merge" => "cfrad_qc")
        qc_file = replace(qc_file, "SEAPOL" => "SEAPOL_QC")
        write_qced_cfradial(file, qc_file, qc_moments, qc_moment_dict)
    end
    
    # Grid the data and use Radx2Grid for QC for now
    println("Gridding the data...")
    cart_grid_dir = temp_dir * "/gridded_data/cartesian"

    qc_files = readdir("$(cfrad_qc)/$(date)"; join=true)
    filter!(!isdir,qc_files)
    for file in qc_files
        radar_volume = read_cfradial(file, qc_moment_dict)
        output_file = "$(cart_grid_dir)/$(date)/gridded_vol_" *
            Dates.format(radar_volume.time[1], "YYYYmmdd") *
            "_" * Dates.format(radar_volume.time[1], "HHMMSS") * ".nc"
        @time grid_radar_composite(radar_volume, qc_moment_dict, output_file)
        #run(`Radx2Grid -params PICCOLO_cartesian_grid.params -outdir $cart_grid_dir -f $file`)
    end

    # Plot the data
    println("Plotting the data...")

    cart_grid_files = readdir("$(cart_grid_dir)/$(date)"; join=true)
    filter!(!isdir,cart_grid_files)
    for file in cart_grid_files
        
        # Plot the large base map
        plot_largemap(file, date, plot_dir, qc_moment_dict)
        
        # Plot a radar centered Cartesian map
        plot_composite(file, date, plot_dir, qc_moment_dict)
    end

    # Clean up and move to archive
    println("Archiving the data...")
    archive_files(temp_dir, archive_dir, collect([sigmet_files; cfrad_files; merge_files; qc_files; cart_grid_files]))

    println("Completed $date $time processing!")
end

