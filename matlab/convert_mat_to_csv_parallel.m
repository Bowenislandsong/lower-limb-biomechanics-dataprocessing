## Convert .mat files to .csv files in parallel
## this function is run in MATLAB and converts all .mat files in a folder to .csv files in parallel. It respects the folder structure and saves the .csv files in the same location as the .mat files.
## use the following to copy data from and to the server
## rsync -avz --progress --include="*/*/{fp,imu}/*.mat" --include="*/" --exclude="*" champange.usc.edu:/media/champagne/lower_limb_dataset/ ./
## rsync -avz --progress --include="*/" --include="*/*/{fp,imu,gcRight,gcLeft}/*.csv" --exclude="" ./ champange.usc.edu:/media/champagne/lower_limb_dataset/

function convert_mat_to_csv_parallel(folder_path, num_workers)
    % Check if folder exists
    if ~exist(folder_path, 'dir')
        error('Folder does not exist.');
    end
    
    % Get list of files in folder
    filelist = dir(fullfile(folder_path, '**','*.*'));
    files = filelist(~[filelist.isdir]);
    % Create parallel pool
    parpool(num_workers);
    
    % Parallel loop
    parfor i = 1:length(files)
        file_path = files(i).folder;
        file_name = files(i).name;
        full_file_path = fullfile(file_path, file_name);
        fprintf(full_file_path,'\n');
        
        % Check if file is a .mat file
        if endsWith(file_name, '.mat')
            % Load MAT file
            f = load(full_file_path);
            
            % Check if data is a struct
            if ~isstruct(f)
                warning('Skipping %s: Loaded data is not a struct.', file_name);
                continue;
            end
            
            % Convert struct to table
            data_table = f.data;
            
            % Replace .mat with .csv
            csv_file_name = strrep(full_file_path, '.mat', '.csv');
            
            % Write table to CSV file
            writetable(data_table, csv_file_name);
            fprintf('Converted %s to %s\n', full_file_path, csv_file_name);
        end
    end
    
    % Delete parallel pool
    delete(gcp);
end