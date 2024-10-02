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