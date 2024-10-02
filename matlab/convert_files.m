%% Convert files
function convert_files(varargin)
    if nargin < 1
        error('Not enough input arguments.');
    end
    
    file_name = varargin{1};
    
    % Check if file exists
    if ~exist(file_name, 'file')
        error('File does not exist.');
    end
    
    % Load MAT file
    f = load(file_name);
    
    % Check if data is a struct
    if ~isstruct(f)
        error('Loaded data is not a struct.');
    end
    
    % Convert struct to table
    data_table = f.data;
    
    % Replace .mat with .csv
    csv_file_name = strrep(file_name, '.mat', '.csv');
    
    % Write table to CSV file
    writetable(data_table, csv_file_name);
end