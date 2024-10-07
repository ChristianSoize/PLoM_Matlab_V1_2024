function sub_setupParallelPool(nbworkers)
    % This function creates a parallel pool with the specified number of workers.
    % It checks if a pool already exists and validates the requested number of workers.

    % Turn off warning backtrace for cleaner warning messages
    warning('off', 'backtrace');

    % Check if Parallel Computing Toolbox is available
    if isempty(ver('parallel'))
        error('Parallel Computing Toolbox is not available.');
    end
    
    % Check if the requested number of workers is valid
    if nbworkers < 1
        error('The number of workers must be at least 1.');
    end

    % Get the maximum number of available workers on the system
    clusterInfo = parcluster;  % Get cluster information
    maxAvailableWorkers = clusterInfo.NumWorkers;  % Maximum available workers on the machine

    % Check if the requested number of workers exceeds the available workers
    if nbworkers > maxAvailableWorkers
        warning('Requested %d workers, but only %d are available. Using %d workers instead.', nbworkers, maxAvailableWorkers, maxAvailableWorkers);
        nbworkers = maxAvailableWorkers; % Adjust to the maximum available workers
    end

    % Check if a parallel pool already exists
    pool = gcp('nocreate'); % Get current pool if it exists
    if isempty(pool)
        % No pool exists, create a new one with the specified number of workers
        try
            parpool(nbworkers); % Create parallel pool
            fprintf('Parallel pool created with %d workers.\n', nbworkers);
        catch ME
            warning('Failed to create parallel pool: %s', ME.message);
        end
    else
        % A pool already exists, check if the size matches the requested number of workers
        if pool.NumWorkers == nbworkers
            fprintf('Parallel pool already running with %d workers.\n', nbworkers);
        else
            fprintf('A parallel pool is already running with %d workers. Shutting it down...\n', pool.NumWorkers);
            delete(pool); % Close the existing pool
            try
                parpool(nbworkers); % Create new pool with the desired number of workers
                fprintf('Parallel pool created with %d workers.\n', nbworkers);
            catch ME
                warning('Failed to create parallel pool: %s', ME.message);
            end
        end
    end
end

