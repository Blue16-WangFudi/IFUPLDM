function models = OVO_UPLDM(X_train, Y_train, lambda1, lambda2, tau, C)
    numClasses = length(unique(Y_train)); % Get class count
    models = cell(numClasses * (numClasses - 1) / 2, 1); % Allocate models for each pairwise comparison
    Kernel = 'Linear'; % You can choose the kernel type
    
    modelIndex = 1;
    
    % Iterate over each pair of classes
    for i = 1:numClasses
        for j = i+1:numClasses
            % Create binary labels for the pair
            binaryY_train = -1 * ones(size(Y_train));
            binaryY_train(Y_train == i) = 1;
            binaryY_train(Y_train == j) = -1;

            % Train UPLDM model for the pair
            s = TC_IF(X_train, binaryY_train, Kernel);
            models{modelIndex} = Train_UPLDM(X_train, binaryY_train, lambda1, lambda2, tau, C*s, Kernel);
            
            modelIndex = modelIndex + 1;
        end
    end
end
