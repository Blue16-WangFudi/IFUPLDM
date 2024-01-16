function Ypred = predict_OVO_UPLDM(models, Samples_Predict, label_Train)
    numModels = length(models);
    numClasses = length(unique(label_Train));
    
    scores = zeros(size(Samples_Predict, 1), numClasses); % Initialize scores matrix
    
    % Iterate over each model (pairwise comparison)
    for i = 1:numModels
        scores(:, i) = Predict_UPLDM(models{i}, Samples_Predict);
    end
    
    % Sum the scores for each class over all models
    classScores = zeros(size(Samples_Predict, 1), numClasses);
    modelIndex = 1;
    
    for i = 1:numClasses
        for j = i+1:numClasses
            classScores(:, i) = classScores(:, i) + scores(:, modelIndex);
            classScores(:, j) = classScores(:, j) - scores(:, modelIndex);
            
            modelIndex = modelIndex + 1;
        end
    end
    
    % Find the class with the highest score for each sample
    [~, Ypred] = max(classScores, [], 2);
end
