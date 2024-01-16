function Ypred = predict_MVM_UPLDM(models, Samples_Predict,~)
    % models: 训练得到的模型矩阵，大小为 numClasses x numClasses
    % Samples_Predict: 测试集特征，大小为 MxD，其中 M 是样本数量，D 是特征维度

    numClasses = size(models, 1);
    scores = zeros(size(Samples_Predict, 1), numClasses); % 初始化分数矩阵

    % 对每一对类别进行预测
    for i = 1:numClasses
        for j = 1:numClasses
            if j ~= i
                % 使用 UPLDM 进行预测
                scores(:, i) = scores(:, i) + Predict_UPLDM(models{i, j}, Samples_Predict);
            end
        end
    end

    % 找到分数最高的类别
    [~, Ypred] = max(scores, [], 2);
end