function Ypred = predict_OVA_UPLDM(model, Samples_Predict,label_Train)
    % X: 测试集特征，大小为MxD，其中M是样本数量，D是特征维度

%     numClasses = length(model.classes); %获取类别数量
    numClasses = length(unique(label_Train));
    scores = zeros(size(Samples_Predict, 1), numClasses); %初始化分数矩阵
   
    % 对每一个类别进行预测
    for i = 1:numClasses
        scores(:, i) = Predict_UPLDM(model{i}, Samples_Predict);
    end
    % 找到分数最高的类别
    [~, Ypred] = max(scores, [], 2);
end