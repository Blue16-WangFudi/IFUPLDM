function models = MvM_UPLDM(X_train, Y_train, lambda1, lambda2, tau, C)
     % X_train: 训练集特征，大小为 NxD，其中 N 是样本数量，D 是特征维度
    % Y_train: 训练集标签，大小为 Nx1，其中 N 是样本数量

    classes = unique(Y_train); % 获取所有类别
    numClasses = length(classes); % 获取类别数量
    models = cell(numClasses, numClasses); % 为每一对类别分配一个模型
    Kernel = 'Linear'; % You can choose the kernel type based on your requirements

    % 对每一对类别进行训练
    for i = 1:numClasses
        for j = 1:numClasses
            if j ~= i
                % 生成二元标签，将类别i和类别j作为正负例
                binaryY_train = ones(size(Y_train));
                binaryY_train(Y_train == classes(j)) = -1;
                
                % 使用 TC_IF 进行训练
                s = TC_IF(X_train, binaryY_train, Kernel);

                % 使用 UPLDM 进行训练
                models{i, j} = Train_UPLDM(X_train, binaryY_train, lambda1, lambda2, tau, C*s, Kernel);
            end
        end
    end
end