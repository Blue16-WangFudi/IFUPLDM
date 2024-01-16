function models = OVA_UPLDM(X_train, Y_train, lambda1, lambda2, tau, C)

    % X: 训练集特征，大小为NxD，其中N是样本数量，D是特征维度
    % Y: 训练集标签，大小为Nx1，其中N是样本数量

    classes = unique(Y_train); %获取所有类别
    numClasses = length(classes); %获取类别数量
    models = cell(numClasses, 1); %为每一个类别分配一个模型
%     Kernel='RBF';
    Kernel='Linear';

    % 对每一个类别进行训练
    for i = 1:numClasses
        % 生成二元标签
        binaryY_train = -1 * ones(size(Y_train));
        binaryY_train(Y_train == classes(i)) = 1;
        s = TC_IF(X_train, binaryY_train, Kernel);
%         s = IF(X_train, binaryY_train, Kernel);
        % 使用 UPLDM 进行训练
        models{i} = Train_UPLDM( X_train, binaryY_train, lambda1, lambda2, tau, C*s, Kernel);
    end


end