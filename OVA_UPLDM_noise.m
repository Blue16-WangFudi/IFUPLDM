function models = OVA_UPLDM_noise(X_train, Y_train, lambda1, lambda2, tau, C)

    % X: 训练集特征，大小为NxD，其中N是样本数量，D是特征维度
    % Y: 训练集标签，大小为Nx1，其中N是样本数量

    classes = unique(Y_train); %获取所有类别
    numClasses = length(classes); %获取类别数量
    models = cell(numClasses, 1); %为每一个类别分配一个模型
    Kernel='Linear';

    % 对每一个类别进行训练
    for i = 1:numClasses
        % 生成二元标签
        binaryY_train = -1 * ones(size(Y_train));
        binaryY_train(Y_train == classes(i)) = 1;
%%所有样本随机选择
%         % 假设我们有1000个样本
%         num_samples = length(Y_train);
%         % 计算要更改的样本数量
%         num_noisy_samples = round(0.1 * num_samples);
%         % 随机选择10%的样本
%         random_indices = randperm(num_samples, num_noisy_samples);
%         % 将选择的样本的标签更改为其相反类
%         binaryY_train(random_indices) =  - binaryY_train(random_indices);
%%正负类同比例选择
% 假设我们有1000个样本
        num_samples = length(Y_train);
        % 计算要更改的样本数量
        num_noisy_samples = round(0.1 * num_samples);
        % 随机选择正类样本的索引
        positive_indices = find(binaryY_train == 1);
        random_positive_indices = randsample(positive_indices, round(num_noisy_samples/2));
        % 随机选择负类样本的索引
        negative_indices = find(binaryY_train == -1);
        random_negative_indices = randsample(negative_indices, round(num_noisy_samples/2));
        % 将选择的样本的标签更改为其相反类
        binaryY_train(random_positive_indices) = -1;
        binaryY_train(random_negative_indices) = 1;
        s = TC_IF(X_train, binaryY_train, Kernel); 
        % 使用 UPLDM 进行训练
        models{i} = Train_UPLDM( X_train, binaryY_train, lambda1, lambda2, tau, C*s, Kernel);
    end


end