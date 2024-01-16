clear,clc;
load batchs;
warning('off','all');
%噪声
% load batch1_p_noise;
% batchS = train_set.data;
% batchS_label=train_set.labels';
%噪声处理
%非噪声
batchS = batch1_p(:,4:2:end);
batchS_label=batch1_p(:,1);
% 假设原始数据集为X和Y，其中X是特征矩阵，Y是标签向量（1维）
% 假设样本总数为N，类别总数为C
% 设定标签噪声比例（例如，将10%的样本标签改为其它类别）
noise_ratio = 0.1;
N = length(batchS);
C = length(unique(batchS_label));
% 计算需要改变标签的样本数量
num_noisy_samples = round(noise_ratio * N);

% 随机选择需要改变标签的样本
noisy_samples_idx = randperm(N, num_noisy_samples);

% 标签噪声实验
for i = 1:num_noisy_samples
    % 生成新的标签类别，不与原标签相同
    new_label = batchS_label(noisy_samples_idx(i));
    while new_label == batchS_label(noisy_samples_idx(i))
        new_label = randi(C);
    end
    
    % 将随机选择的样本的标签修改为新的标签类别
    batchS_label(noisy_samples_idx(i)) = new_label;
end
%高斯噪声实验
% mm = Normalizer(1, batchS);
% batchS = mm.transform(batchS);
% batchS = noise(0.1, batchS);
% mm = Normalizer(1, batchS);
% batchS = mm.transform(batchS);
% % % 保存结构体到.mat文件
% train_set.data = batchS;
% train_set.labels = batchS_label;
% save('batch1_p_noise.mat', 'train_set');

ind_lambda = 10.^(-6:6);
ind_d = 128;
C1_Interval = 2.^(-6:-4);
ind_lambda1 = 2.^(-6:-2);
ind_lambda2 = 2.^(-6:-2);
ind_tau= 0:0.2:1;
best_d = [];
best_lambda = [];
best_C = [];
best_lambda1 = [];
best_lambda2 = [];
best_tau = [];
best_accuracy = [];
NUMTIMES = length(ind_lambda)*length(ind_d)*length(C1_Interval)*length(ind_lambda1)*length(ind_lambda2)*length(ind_tau)*5;
for i = 5
    best_temp_accuracy = 0;
    best_temp_lambda = 0;
    best_temp_d = 0;
    best_temp_C = 0;
    best_temp_lambda1 = 0;
    best_temp_lambda2 = 0;
    best_temp_tau = 0;
    value = eval(['batch',num2str(i),'_p']); % 访问变量
    batchT = value(:,4:2:end);
    batchT_label=value(:,1);
%     mm = Normalizer(1, batchT);
%     batchT = mm.transform(batchT);
    temp_NUMTIMES = length(ind_lambda)*length(ind_d)*length(C1_Interval)*length(ind_lambda1)*length(ind_lambda2)*length(ind_tau);
    %寻找最优参数组合
    for lambda_ =ind_lambda
        for d = ind_d
            for C = C1_Interval
                [batchS_P,batchT_P]=DRCA(batchS,batchT,lambda_,d);
                batchS_P = real(batchS_P);
                batchT_P = real(batchT_P);
                for lambda1 = ind_lambda1
                    for lambda2 = ind_lambda2
                        for tau = ind_tau
                            tic
                            model = OVA_UPLDM(batchS_P, batchS_label, lambda1, lambda2, tau, C);
                            Ypred = predict_OVA_UPLDM(model, batchT_P,batchS_label);
                            temp_accuracy = sum(Ypred == batchT_label) / length(batchT_label);
                            if temp_accuracy > best_temp_accuracy
                                 best_temp_accuracy = temp_accuracy;
                                 best_temp_lambda = lambda_;
                                 best_temp_d = d;
                                 best_temp_C = C;
                                 best_temp_lambda1 = lambda1;
                                 best_temp_lambda2 = lambda2;
                                 best_temp_tau = tau;
                            end
                            NUMTIMES = NUMTIMES-1;
                            time = toc;
                            temp_NUMTIMES = temp_NUMTIMES - 1;
                            timeleft = (time  * NUMTIMES)/3600;
                            temp_timeleft = (time * temp_NUMTIMES)/60;
                            disp(['当前批次：' num2str(i) '' ,'剩余' num2str(temp_NUMTIMES) ' 次运行，','预计还需 ' num2str(temp_timeleft) ' 分钟'])
                            disp(['总剩余 ' num2str(NUMTIMES) ' 次运行，','预计还需 ' num2str(timeleft) ' 小时' ]);
                            disp(['当前参数组合  lambda:' num2str(lambda_) ' ' ,'d:' num2str(d)  ' ' , 'C:' num2str(C) ' ' ,...
                                'lambda1:' num2str(lambda1) ' ' , 'lambda2:' num2str(lambda2) ' ' , 'tau:' num2str(tau)])
                            disp(['当前最优参数组合  lambda:' num2str(best_temp_lambda) ' ' ,'d:' num2str(best_temp_d)  ' ' , 'C:' num2str(best_temp_C) ' ' ,...
                                'lambda1:' num2str(best_temp_lambda1) ' ' , 'lambda2:' num2str(best_temp_lambda2) ' ' , 'tau:' num2str(best_temp_tau)])
                            disp(['当前准确率：' num2str(temp_accuracy)]);
                            disp(['当前最优准确率：' num2str(best_temp_accuracy)]);
                            disp(" ");
                        end
                    end
                end
            end
         end
    end
    best_accuracy = [best_accuracy,best_temp_accuracy];
    best_lambda = [best_lambda,best_temp_lambda];
    best_d = [best_d,best_temp_d];
    best_C = [best_C,best_temp_C];
    best_lambda1 = [best_lambda1,best_temp_lambda1];
    best_lambda2 = [best_lambda2,best_temp_lambda2];
    best_tau = [best_tau,best_temp_tau];
end


