function X = noise(std, data)

    % 假设原始数据集保存在变量data中
    % 添加均值为0，标准差为std的高斯噪声
    noise = std*randn(size(data));
    noisy_data = data + noise;
    X = noisy_data;

end