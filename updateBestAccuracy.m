function result = updateBestAccuracy(result, paramName, paramValue, params, accuracy, j)
    % 设置当前参数值
    params.(paramName) = paramValue;

    % 创建新的表格行
    newRow = struct2cell(params);
    newRow = [newRow; {accuracy}];
    
    % 查找当前参数值是否已存在
    existingRow = find(result.(paramName){:,j} == paramValue, 1);
    if isempty(existingRow)
        % 如果参数值不存在，添加新行
        result.(paramName) = [result.(paramName); newRow'];
    else
        % 如果参数值已存在，比较准确率
        if accuracy > result.(paramName){existingRow, end}
            % 如果新准确率更高，则删除旧记录并添加新记录
            result.(paramName)(existingRow, :) = [];
            result.(paramName) = [result.(paramName); newRow'];
%         elseif accuracy == result.(paramName){existingRow, end}
%             % 如果新旧准确率相等，则新增一条记录
%             result.(paramName) = [result.(paramName); newRow'];
        end
    end
end
