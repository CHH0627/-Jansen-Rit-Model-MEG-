% =========================================================================
% 重新編號 Manifold Tracer 的 Iteration
% =========================================================================
clear; clc;

% 1. 指定您的 CSV 檔案名稱
filename = 'Manifold_Tracer_Results(125333,26896~122164,23790).csv';

% 2. 讀取資料表
data = readtable(filename);

% 3. 確保資料按照 'RunTime' 排序 (若有跨次執行的資料)
% 如果您的 MATLAB 版本較舊，不支援 sortrows 針對字串/日期排序，
% 也可以單純依賴原本資料由上往下讀取的順序來分組
try
    data = sortrows(data, {'RunTime', 'Iteration'});
catch
    disp('跳過排序，依照原始資料列順序進行重編號。');
end

% 4. 找出時間與原始迭代次數的唯一組合，以進行分組
% 將 RunTime 和 Iteration 結合成一個識別字串
group_id_str = strcat(string(data.RunTime), '_', string(data.Iteration));

% 使用 unique 找出不重複的群組 (保持原始出現順序 'stable')
[~, ~, ic] = unique(group_id_str, 'stable');

% 5. 將重新編號的結果 (ic 就是 1, 2, 3...) 寫回 Iteration 欄位
data.Iteration = ic;

% 6. 覆蓋原本的 CSV 檔案
writetable(data, filename);

% 7. 顯示結果
max_iter = max(ic);
fprintf('=== 重新編號完成！===\n');
fprintf('總共將資料整理為 %d 次連續迭代。\n', max_iter);