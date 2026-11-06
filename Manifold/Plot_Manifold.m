% =========================================================================
% 批次讀取資料夾內 Manifold_Tracer_Results_j*_P*.csv 並繪製於同一張圖
% 軌跡的第 1 個點設為：第 1 次迭代選取的 (j_Initial, P_Initial) 中心點 P1
% =========================================================================
clear; clc; close all;

% 1. 設定資料夾路徑與檔案搜尋規則
folder_path = 'C:\Users\cheng\OneDrive\Desktop\code\科導程式\-Jansen-Rit-Model-MEG-\Manifold'; 
file_pattern = fullfile(folder_path, 'Manifold_Tracer_Results*.csv');
file_list = dir(file_pattern);

num_files = length(file_list);
if num_files == 0
    error('在資料夾 "%s" 中找不到符合 Manifold_Tracer_Results_j*_P*.csv 的檔案！', folder_path);
end
fprintf('共找到 %d 個 CSV 檔案，開始讀取與解析...\n', num_files);

% 2. 預先讀取所有檔案，提取初始中心點 P1 並過濾有效探索點
all_clean_data   = table();
file_data_cell   = cell(num_files, 1);
run_labels       = cell(num_files, 1);
init_centers_P1  = nan(num_files, 2); % 儲存每個檔案第 1 次迭代的 (j_Initial, P_Initial) 中心點

for f = 1:num_files
    fname = file_list(f).name;
    full_path = fullfile(file_list(f).folder, fname);
    temp_data = readtable(full_path);
    
    % ---------------------------------------------------------------------
    % 關鍵修改：從「未過濾」的原始資料中，抓取第 1 次迭代的初始網格中心點
    % ---------------------------------------------------------------------
    if ~isempty(temp_data)
        if ismember('RunTime', temp_data.Properties.VariableNames)
            raw_groups = strcat(string(temp_data.RunTime), '_', string(temp_data.Iteration));
            [~, ~, raw_iter_id] = unique(raw_groups, 'stable');
        else
            [~, ~, raw_iter_id] = unique(temp_data.Iteration, 'stable');
        end
        first_iter_mask = (raw_iter_id == 1);
        
        % 對對稱網格 (-5:5) 的初始座標取平均，即可精準還原中心點 P1
        j_init_center = mean(temp_data.j_Initial(first_iter_mask), 'omitnan');
        P_init_center = mean(temp_data.P_Initial(first_iter_mask), 'omitnan');
        init_centers_P1(f, :) = [j_init_center, P_init_center];
    end
    
    % 過濾掉無效或被懲罰的點 (Cost 異常高或為 NaN)，供後續繪製 Manifold 與最佳收斂點
    valid_idx = ~isnan(temp_data.Cost) & temp_data.Cost < 1000;
    clean_temp = temp_data(valid_idx, :);
    
    file_data_cell{f} = clean_temp;
    all_clean_data = [all_clean_data; clean_temp]; %#ok<AGROW>
    
    % 使用 regexp 解析檔名中的 j、P 數值與附加標記 (例如 (div))
    tokens = regexp(fname, ...
        '^Manifold_Tracer_Results_j([+-]?\d+\.?\d*)_P([+-]?\d+\.?\d*)(.*)\.csv$', ...
        'tokens', 'once');
    
    if ~isempty(tokens)
        j_str   = tokens{1};
        P_str   = tokens{2};
        tag_str = strtrim(tokens{3}); % 例如 '(div)' 或空字串
        
        if ~isempty(tag_str)
            run_labels{f} = sprintf('j=%s, P=%s %s', j_str, P_str, tag_str);
        else
            run_labels{f} = sprintf('j=%s, P=%s', j_str, P_str);
        end
    else
        [~, name_only, ~] = fileparts(fname);
        run_labels{f} = erase(name_only, 'Manifold_Tracer_Results_');
    end
end

% =========================================================================
% 3. 繪製合併流形與各檔案收斂軌跡
% =========================================================================
figure('Color', 'w', 'Name', 'Multi-File Manifold Trajectory Comparison', ...
       'Position', [100, 100, 1050, 750]);
hold on;

% (A) 繪製所有檔案合併後的有效探索點 (Manifold 底層峽谷)
scatter(all_clean_data.j_Optimized, all_clean_data.P_Optimized, 20, all_clean_data.Cost, ...
    'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Explored Optima (Manifold)');
colormap('parula');
cb = colorbar;
cb.Label.String = 'Optimization Cost';
cb.Label.FontWeight = 'bold';

% 根據檔案數量自動產生高對比色票
line_colors = lines(num_files);

fprintf('\n=== 各檔案迭代軌跡紀錄 ===\n');

% (B) 逐一計算並繪製每個檔案的中心軌跡 (P1 -> Iter 1 Best -> ... -> Final Optimum)
for f = 1:num_files
    run_data = file_data_cell{f};
    p1_center = init_centers_P1(f, :);
    
    if isempty(run_data)
        fprintf('\n[%s] 無有效收斂點，僅標示初始中心 P1(%.4f, %.4f)。\n', ...
            file_list(f).name, p1_center(1), p1_center(2));
        continue;
    end
    
    % 結合 RunTime 與 Iteration 進行分組
    if ismember('RunTime', run_data.Properties.VariableNames)
        group_keys = strcat(string(run_data.RunTime), '_', string(run_data.Iteration));
        [~, ~, iter_groups] = unique(group_keys, 'stable');
    else
        [~, ~, iter_groups] = unique(run_data.Iteration, 'stable');
    end
    num_iters = max(iter_groups);
    
    opt_j    = zeros(num_iters, 1);
    opt_P    = zeros(num_iters, 1);
    opt_cost = zeros(num_iters, 1);
    
    for i = 1:num_iters
        iter_data = run_data(iter_groups == i, :);
        [min_c, min_idx] = min(iter_data.Cost);
        opt_j(i)    = iter_data.j_Optimized(min_idx);
        opt_P(i)    = iter_data.P_Optimized(min_idx);
        opt_cost(i) = min_c;
    end
    
    % ---------------------------------------------------------------------
    % 關鍵修改：將第 1 次選取的初始中心點 P1 放在軌跡陣列的第 1 個位置
    % 完整軌跡長度為 num_iters + 1：從 P1 出發，依序連向每次迭代的最佳收斂點
    % ---------------------------------------------------------------------
    traj_j = [p1_center(1); opt_j];
    traj_P = [p1_center(2); opt_P];
    
    c_color = line_colors(f, :);
    
    % 繪製中心點的完整迭代軌跡線 (包含從 P1 跳入流形的第一步)
    plot(traj_j, traj_P, '-', 'Color', c_color, 'LineWidth', 2.2, ...
        'DisplayName', sprintf('Traj [%s]', run_labels{f}));
    
    % 標示起點：第 1 次選取的 (j_Initial, P_Initial) 中心點 P1 (方形 's')
    scatter(traj_j(1), traj_P(1), 140, c_color, 's', 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.2, ...
        'DisplayName', sprintf('Start P1 [%s]', run_labels{f}));
    
    % 標示最終收斂終點 (星形 'p')
    scatter(traj_j(end), traj_P(end), 280, c_color, 'p', 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Optimum [%s]', run_labels{f}));
        
    % 在命令視窗輸出該檔案的軌跡資訊
    fprintf('\n[%s] (標籤: %s)\n', file_list(f).name, run_labels{f});
    fprintf('  -> 初始中心 P1       : (j = %.4f, P = %.4f)\n', traj_j(1), traj_P(1));
    fprintf('  -> Iter 1 最佳收斂點 : (j = %.4f, P = %.4f), Cost = %.4f\n', ...
        opt_j(1), opt_P(1), opt_cost(1));
    fprintf('  -> 最終收斂 (Iter %d): (j = %.4f, P = %.4f), Cost = %.4f\n', ...
        num_iters, traj_j(end), traj_P(end), opt_cost(end));
end

% 設定圖表外觀
xlabel('Connectivity Parameter (j)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('External Input (P)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Comparison of Manifold Traces from Initial Centers P_1 (%d Files)', num_files), ...
    'FontSize', 14, 'FontWeight', 'bold');

legend('Location', 'best', 'FontSize', 9, 'Interpreter', 'none');
grid on;
box on;