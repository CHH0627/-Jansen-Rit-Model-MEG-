clear; clc; close all;

% 1. 讀取 CSV 檔案 (請確認檔名與您電腦中的一致)
filename = '17399~2555.csv';
data = readtable(filename);

% 2. 過濾掉無效或被懲罰的點 (Cost 異常高或為 NaN)
valid_idx = ~isnan(data.Cost) & data.Cost < 1000;
clean_data = data(valid_idx, :);

% 3. 提取每次迭代中 Cost 最小的點，作為「中心軌跡」
iterations = unique(clean_data.Iteration);
num_iters = length(iterations);

traj_j = zeros(num_iters, 1);
traj_P = zeros(num_iters, 1);
traj_cost = zeros(num_iters, 1);

for i = 1:num_iters
    % 取得單一迭代的所有數據
    iter_data = clean_data(clean_data.Iteration == iterations(i), :);
    
    % 找出該次迭代中 Cost 最低的點
    [min_c, min_idx] = min(iter_data.Cost);
    traj_j(i) = iter_data.j_Optimized(min_idx);
    traj_P(i) = iter_data.P_Optimized(min_idx);
    traj_cost(i) = min_c;
end


figure('Color', 'w', 'Name', 'Manifold Optimization Trajectory', 'Position', [100, 100, 900, 700]);
hold on;

% (A) 繪製所有有效的探索點 (Manifold 底層峽谷)
% 使用散佈圖，顏色深淺代表 Cost 高低
scatter(clean_data.j_Optimized, clean_data.P_Optimized, 20, clean_data.Cost, ...
    'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', 'Explored Optima (Manifold)');
colormap('parula');
cb = colorbar;
cb.Label.String = 'Optimization Cost';
cb.Label.FontWeight = 'bold';

% (B) 繪製中心點的迭代軌跡
plot(traj_j, traj_P, 'k-', 'LineWidth', 2, 'DisplayName', 'Center Trajectory');

% (C) 標示起點與終點
scatter(traj_j(1), traj_P(1), 150, 'g', 's', 'filled', 'MarkerEdgeColor', 'k', ...
    'LineWidth', 1.5, 'DisplayName', 'Start (Iter 1)');
scatter(traj_j(end), traj_P(end), 300, 'r', 'p', 'filled', 'MarkerEdgeColor', 'k', ...
    'LineWidth', 1.5, 'DisplayName', 'Final Optimum');

% 設定圖表外觀
xlabel('Connectivity Parameter (j)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('External Input (P)', 'FontSize', 12, 'FontWeight', 'bold');
title('Manifold Tracer Optimization Trajectory', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 11);
grid on;
box on;


fprintf('\n=== 迭代軌跡紀錄 ===\n');
for i = 1:num_iters
    fprintf('Iter %2d: (j = %.4f, P = %.4f), Cost = %.4f\n', iterations(i), traj_j(i), traj_P(i), traj_cost(i));
end