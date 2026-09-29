% Jansen-Rit MEG fit: main result only. Requires Optimization Toolbox.
% Keep FingerLifting4Student102.mat in the working folder.
% Main multi-start optimization is retained; post-fit initialization tests are removed.
clear; clc; close all;
rng(1,'twister');              % reproducible multi-start experiment
delete(gcp("nocreate"));
parpool('local', 16);  % adjust to your CPU core count

%% 1. Settings
fit_mode = 'joint_jP';
fixed_j = 14.00;
paper_j_bounds = [10, 14];
paper_P_bounds = [-4, 4];
paper_seed = [12.5338, 2.6901];   % [j,P], previous data-priority optimum
require_sustained_oscillation = true;

%% 2. MEG data
if ~exist('FingerLifting4Student102.mat', 'file')
    error(['Cannot find FingerLifting4Student102.mat. ', ...
           'Place the data file in the same folder as this script.']);
end
load('FingerLifting4Student102.mat');   % must contain pdata
channel_id = 13;
data_range = 12000:28000;
y_data_raw = pdata.PdataF(data_range, channel_id);
y_data_raw = y_data_raw(:);
fs = pdata.sample_fif;
N = length(y_data_raw);
t_exp = (0:N-1)'/fs;
y_data = normalize01(y_data_raw);
f_data = get_dominant_freq(y_data, fs);
f_target = f_data;
f_scale = max(abs(f_target), 1);
signal_cfg.psd_band = [3.4 4.6];
signal_cfg.raw_frequency_band = [0.5 15];
signal_cfg.min_raw_std = 1e-4;       % model-output units (mV proxy)
signal_cfg.min_sustain_ratio = 0.50; % std(second half)/std(first half)
signal_cfg.max_sustain_ratio = 2.00;
signal_cfg.require_sustained_oscillation = require_sustained_oscillation;
data_features = build_signal_features( y_data,fs,signal_cfg.psd_band);
data_features.signal_cfg = signal_cfg;

%% 3. Cost weights
weights.waveform = 0.35;
weights.envelope = 0.20;
weights.psd      = 0.15;
weights.frequency = 0.10;
weights.phase    = 0.20;
w_sum = weights.waveform + weights.envelope + weights.psd + weights.frequency + weights.phase;
weights.waveform = weights.waveform/w_sum;
weights.envelope = weights.envelope/w_sum;
weights.psd = weights.psd/w_sum;
weights.frequency = weights.frequency/w_sum;
weights.phase = weights.phase/w_sum;
n_lock = 1;
m_lock = 1;

%% 4. Model and bounds
lb = [paper_j_bounds(1), paper_P_bounds(1)];   % full [j,P] bounds
ub = [paper_j_bounds(2), paper_P_bounds(2)];
if ~isscalar(fixed_j) || ~isfinite(fixed_j) || fixed_j <= 0
    error('fixed_j must be a finite positive scalar.');
end
switch lower(fit_mode)
    case 'p_only'
        fixed_paper_params = [fixed_j, NaN];
        theta_seed_full = [fixed_j, paper_seed(2)];
        num_starts = 30;
    case 'joint_jp'
        fixed_paper_params = [NaN, NaN];
        theta_seed_full = paper_seed;
        num_starts = 20;
    otherwise
        error('fit_mode must be either ''P_only'' or ''joint_jP''.');
end
var_idx = isnan(fixed_paper_params);
lb_var = lb(var_idx);
ub_var = ub(var_idx);
theta_seed_var = theta_seed_full(var_idx);
if any(theta_seed_var < lb_var) || any(theta_seed_var > ub_var)
    error('The free-coordinate seed must lie inside its paper bounds.');
end
basePar = defaultJansenParameters();
sim.T_trans = 5;         % fitting transient, seconds
sim.final_timeout = 8;   % timeout for one model evaluation, seconds
sim.RelTol = 1e-6;
sim.AbsTol = 1e-8;
sim.MaxStep = min(1e-2, 1/fs);
sim.X0 = [0.05; 8; 1; 0; 0; 0];

%% 5. Initial Parameter Dependence Test
opts_mstart = optimoptions('lsqnonlin', 'Display', 'off', 'MaxIterations', 80, ...
    'MaxFunctionEvaluations', 800);

% 定義測試的初始參數網格
test_j_seeds = linspace(10, 14, 20); 
test_P_seeds = linspace(-4, 4, 40);
[J_GRID, P_GRID] = meshgrid(test_j_seeds, test_P_seeds);

% 將 2D 網格攤平成 1D 陣列以利 parfor 分配工作
num_points = numel(J_GRID);
J_GRID_flat = J_GRID(:);
P_GRID_flat = P_GRID(:);

% 預先配置結果陣列
J_OPT_flat = zeros(num_points, 1);
P_OPT_flat = zeros(num_points, 1);
COST_flat  = zeros(num_points, 1);

obj_fun = @(theta_var) jansen_residual_weighted( ...
    apply_fixed_params(theta_var,fixed_paper_params),basePar,sim, ...
    t_exp, y_data, data_features, fs, f_target, f_scale, weights, n_lock, m_lock);

fprintf('\n=== 開始平行測試初始參數相依性 (共 %d 個點) ===\n', num_points);

% 1. 建立彈出式視覺化進度條
h_wait = waitbar(0, '準備啟動平行運算池 (可能需要幾秒鐘)...');
D = parallel.pool.DataQueue;

% 確保進度計數器歸零
clear update_waitbar; 

% 2. 綁定進度條更新函數
afterEach(D, @(~) update_waitbar(h_wait, num_points));

% 啟動平行運算迴圈
parfor i = 1:num_points
    theta0_var = [J_GRID_flat(i), P_GRID_flat(i)];
    
    try
        [theta_opt_var, resnorm] = lsqnonlin(obj_fun, theta0_var, lb_var, ub_var, opts_mstart);
        J_OPT_flat(i) = theta_opt_var(1);
        P_OPT_flat(i) = theta_opt_var(2);
        COST_flat(i)  = resnorm;
    catch
        J_OPT_flat(i) = NaN; 
        P_OPT_flat(i) = NaN; 
        COST_flat(i)  = NaN;
    end
    
    % 3. 單一任務完成時，發送訊號更新進度條
    send(D, i);
end

% 運算結束，自動關閉進度條視窗
if isvalid(h_wait), close(h_wait); end

% 將 1D 結果還原回 2D 矩陣形狀
J_OPT_RES = reshape(J_OPT_flat, size(J_GRID));
P_OPT_RES = reshape(P_OPT_flat, size(P_GRID));

% 統計收斂點與自動設定最佳參數
% 過濾掉計算失敗 (NaN) 的點
valid_idx = ~isnan(COST_flat);
J_valid = J_OPT_flat(valid_idx);
P_valid = P_OPT_flat(valid_idx);
C_valid = COST_flat(valid_idx);

% 四捨五入到小數點後 4 位，將微小差異歸類為同一個收斂點
JP_rounded = round([J_valid, P_valid], 4);
[uJP, ~, ic] = unique(JP_rounded, 'rows');
counts = accumarray(ic, 1);

% 依照收斂點數量由多到少進行排序
[counts_sorted, sort_idx] = sort(counts, 'descend');
uJP_sorted = uJP(sort_idx, :);

fprintf('\n=== 收斂點統計 ===\n');
for k = 1:size(uJP_sorted, 1)
    fprintf('P%d (%.4f, %.4f): %d 個點\n', k, uJP_sorted(k,1), uJP_sorted(k,2), counts_sorted(k));
end

% 自動找出全域最低 Cost 的點
[min_cost, min_idx] = min(C_valid);
best_j = J_valid(min_idx);
best_P = P_valid(min_idx);

fprintf('\n=== 最佳化目標更新 ===\n');
fprintf('全域最低 Cost = %.4f，位於 (j=%.4f, P=%.4f)\n', min_cost, best_j, best_P);
fprintf('已將此點自動設定為後續 P scan 與診斷圖表的目標參數。\n');

% 重新計算最佳參數的物理參數與動態資訊 (供後續區塊使用)
[best_C, best_p] = paperToDimensional(best_j, best_P, basePar);
[~, best_info] = jansen_residual_weighted( ...
    [best_j, best_P], basePar, sim, ...
    t_exp, y_data, data_features, fs, f_target, f_scale, weights, n_lock, m_lock);

% 額外輸出：收斂軌跡向量圖
figure('Color','w', 'Name', 'Initial Parameter Dependence');
hold on;
% 畫出收斂向量
quiver(J_GRID, P_GRID, J_OPT_RES - J_GRID, P_OPT_RES - P_GRID, 0, ...
    'Color', [0.6 0.6 0.6], 'MaxHeadSize', 0.5, 'LineWidth', 1);
% 標示起始點與收斂終點
scatter(J_GRID(:), P_GRID(:), 20, 'b', 'filled', 'MarkerEdgeColor', 'k');
scatter(uJP_sorted(:,1), uJP_sorted(:,2), 80, 'r', 'p', 'filled', 'MarkerEdgeColor', 'k');
% 特別標示全域最佳解
scatter(best_j, best_P, 150, 'y', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

xlabel('Connectivity Parameter (j)');
ylabel('External Input (P)');
title('Convergence from Different Initial Seeds');
legend('Convergence Path', 'Initial Seeds', 'Local Minima', 'Global Best Minimum', 'Location', 'best');
grid on; box on;

%% 6. P scan

N_grid = 81;
P_grid = linspace(lb(2),ub(2),N_grid);
scan_cost = nan(N_grid,1);
valid_fit_profile = false(N_grid,1);
f_raw_profile = nan(N_grid,1);
f_filtered_profile = nan(N_grid,1);
R_profile = nan(N_grid,1);
parfor i = 1:N_grid
    theta_scan = [best_j,P_grid(i)];
    try
        [residual_scan,info_scan] = jansen_residual_weighted( theta_scan,basePar,sim, ...
            t_exp,y_data,data_features,fs,f_target,f_scale, weights,n_lock,m_lock);
        valid_fit_profile(i) = info_scan.is_valid_fit;
        if info_scan.is_valid_fit
            scan_cost(i) = sum(residual_scan.^2);
        end
        f_raw_profile(i) = info_scan.f_dom;
        f_filtered_profile(i) = info_scan.f_dom_filtered;
        R_profile(i) = info_scan.R;
    catch ME
        warning('%s',ME.message);
    end
end

%% 7. Cost threshold
valid_scan = valid_fit_profile & isfinite(scan_cost);
if any(valid_scan)
    J_min = min(scan_cost(valid_scan));
    valid_indices = find(valid_scan);
    [~,local_min_index] = min(scan_cost(valid_scan));
    grid_min_index = valid_indices(local_min_index);
else
    J_min = best_cost;
    grid_min_index = NaN;
end
relative_tolerance = 0.05;
threshold_line = J_min*(1+relative_tolerance);
if isfinite(grid_min_index) && (grid_min_index == 1 || grid_min_index == N_grid)
    warning(['The smallest conditional cost is at a P-grid boundary. ', ...
             'Do not call this an interior optimum before extending paper_P_bounds.']);
end

%% 8. Best-fit trajectory
bestPar = basePar;
bestPar.C = best_C;
bestPar = updateConnectivity(bestPar);
sim_validation = sim;
sim_validation.T_trans = 15;
sim_validation.final_timeout = 30;
[ok_fit, X_fit, EEG_fit] = simulate_jansen_for_fit( bestPar,best_p,sim_validation,t_exp);
if ~ok_fit
    error('Best-fit Jansen trajectory could not be reconstructed.');
end
raw_dynamics = trajectoryDiagnostics(EEG_fit,fs,signal_cfg);
if require_sustained_oscillation && ~raw_dynamics.is_sustained
    warning(['The best fit does not remain a sustained raw oscillation ', ...
             'after the longer transient. Interpret it as a transient, ', ...
             'not as a stable limit cycle.']);
end
EEG_fit_filtered = comparison_bandpass(EEG_fit,fs);
y_fit = normalize01(EEG_fit_filtered);
fit_features = build_signal_features( y_fit,fs,signal_cfg.psd_band);
phi_data = data_features.phase;
phi_fit = fit_features.phase;
phase_diff_unwrapped = n_lock*phi_data - m_lock*phi_fit;
phase_diff_wrapped = wrap_pi(phase_diff_unwrapped);
Delta0 = angle(mean(exp(1i*phase_diff_wrapped)));
R = abs(mean(exp(1i*phase_diff_wrapped)));
dphi_corrected = wrap_pi(phase_diff_wrapped-Delta0);

%% ========================================================================
%  9. Diagnostic figures
%  ========================================================================
screenSize = get(0,'ScreenSize');
margin = 40;
figPos = [ ...
    screenSize(1)+margin, ...
    screenSize(2)+margin, ...
    max(900,screenSize(3)-2*margin), ...
    max(700,screenSize(4)-2*margin)];

figure('Position',figPos,'Color','w');

% A. Waveform fit
subplot(3,3,[1 2]);
plot(t_exp,y_data,'r','LineWidth',1.2); hold on;
plot(t_exp,y_fit,'b','LineWidth',1.1);
legend('Target data (MEG)','Jansen fit','Location','best');
xlabel('Time (s)');
ylabel('Normalized amplitude');
title(sprintf('Waveform Fit: J_{new} = %.6f',best_cost));
grid on;

% B. Corrected wrapped phase difference
subplot(3,3,[4 5]);
plot(t_exp,dphi_corrected,'m','LineWidth',1.0);
xlabel('Time (s)');
ylabel('\Delta\phi-\Delta_0');
ylim([-pi pi]);
title({ ...
    'Corrected Wrapped Phase Difference', ...
    sprintf('R_{%d:%d}=%.4f, \\Delta_0=%.3f rad', ...
    n_lock,m_lock,R,Delta0)});
grid on;

% C. Jansen state-space projection
% Mathematical coordinates: (y0, y1, y2), with EEG=y1-y2.
subplot(3,3,3);
plot3(X_fit(:,1),X_fit(:,2),X_fit(:,3),'LineWidth',0.8);
xlabel('y_0'); ylabel('y_1'); zlabel('y_2');
title('Best Jansen Trajectory');
view(3); grid on;

% D. Conditional cost versus paper input P at fixed j
subplot(3,3,7);
plot(P_grid,scan_cost,'k-','LineWidth',1.5); hold on;
yline(threshold_line,'r--','LineWidth',1.2);
xline(best_P,'b:','LineWidth',1.2);
xlabel('Paper input P');
ylabel(sprintf('J(P; j=%.3f)',best_j));
title('Conditional P Cost (not continuation)');
legend('conditional cost','5% above minimum','best P','Location','best');
grid on;

% E. Unfiltered model frequency gives the dynamical frequency.  The
% filtered frequency is displayed only to expose the effect of the
% 3.6--4.4 Hz comparison filter.
subplot(3,3,8);
plot(P_grid,f_raw_profile,'b-','LineWidth',1.5); hold on;
plot(P_grid,f_filtered_profile,'Color',[0.6 0.6 0.6], ...
    'LineStyle','--','LineWidth',1.1);
yline(f_data,'r--','LineWidth',1.2);
plot(best_P,best_info.f_dom,'r*','MarkerSize',10,'LineWidth',1.5);
xlabel('Paper input P'); ylabel('Dominant frequency (Hz)');
title('Raw Frequency vs. P');
legend('raw model','filtered model','MEG target','best point', ...
    'Location','best');
grid on;

% F. Phase-locking index along the fixed-j P scan
subplot(3,3,9);
plot(P_grid,R_profile,'g-','LineWidth',1.5); hold on;
plot(best_P,best_info.R,'r*','MarkerSize',10,'LineWidth',1.5);
xlabel('Paper input P'); ylabel('R_{1:1}(P)');
ylim([0 1]);
title('Phase Locking vs. P');
grid on;

sgtitle({ ...
    'Jansen-Rit MEG Fit in Touboul Paper Coordinates', ...
    sprintf(['j=%.4f, P=%.4f (C=%.3f, p=%.3f), ', ...
             'R=%.4f, f_{raw}=%.3f Hz'], ...
    best_j,best_P,best_C,best_p,R,raw_dynamics.f_dom)}, ...
    'FontWeight','bold');


%% Local functions
function par = defaultJansenParameters
par.A = 3.25;        % mV
par.B = 22.0;        % mV
par.a = 100.0;       % s^(-1)
par.b = 50.0;        % s^(-1)
par.v0 = 6.0;        % mV
par.nuMax = 5.0;     % s^(-1)
par.r = 0.56;        % mV^(-1)
par.C = 135.0;
par = updateConnectivity(par);
end

function par = updateConnectivity(par)
par.C1 = 1.00*par.C;
par.C2 = 0.80*par.C;
par.C3 = 0.25*par.C;
par.C4 = 0.25*par.C;
end

function dX = jansenRHS(~,X,par,p)
y0 = X(1); y1 = X(2); y2 = X(3);
y3 = X(4); y4 = X(5); y5 = X(6);
S0 = sigmoidFunction(y1-y2,par);
S1 = sigmoidFunction(par.C1*y0,par);
S2 = sigmoidFunction(par.C3*y0,par);
dX = zeros(6,1);
dX(1) = y3;
dX(2) = y4;
dX(3) = y5;
dX(4) = par.A*par.a*S0 - 2*par.a*y3 - par.a^2*y0;
dX(5) = par.A*par.a*(p + par.C2*S1) - 2*par.a*y4 - par.a^2*y1;
dX(6) = par.B*par.b*par.C4*S2 - 2*par.b*y5 - par.b^2*y2;
end

function S = sigmoidFunction(v,par)
S = par.nuMax ./ (1 + exp(par.r*(par.v0-v)));
end

function [r,info] = jansen_residual_weighted( theta,basePar,sim,t_exp,y_data,data_features,fs, ...
    f_target,f_scale,weights,n_lock,m_lock)
theta = theta(:)';
if numel(theta) ~= 2 || any(~isfinite(theta))
    [r,info] = penaltyResidual(data_features.residual_length);
    return;
end
j = theta(1);
P = theta(2);
[C,p,G,d] = paperToDimensional(j,P,basePar);
par = basePar;
par.C = C;
par = updateConnectivity(par);
[ok,~,EEG_fit] = simulate_jansen_for_fit(par,p,sim,t_exp);
if ~ok || length(EEG_fit) ~= length(y_data) || any(~isfinite(EEG_fit))
    [r,info] = penaltyResidual(data_features.residual_length);
    info.j = j; info.P = P; info.C = C; info.p = p;
    info.G = G; info.d = d;
    return;
end
raw_diag = trajectoryDiagnostics( EEG_fit,fs,data_features.signal_cfg);
if data_features.signal_cfg.require_sustained_oscillation && ~raw_diag.is_sustained
    [r,info] = penaltyResidual(data_features.residual_length);
    info.j = j; info.P = P; info.C = C; info.p = p;
    info.G = G; info.d = d;
    info.f_dom = raw_diag.f_dom;
    info.raw_std = raw_diag.raw_std;
    info.sustain_ratio = raw_diag.sustain_ratio;
    info.trajectory_class = raw_diag.classification;
    info.paper_X_min = basePar.r*raw_diag.raw_min;
    info.paper_X_max = basePar.r*raw_diag.raw_max;
    info.paper_X_std = basePar.r*raw_diag.raw_std;
    return;
end
EEG_fit_filtered = comparison_bandpass(EEG_fit,fs);
y_fit = normalize01(EEG_fit_filtered);
fit_features = build_signal_features( y_fit,fs,data_features.psd_band);
if length(fit_features.psd) ~= length(data_features.psd) || any(~isfinite(fit_features.psd)) || ...
        any(~isfinite(fit_features.envelope))
    [r,info] = penaltyResidual(data_features.residual_length);
    return;
end
r_waveform = sqrt(weights.waveform/length(y_data))*(y_data-y_fit);
E_y = mean((y_data-y_fit).^2);
env_difference = data_features.envelope-fit_features.envelope;
env_denominator = max(sum(data_features.envelope.^2),eps);
r_envelope = sqrt(weights.envelope/env_denominator)*env_difference;
E_env = sum(env_difference.^2)/env_denominator;
rho_env = safe_correlation( data_features.envelope,fit_features.envelope);
sqrt_psd_data = sqrt(max(data_features.psd,0));
sqrt_psd_fit = sqrt(max(fit_features.psd,0));
psd_difference = sqrt_psd_data-sqrt_psd_fit;
r_psd = sqrt(weights.psd)*psd_difference;
E_psd = sum(psd_difference.^2);
f_dom_raw = raw_diag.f_dom;
f_dom_filtered = get_dominant_freq( y_fit,fs,data_features.signal_cfg.raw_frequency_band);
r_freq = sqrt(weights.frequency)*(f_dom_raw-f_target)/f_scale;
E_f = ((f_dom_raw-f_target)/f_scale)^2;
phi_data = data_features.phase;
phi_fit = fit_features.phase;
phase_diff = n_lock*phi_data - m_lock*phi_fit;
R = abs(mean(exp(1i*wrap_pi(phase_diff))));
R = min(max(real(R),0),1);
r_phase = sqrt(weights.phase*max(0,1-R));
E_theta = 1-R;
r = [r_waveform; r_envelope; r_psd; r_freq; r_phase];
info.R = R;
info.is_valid_fit = true;
info.f_dom = f_dom_raw;
info.f_dom_filtered = f_dom_filtered;
info.raw_std = raw_diag.raw_std;
info.sustain_ratio = raw_diag.sustain_ratio;
info.trajectory_class = raw_diag.classification;
info.paper_X_min = basePar.r*raw_diag.raw_min;
info.paper_X_max = basePar.r*raw_diag.raw_max;
info.paper_X_std = basePar.r*raw_diag.raw_std;
info.j = j;
info.P = P;
info.C = C;
info.p = p;
info.G = G;
info.d = d;
info.E_y = E_y;
info.E_env = E_env;
info.E_psd = E_psd;
info.E_f = E_f;
info.E_theta = E_theta;
info.rho_env = rho_env;
info.weighted_waveform = weights.waveform*E_y;
info.weighted_envelope = weights.envelope*E_env;
info.weighted_psd = weights.psd*E_psd;
info.weighted_frequency = weights.frequency*E_f;
info.weighted_phase = weights.phase*E_theta;
info.legacy_J = 0.625*E_y + 0.125*E_f + 0.250*E_theta;
end

function [ok,X_fit,EEG_fit] = simulate_jansen_for_fit(par,p,sim,t_exp)
ok = false;
X_fit = [];
EEG_fit = [];
t_start = tic;
stop_fcn = @(t,y,flag) stop_by_timeout( t,y,flag,t_start,sim.final_timeout);
ode_opts = odeset( 'OutputFcn',stop_fcn, 'RelTol',sim.RelTol, 'AbsTol',sim.AbsTol, ...
    'MaxStep',sim.MaxStep);
try
    [~,Xtrans] = ode15s( @(t,X) jansenRHS(t,X,par,p), [0 sim.T_trans],sim.X0,ode_opts);
    if isempty(Xtrans) || toc(t_start) > sim.final_timeout
        return;
    end
    X0_steady = Xtrans(end,:)';
    [~,X_fit] = ode15s( @(t,X) jansenRHS(t,X,par,p), t_exp,X0_steady,ode_opts);
    if size(X_fit,1) ~= length(t_exp)
        X_fit = [];
        return;
    end
    EEG_fit = X_fit(:,2)-X_fit(:,3);
    if any(~isfinite(EEG_fit))
        X_fit = [];
        EEG_fit = [];
        return;
    end
    ok = true;
catch
    X_fit = [];
    EEG_fit = [];
end
end

function [r,info] = penaltyResidual(residual_length)
r = ones(residual_length,1)*99;
info.R = 0;
info.is_valid_fit = false;
info.f_dom = 0;
info.f_dom_filtered = 0;
info.raw_std = 0;
info.sustain_ratio = 0;
info.trajectory_class = 'invalid or non-oscillatory';
info.paper_X_min = NaN;
info.paper_X_max = NaN;
info.paper_X_std = NaN;
info.j = NaN;
info.P = NaN;
info.C = NaN;
info.p = NaN;
info.G = NaN;
info.d = NaN;
info.E_y = 99;
info.E_env = 99;
info.E_psd = 99;
info.E_f = 99;
info.E_theta = 99;
info.rho_env = 0;
info.weighted_waveform = 99;
info.weighted_envelope = 99;
info.weighted_psd = 99;
info.weighted_frequency = 99;
info.weighted_phase = 99;
info.legacy_J = Inf;
end

function theta_full = apply_fixed_params(theta_var,fixed_params)
theta_full = fixed_params;
var_idx = isnan(fixed_params);
if numel(theta_var) ~= sum(var_idx)
    error(['Number of supplied free parameters (%d) does not match ', ...
           'the number of NaN entries in fixed_params (%d).'], numel(theta_var),sum(var_idx));
end
theta_full(var_idx) = reshape(theta_var,1,[]);
end

function [C,p,G,d] = paperToDimensional(j,P,par)
j_scale = par.r*par.A*par.nuMax/par.a;
P_scale = par.r*par.A/par.a;
if j_scale <= 0 || P_scale <= 0
    error('The dimensional-to-paper scaling factors must be positive.');
end
C = j/j_scale;
p = P/P_scale;
G = par.B/par.A;
d = par.b/par.a;
end

function diag = trajectoryDiagnostics(sig,fs,cfg)
sig = sig(:);
n = length(sig);
if n < 4 || any(~isfinite(sig))
    diag.raw_std = 0;
    diag.raw_min = NaN;
    diag.raw_max = NaN;
    diag.sustain_ratio = 0;
    diag.is_sustained = false;
    diag.classification = 'invalid or too short';
    diag.f_dom = 0;
    return;
end
half_index = max(2,floor(n/2));
std_first = std(sig(1:half_index));
steady_sig = sig(half_index+1:end);
std_second = std(steady_sig);
diag.raw_std = std_second;
diag.raw_min = min(steady_sig);
diag.raw_max = max(steady_sig);
diag.sustain_ratio = std_second/max(std_first,sqrt(eps));
has_amplitude = isfinite(diag.raw_std) && diag.raw_std >= cfg.min_raw_std;
has_stationary_amplitude = isfinite(diag.sustain_ratio) && ...
    diag.sustain_ratio >= cfg.min_sustain_ratio && diag.sustain_ratio <= cfg.max_sustain_ratio;
diag.is_sustained = has_amplitude && has_stationary_amplitude;
if ~has_amplitude
    diag.classification = 'equilibrium or numerically flat';
    diag.f_dom = 0;
elseif diag.sustain_ratio < cfg.min_sustain_ratio
    diag.classification = 'decaying transient';
    diag.f_dom = get_dominant_freq(steady_sig,fs,cfg.raw_frequency_band);
elseif diag.sustain_ratio > cfg.max_sustain_ratio
    diag.classification = 'growing/nonstationary transient';
    diag.f_dom = get_dominant_freq(steady_sig,fs,cfg.raw_frequency_band);
else
    diag.classification = 'sustained oscillation candidate';
    diag.f_dom = get_dominant_freq(steady_sig,fs,cfg.raw_frequency_band);
end
end

function y = normalize01(x)
x = x(:);
xmin = min(x);
xmax = max(x);
if isfinite(xmin) && isfinite(xmax) && xmax-xmin > 1e-12
    y = (x-xmin)/(xmax-xmin);
else
    y = zeros(size(x));
end
end

function features = build_signal_features(sig,fs,psd_band)
sig = sig(:);
features.phase = get_phase(sig);
features.envelope = relative_hilbert_envelope(sig);
[features.psd,features.psd_frequency] = normalized_psd_shape(sig,fs,psd_band);
features.psd_band = psd_band;
features.residual_length = 2*length(sig)+length(features.psd)+2;
end

function envelope_relative = relative_hilbert_envelope(sig)
sig = sig(:)-mean(sig(:));
if exist('hilbert','file') == 2
    analytic_sig = hilbert(sig);
else
    analytic_sig = analyticSignalFFT(sig);
end
envelope_value = abs(analytic_sig);
mean_envelope = mean(envelope_value);
if isfinite(mean_envelope) && mean_envelope > sqrt(eps)
    envelope_relative = envelope_value/mean_envelope;
else
    envelope_relative = zeros(size(envelope_value));
end
end

function [P_normalized,f_selected] = normalized_psd_shape(sig,fs,band)
sig = sig(:)-mean(sig(:));
n = length(sig);
if n < 4
    error('Signal is too short for PSD estimation.');
end
k = (0:n-1)';
hann_window = 0.5-0.5*cos(2*pi*k/(n-1));
Y = fft(sig.*hann_window);
P = abs(Y(1:floor(n/2)+1)).^2;
if mod(n,2) == 0
    if length(P) > 2
        P(2:end-1) = 2*P(2:end-1);
    end
else
    if length(P) > 1
        P(2:end) = 2*P(2:end);
    end
end
f = fs*(0:floor(n/2))'/n;
selected = f >= band(1) & f <= band(2);
if ~any(selected)
    error('PSD band [%.3f, %.3f] Hz contains no FFT bins.',band(1),band(2));
end
P_normalized = P(selected);
f_selected = f(selected);
power_sum = sum(P_normalized);
if isfinite(power_sum) && power_sum > eps
    P_normalized = P_normalized/power_sum;
else
    P_normalized = ones(size(P_normalized))/length(P_normalized);
end
end

function rho = safe_correlation(x,y)
x = x(:)-mean(x(:));
y = y(:)-mean(y(:));
denominator = norm(x)*norm(y);
if denominator > sqrt(eps)
    rho = real((x'*y)/denominator);
    rho = min(max(rho,-1),1);
else
    rho = 0;
end
end

function y = comparison_bandpass(x,fs)
x = x(:);
if exist('bandpass','file') == 2
    try
        y = bandpass(x,[3.6 4.4],fs);
        y = y(:);
        return;
    catch
    end
end
Nf = length(x);
freq = (0:Nf-1)'*fs/Nf;
abs_freq = min(freq,fs-freq);
H = zeros(Nf,1);
H(abs_freq >= 3.6 & abs_freq <= 4.4) = 1;
idx_low = abs_freq > 3.4 & abs_freq < 3.6;
H(idx_low) = 0.5*(1-cos(pi*(abs_freq(idx_low)-3.4)/(3.6-3.4)));
idx_high = abs_freq > 4.4 & abs_freq < 4.6;
H(idx_high) = 0.5*(1+cos(pi*(abs_freq(idx_high)-4.4)/(4.6-4.4)));
y = real(ifft(fft(x).*H));
end

function f_dom = get_dominant_freq(sig,fs,band)
if nargin < 3 || isempty(band)
    band = [0 fs/2];
end
sig = sig(:)-mean(sig);
n = length(sig);
if n < 4 || all(abs(sig) < 1e-12)
    f_dom = 0;
    return;
end
Y = fft(sig);
P2 = abs(Y/n);
P1 = P2(1:floor(n/2)+1);
if length(P1) > 2
    P1(2:end-1) = 2*P1(2:end-1);
end
freq = fs*(0:floor(n/2))'/n;
P1(1) = 0;
selected = freq >= max(0,band(1)) & freq <= min(fs/2,band(2));
if ~any(selected)
    f_dom = 0;
    return;
end
P1(~selected) = 0;
[~,max_idx] = max(P1);
f_dom = freq(max_idx);
end

function phi = get_phase(sig)
sig = sig(:)-mean(sig);
if exist('hilbert','file') == 2
    analytic_sig = hilbert(sig);
else
    analytic_sig = analyticSignalFFT(sig);
end
phi = angle(analytic_sig);
end

function z = analyticSignalFFT(x)
N = length(x);
X = fft(x);
h = zeros(N,1);
if mod(N,2) == 0
    h(1) = 1;
    h(N/2+1) = 1;
    h(2:N/2) = 2;
else
    h(1) = 1;
    h(2:(N+1)/2) = 2;
end
z = ifft(X.*h);
end

function y = wrap_pi(x)
y = atan2(sin(x),cos(x));
end

function status = stop_by_timeout(~,~,flag,t_start,timeout)
status = 0;
if strcmp(flag,'init')
    return;
end
if toc(t_start) > timeout
    status = 1;
end
end

function update_waitbar(h, total_points)
    % 使用 persistent 變數記住目前的進度
    persistent count;
    if isempty(count)
        count = 0;
    end
    count = count + 1;
    
    % 強制更新 UI 進度條與文字
    if isvalid(h)
        progress_msg = sprintf('平行運算進度: [%d/%d] 已完成', count, total_points);
        waitbar(count / total_points, h, progress_msg);
        drawnow; % 強制立即刷新畫面
    end
end