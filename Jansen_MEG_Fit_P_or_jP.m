%% Jansen-Rit Neural Mass Model: MEG fitting in Touboul paper coordinates
% Modified from mainRosslerModelSinFunction9ssV3.m
% Core dynamics replaced by the six-dimensional Jansen neural mass model
% used in Jansen_Scenario_Study.m.
%
% Fitted model output:
%       EEG(t) = y1(t) - y2(t)
%
% Paper-compatible fitting setup used in this version:
%       The dimensional parameters are fixed at the standard values used by
%       Touboul et al. (2011), including A=3.25 mV and B=22 mV.
%       Two fitting modes are available in the paper's dimensionless
%       coordinates (never in the mixed dimensional coordinates [A,C,p]):
%         'P_only'   : keep j fixed and estimate P only;
%         'joint_jP' : estimate j and P jointly.
%
% Exact conversion between paper and MATLAB coordinates:
%       j = (r*A*nuMax/a)*C,       P = (r*A/a)*p,
%       C = j/(r*A*nuMax/a),       p = P/(r*A/a).
%       X_paper = r*(y1-y2).
% With the standard parameters below:
%       j = 0.091*C, P = 0.0182*p.
%
% Fixed standard Jansen parameters:
%       A = 3.25 mV, B = 22 mV,
%       a = 100 s^(-1), b = 50 s^(-1),
%       v0 = 6 mV, nuMax = 5 s^(-1), r = 0.56 mV^(-1)
%
% Connectivity conditions:
%       C1 = C, C2 = 0.8 C, C3 = 0.25 C, C4 = 0.25 C.
%
% Enhanced optimization objective:
%       J = w_y E_y + w_env E_env + w_psd E_psd
%           + w_f E_f + w_theta (1-R_{n:m}).
% E_env compares the relative Hilbert amplitude envelopes.  E_psd compares
% normalized power-spectrum shapes.  Absolute MEG and model amplitudes are
% not assumed to have the same physical units.

clear; clc; close all;
rng(1,'twister');              % reproducible multi-start experiment

%% ========================================================================
%  0. USER SETTINGS: Touboul paper coordinates
%  ========================================================================
% Select one optimization mode:
%   'P_only'   : fix j at fixed_j and optimize P only (recommended first).
%   'joint_jP' : optimize the pair theta=[j,P].
fit_mode = 'joint_jP';

% Touboul Figure 5 explicitly varies P at j=14, so this is the default
% fixed-j slice for P_only mode.  It is ignored in joint_jP mode.
fixed_j = 14.00;

% Recommended range for the approximately 4-Hz MEG target:
%   j in (12.93,14] is paper region H (above the cusp of limit cycles).
%   P in [2,6] covers the delta/epileptiform-to-theta transition and,
%   depending on j, the approach toward alpha activity.
% Paper zone boundaries (Table 4):
%   A: j<5.38; B: 5.38--10.05; C: 10.05--12.10;
%   D: 12.10--12.38; E: 12.38--12.48; F: 12.48--12.55;
%   G: 12.55--12.93; H: j>12.93.
% This fit deliberately searches H because the paper reports theta activity
% only there.  Change the bounds only when testing a different paper zone.
% At A=3.25 these correspond to
%   C in [142.20,153.85], p in [109.89,329.67] s^(-1).
paper_j_bounds = [12.40, 12.60];  % strict H-zone lower side (j_CLC=12.93)
paper_P_bounds = [ 2.60,  2.80];

% Touboul's explicit theta example is j=14, P=5, corresponding to
% C=153.846 and p=274.725 in the dimensional Jansen-Rit equations.
paper_seed = [12.50, 2.7];       % [j,P]

% Set this to false only if you deliberately want equilibrium candidates.
% It prevents band-pass filtering and normalization from magnifying tiny
% numerical residuals around a stable equilibrium into an apparent 4-Hz fit.
require_sustained_oscillation = true;
run_multistability_check = true;

%% ========================================================================
%  1. Load experimental MEG data
%  ========================================================================
fprintf('=== Step 1: Load and preprocess MEG/EMG data ===\n');

if ~exist('FingerLifting4Student102.mat', 'file')
    error(['Cannot find FingerLifting4Student102.mat. ', ...
           'Place the data file in the same folder as this script.']);
end

load('FingerLifting4Student102.mat');   % must contain pdata

channel_id = 13;
para.channel_ide = channel_id;

data_range = 12000:28000;
y_data_raw = pdata.PdataF(data_range, channel_id);
y_data_raw = y_data_raw(:);

fs = pdata.sample_fif;
N = length(y_data_raw);
t_exp = (0:N-1)'/fs;

% Keep y_data_raw for the observation-scale diagnostic.  The optimization
% waveform remains normalized, so the new cost is comparable across model
% evaluations even though MEG and Jansen outputs have different units.
% Mathematical normalization condition:
% y_norm = (y-y_min)/(y_max-y_min), if y_max-y_min > 1e-12.
y_data = normalize01(y_data_raw);

f_data = get_dominant_freq(y_data, fs);
fprintf('Detected data dominant frequency = %.3f Hz\n', f_data);

% Same target used in the original Rössler fitting script.
f_target = f_data;
f_scale = max(abs(f_target), 1);
fprintf('Optimization target frequency = %.3f Hz\n', f_target);

% Signal features used by every model evaluation.  The PSD comparison band
% slightly includes the 3.6--4.4 Hz filter transition region.
signal_cfg.psd_band = [3.4 4.6];
signal_cfg.raw_frequency_band = [0.5 15];
signal_cfg.min_raw_std = 1e-4;       % model-output units (mV proxy)
signal_cfg.min_sustain_ratio = 0.50; % std(second half)/std(first half)
signal_cfg.max_sustain_ratio = 2.00;
signal_cfg.require_sustained_oscillation = require_sustained_oscillation;
data_features = build_signal_features( ...
    y_data,fs,signal_cfg.psd_band);
data_features.signal_cfg = signal_cfg;

%% ========================================================================
%  2. Multi-objective weights and n:m phase locking
%  ========================================================================
weights.waveform = 0.35;
weights.envelope = 0.20;
weights.psd      = 0.15;
weights.frequency = 0.10;
weights.phase    = 0.20;

w_sum = weights.waveform + weights.envelope + weights.psd + ...
        weights.frequency + weights.phase;
weights.waveform = weights.waveform/w_sum;
weights.envelope = weights.envelope/w_sum;
weights.psd = weights.psd/w_sum;
weights.frequency = weights.frequency/w_sum;
weights.phase = weights.phase/w_sum;

fprintf(['Weights: waveform=%.2f, envelope=%.2f, PSD=%.2f, ', ...
         'frequency=%.2f, phase=%.2f\n'], ...
    weights.waveform,weights.envelope,weights.psd, ...
    weights.frequency,weights.phase);

% Phase-locking condition:
% R_{n:m} = | < exp(i[n phi_data - m phi_model]) > |, 0 <= R <= 1.
n_lock = 1;
m_lock = 1;
fprintf('Phase-locking analysis: %d:%d\n', n_lock, m_lock);

%% ========================================================================
%  3. Paper parameters and optimization bounds
%  ========================================================================
% Full paper-coordinate order is always theta=[j,P].  NaN marks a free
% coordinate; a numeric entry marks a fixed coordinate.
lb = [paper_j_bounds(1), paper_P_bounds(1)];   % full [j,P] bounds
ub = [paper_j_bounds(2), paper_P_bounds(2)];

if ~isscalar(fixed_j) || ~isfinite(fixed_j) || fixed_j <= 0
    error('fixed_j must be a finite positive scalar.');
end

switch lower(fit_mode)
    case 'p_only'
        fixed_paper_params = [fixed_j, NaN];
        theta_seed_full = [fixed_j, paper_seed(2)];
        fitted_coordinate_text = 'P only (j fixed)';
        num_starts = 30;
    case 'joint_jp'
        fixed_paper_params = [NaN, NaN];
        theta_seed_full = paper_seed;
        fitted_coordinate_text = '[j,P] jointly';
        num_starts = 20;
    otherwise
        error('fit_mode must be either ''P_only'' or ''joint_jP''.');
end

output_tag = lower(fit_mode);

var_idx = isnan(fixed_paper_params);
lb_var = lb(var_idx);
ub_var = ub(var_idx);
theta_seed_var = theta_seed_full(var_idx);

if any(theta_seed_var < lb_var) || any(theta_seed_var > ub_var)
    error('The free-coordinate seed must lie inside its paper bounds.');
end

% Standard dimensional parameters corresponding to G=6.76923 and d=0.5.
basePar = defaultJansenParameters();
[seed_C,seed_p,seed_G,seed_d] = paperToDimensional( ...
    theta_seed_full(1),theta_seed_full(2),basePar);
fprintf(['Paper-compatible parameterization: G=%.5f, d=%.3f, ', ...
         'A=%.3f mV\n'],seed_G,seed_d,basePar.A);
fprintf('Paper sigmoid parameter: log(k0)=r*v0=%.4f\n', ...
    basePar.r*basePar.v0);
fprintf('Optimization mode: %s\n',fit_mode);
if strcmpi(fit_mode,'P_only')
    fprintf('Fixed paper coordinate: j=%.4f (C=%.4f)\n', ...
        fixed_j,seed_C);
end
fprintf(['Initial paper point: j=%.4f, P=%.4f ', ...
         '<=> C=%.4f, p=%.4f s^(-1)\n'], ...
    theta_seed_full(1),theta_seed_full(2),seed_C,seed_p);

% Integration settings.
% Jansen's standard synaptic rates a=100 and b=50 s^(-1) are much faster
% than the MEG observation time scale, therefore ode15s is retained.
sim.T_trans = 5;         % fitting transient, seconds
sim.final_timeout = 8;   % timeout for one model evaluation, seconds
sim.RelTol = 1e-6;
sim.AbsTol = 1e-8;
sim.MaxStep = min(1e-2, 1/fs);

% Upper-state / oscillatory seed. This follows the same spirit as the
% 'upperPerturbed' initialization in Jansen_Scenario_Study.m and avoids
% forcing every trial into the lower equilibrium basin.
% State order: X=[y0;y1;y2;y3;y4;y5].
sim.X0 = [0.05; 8; 1; 0; 0; 0];

%% ========================================================================
%  4. Multi-start optimization
%  ========================================================================
fprintf('\n=== Step 2: Multi-start optimization of Jansen model ===\n');
fprintf('Fitted paper coordinates: %s\n',fitted_coordinate_text);

opts_mstart = optimoptions('lsqnonlin', ...
    'Display', 'off', ...
    'MaxIterations', 80, ...
    'MaxFunctionEvaluations', 800);

para.num_starts = num_starts;
para.fit_mode = fit_mode;

best_cost = Inf;
best_theta = [];
best_info = [];

for k = 1:num_starts

    if k == 1
        theta0_var = theta_seed_var;
    else
        theta0_var = lb_var + ...
            (ub_var-lb_var).*rand(1,sum(var_idx));
    end

    obj_fun = @(theta_var) jansen_residual_weighted( ...
        apply_fixed_params(theta_var,fixed_paper_params),basePar,sim, ...
        t_exp, y_data, data_features, fs, f_target, f_scale, ...
        weights, n_lock, m_lock);

    try
        [theta_opt_var, resnorm] = lsqnonlin( ...
            obj_fun,theta0_var,lb_var,ub_var,opts_mstart);

        theta_opt = apply_fixed_params( ...
            theta_opt_var,fixed_paper_params);

        [~, info] = jansen_residual_weighted( ...
            theta_opt, basePar, sim, ...
            t_exp, y_data, data_features, fs, f_target, f_scale, ...
            weights, n_lock, m_lock);

        if info.is_valid_fit && isfinite(resnorm) && resnorm < best_cost
            best_cost = resnorm;
            best_theta = theta_opt;
            best_info = info;
        end

        fprintf(['  [start %2d/%2d] Cost = %.6f | best = %.6f ', ...
                 '| j=%.4f P=%.4f | C=%.3f p=%.3f\n'], ...
            k, num_starts, resnorm, best_cost, ...
            theta_opt(1),theta_opt(2),info.C,info.p);

    catch ME
        fprintf('  [start %2d/%2d] optimization failed: %s\n', ...
            k, num_starts, ME.message);
    end
end

if isempty(best_theta)
    error(['Multi-start optimization did not find a valid solution. ', ...
           'Check parameter bounds, initial condition, or integration settings.']);
end

best_j = best_theta(1);
best_P = best_theta(2);
best_A = basePar.A;
[best_C,best_p,best_G,best_d] = paperToDimensional( ...
    best_j,best_P,basePar);

fprintf('\n>>> Best-fit Touboul/Jansen-Rit parameters:\n');
fprintf('    j = %.6f (dimensionless)\n',best_j);
fprintf('    P = %.6f (dimensionless)\n',best_P);
fprintf('    A = %.6f mV\n', best_A);
fprintf('    C = %.6f\n', best_C);
fprintf('    p = %.6f s^(-1)\n', best_p);
fprintf('    G = B/A = %.6f, d = b/a = %.6f\n',best_G,best_d);
fprintf('    Paper j-region: %s\n',paperRegionFromJ(best_j));
fprintf(['>>> Minimum Cost J = %.6f | R = %.4f | ', ...
         'raw model frequency = %.3f Hz\n'], ...
    best_cost, best_info.R, best_info.f_dom);
fprintf(['>>> Raw dynamics: std=%.6g, sustain ratio=%.4f, ', ...
         'classification=%s\n'], ...
    best_info.raw_std,best_info.sustain_ratio,best_info.trajectory_class);
fprintf('>>> Paper-X envelope from this trajectory: [%.5f, %.5f]\n', ...
    best_info.paper_X_min,best_info.paper_X_max);
fprintf(['>>> Interpretation note: a paper-region label describes the ', ...
         'mathematical model, not a clinical diagnosis of this MEG record.\n']);
fprintf(['>>> Cost components (unweighted): E_y=%.5f, E_env=%.5f, ', ...
         'E_psd=%.5f, E_f=%.5f, 1-R=%.5f\n'], ...
    best_info.E_y,best_info.E_env,best_info.E_psd, ...
    best_info.E_f,best_info.E_theta);
fprintf(['>>> Weighted J contributions: waveform=%.5f, envelope=%.5f, ', ...
         'PSD=%.5f, frequency=%.5f, phase=%.5f\n'], ...
    best_info.weighted_waveform,best_info.weighted_envelope, ...
    best_info.weighted_psd,best_info.weighted_frequency, ...
    best_info.weighted_phase);
fprintf(['>>> Legacy J (old 0.625/0.125/0.250 objective, ', ...
         'for comparison only) = %.6f\n'],best_info.legacy_J);
fprintf('>>> Envelope correlation rho_env = %.4f\n',best_info.rho_env);

%% ========================================================================
%  5. Conditional P scan at the fitted j
%  ========================================================================
% This follows the orientation of Touboul et al. Figure 5: P varies while
% j is fixed.  It is a fitting-cost scan, NOT a numerical continuation or
% a substitute for a true bifurcation diagram.
fprintf(['\n=== Step 3: Conditional P scan ', ...
         '(paper j=%.4f fixed) ===\n'],best_j);

N_grid = 81;
P_grid = linspace(lb(2),ub(2),N_grid);

scan_cost = nan(N_grid,1);
valid_fit_profile = false(N_grid,1);
f_raw_profile = nan(N_grid,1);
f_filtered_profile = nan(N_grid,1);
R_profile = nan(N_grid,1);
Eenv_profile = nan(N_grid,1);
Epsd_profile = nan(N_grid,1);
rhoenv_profile = nan(N_grid,1);
rawstd_profile = nan(N_grid,1);
sustain_profile = nan(N_grid,1);
paper_X_min_profile = nan(N_grid,1);
paper_X_max_profile = nan(N_grid,1);
paper_X_std_profile = nan(N_grid,1);
p_profile = P_grid(:)/(basePar.r*basePar.A/basePar.a);
trajectory_profile = repmat({''},N_grid,1);

for i = 1:N_grid
    theta_scan = [best_j,P_grid(i)];

    try
        [residual_scan,info_scan] = jansen_residual_weighted( ...
            theta_scan,basePar,sim, ...
            t_exp,y_data,data_features,fs,f_target,f_scale, ...
            weights,n_lock,m_lock);

        valid_fit_profile(i) = info_scan.is_valid_fit;
        % lsqnonlin reports resnorm=sum(residual.^2); use the same scale.
        % Invalid/equilibrium candidates remain NaN so their finite penalty
        % residual cannot be mistaken for part of the accepted profile.
        if info_scan.is_valid_fit
            scan_cost(i) = sum(residual_scan.^2);
        end
        f_raw_profile(i) = info_scan.f_dom;
        f_filtered_profile(i) = info_scan.f_dom_filtered;
        R_profile(i) = info_scan.R;
        Eenv_profile(i) = info_scan.E_env;
        Epsd_profile(i) = info_scan.E_psd;
        rhoenv_profile(i) = info_scan.rho_env;
        rawstd_profile(i) = info_scan.raw_std;
        sustain_profile(i) = info_scan.sustain_ratio;
        paper_X_min_profile(i) = info_scan.paper_X_min;
        paper_X_max_profile(i) = info_scan.paper_X_max;
        paper_X_std_profile(i) = info_scan.paper_X_std;
        trajectory_profile{i} = info_scan.trajectory_class;

        fprintf(['  [grid %2d/%2d] P=%.4f (p=%.2f) | J=%.6f ', ...
                 '| f_raw=%.3f Hz | R=%.4f | %s\n'], ...
            i,N_grid,P_grid(i),p_profile(i),scan_cost(i), ...
            f_raw_profile(i),R_profile(i),trajectory_profile{i});

    catch ME
        fprintf('  [grid %2d/%2d] P=%.4f failed: %s\n', ...
            i,N_grid,P_grid(i),ME.message);
    end
end

%% ========================================================================
%  6. Relative profile-cost tolerance
%  ========================================================================
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

% J is a custom multi-objective cost, not a calibrated negative
% log-likelihood.  Therefore use a transparent 5% relative tolerance rather
% than calling it a statistical 95% confidence threshold.
relative_tolerance = 0.05;
threshold_line = J_min*(1+relative_tolerance);

fprintf('\n=== Relative profile-cost tolerance ===\n');
fprintf('  Minimum profile cost = %.6f\n', J_min);
fprintf('  5%%-above-minimum threshold = %.6f\n', threshold_line);

accepted_profile = valid_scan & (scan_cost <= threshold_line);
if any(accepted_profile)
    P_profile_low = min(P_grid(accepted_profile));
    P_profile_high = max(P_grid(accepted_profile));
    p_profile_low = P_profile_low/(basePar.r*basePar.A/basePar.a);
    p_profile_high = P_profile_high/(basePar.r*basePar.A/basePar.a);
    fprintf('  Accepted paper-P interval = [%.5f, %.5f]\n', ...
        P_profile_low,P_profile_high);
    fprintf('  Equivalent dimensional-p interval = [%.3f, %.3f] s^(-1)\n', ...
        p_profile_low,p_profile_high);
    if accepted_profile(1) || accepted_profile(end)
        warning(['The 5%% tolerance interval touches a P-grid boundary. ', ...
                 'The reported interval is truncated; extend paper_P_bounds.']);
    end
else
    P_profile_low = NaN;
    P_profile_high = NaN;
end

if isfinite(grid_min_index) && ...
        (grid_min_index == 1 || grid_min_index == N_grid)
    warning(['The smallest conditional cost is at a P-grid boundary. ', ...
             'Do not call this an interior optimum before extending paper_P_bounds.']);
end

% Save both coordinate systems and raw-dynamics diagnostics.
j_column = repmat(best_j,N_grid,1);
C_column = repmat(best_C,N_grid,1);
profile_results = table(j_column,P_grid(:),C_column,p_profile, ...
    valid_fit_profile,scan_cost, ...
    f_raw_profile,f_filtered_profile,R_profile,Eenv_profile,Epsd_profile, ...
    rhoenv_profile,rawstd_profile,sustain_profile, ...
    paper_X_min_profile,paper_X_max_profile,paper_X_std_profile, ...
    trajectory_profile, ...
    'VariableNames',{'j','P','C','p','ValidFit','FitCost', ...
    'RawFrequency_Hz','FilteredFrequency_Hz','PhaseLocking_R', ...
    'EnvelopeError','PSDError', ...
    'EnvelopeCorrelation','RawStd','SustainRatio','PaperX_Min', ...
    'PaperX_Max','PaperX_Std','TrajectoryClass'});
profile_output_file = sprintf( ...
    'Jansen_P_scan_%s_paper_coordinates.csv',output_tag);
try
    writetable(profile_results,profile_output_file);
    fprintf('  Saved conditional-scan metrics to %s\n',profile_output_file);
catch ME
    warning('Could not write %s: %s',profile_output_file,ME.message);
end

%% ========================================================================
%  7. Reconstruct best-fit trajectory
%  ========================================================================
fprintf('\n=== Step 4: Reconstruct best-fit Jansen trajectory ===\n');

bestPar = basePar;
bestPar.C = best_C;
bestPar = updateConnectivity(bestPar);

% Use a longer transient for the final scientific diagnostic than during
% optimization.  This helps reject slowly decaying transients near a
% bifurcation boundary.
sim_validation = sim;
sim_validation.T_trans = 15;
sim_validation.final_timeout = 30;
[ok_fit, X_fit, EEG_fit] = simulate_jansen_for_fit( ...
    bestPar,best_p,sim_validation,t_exp);

if ~ok_fit
    error('Best-fit Jansen trajectory could not be reconstructed.');
end

raw_dynamics = trajectoryDiagnostics(EEG_fit,fs,signal_cfg);
X_fit_paper = basePar.r*EEG_fit;
fprintf(['Final long-transient raw dynamics: f=%.4f Hz, std=%.6g, ', ...
         'sustain ratio=%.4f, %s\n'], ...
    raw_dynamics.f_dom,raw_dynamics.raw_std, ...
    raw_dynamics.sustain_ratio,raw_dynamics.classification);
fprintf('Final paper-X envelope: [%.5f, %.5f]\n', ...
    basePar.r*raw_dynamics.raw_min,basePar.r*raw_dynamics.raw_max);
if require_sustained_oscillation && ~raw_dynamics.is_sustained
    warning(['The best fit does not remain a sustained raw oscillation ', ...
             'after the longer transient. Interpret it as a transient, ', ...
             'not as a stable limit cycle.']);
end

% Touboul's D--H regions can be multistable.  A single initial condition
% cannot reveal coexisting equilibria or cycles, so test three seeds at the
% fitted paper point.  This remains a numerical attractor check rather than
% a continuation-based proof.
if run_multistability_check
    fprintf('\n=== Step 5: Initial-condition / multistability check ===\n');
    ic_labels = {'upper_seed';'zero_seed';'middle_seed'};
    ic_matrix = [sim.X0,zeros(6,1),[0.05;4;2;0;0;0]];
    attractor_ok = false(3,1);
    attractor_frequency = nan(3,1);
    attractor_std = nan(3,1);
    attractor_X_std = nan(3,1);
    attractor_sustain = nan(3,1);
    attractor_class = repmat({''},3,1);

    for ic = 1:3
        if ic == 1
            ok_ic = true;
            EEG_ic = EEG_fit;
            diag_ic = raw_dynamics;
        else
            sim_ic = sim_validation;
            sim_ic.X0 = ic_matrix(:,ic);
            [ok_ic,~,EEG_ic] = simulate_jansen_for_fit( ...
                bestPar,best_p,sim_ic,t_exp);
            if ok_ic
                diag_ic = trajectoryDiagnostics(EEG_ic,fs,signal_cfg);
            end
        end

        attractor_ok(ic) = ok_ic;
        if ok_ic
            attractor_frequency(ic) = diag_ic.f_dom;
            attractor_std(ic) = diag_ic.raw_std;
            attractor_X_std(ic) = basePar.r*diag_ic.raw_std;
            attractor_sustain(ic) = diag_ic.sustain_ratio;
            attractor_class{ic} = diag_ic.classification;
            fprintf(['  %-11s | f=%.4f Hz | std=%.5g | ', ...
                     'sustain=%.3f | %s\n'], ...
                ic_labels{ic},diag_ic.f_dom,diag_ic.raw_std, ...
                diag_ic.sustain_ratio,diag_ic.classification);
        else
            attractor_class{ic} = 'integration failed';
            fprintf('  %-11s | integration failed\n',ic_labels{ic});
        end
    end

    attractor_results = table(ic_labels,attractor_ok,attractor_frequency, ...
        attractor_std,attractor_X_std,attractor_sustain,attractor_class, ...
        'VariableNames',{'InitialCondition','IntegrationOK', ...
        'RawFrequency_Hz','RawStd','PaperX_Std','SustainRatio', ...
        'TrajectoryClass'});
    attractor_output_file = sprintf( ...
        'Jansen_best_attractor_check_%s.csv',output_tag);
    try
        writetable(attractor_results,attractor_output_file);
        fprintf('  Saved initial-condition check to %s\n', ...
            attractor_output_file);
    catch ME
        warning('Could not write %s: %s',attractor_output_file,ME.message);
    end
end

% Compare like with like: pdata.PdataF is narrow-band filtered, therefore
% apply the same 3.6--4.4 Hz comparison band to the simulated EEG.
EEG_fit_filtered = comparison_bandpass(EEG_fit,fs);
y_fit = normalize01(EEG_fit_filtered);

% Relative amplitude-envelope and PSD diagnostics.  The envelope is divided
% by its own mean, so this measures modulation shape rather than an invalid
% direct comparison between MEG sensor units and model mV.
fit_features = build_signal_features( ...
    y_fit,fs,signal_cfg.psd_band);
env_data_relative = data_features.envelope;
env_fit_relative = fit_features.envelope;

% Observation-scale diagnostic y_MEG ~= offset + gain*y_model.  This does
% not enter the optimization; it only makes a raw-unit residual meaningful.
[EEG_fit_affine,obs_gain,obs_offset] = affine_align_signal( ...
    EEG_fit_filtered,y_data_raw);
affine_residual = y_data_raw-EEG_fit_affine;
affine_nrmse = norm(affine_residual)/ ...
    max(norm(y_data_raw-mean(y_data_raw)),sqrt(eps));

% Wider PSD view for the diagnostic figure.  The objective itself uses the
% narrower signal_cfg.psd_band defined above.
[P_data_plot,f_psd_plot] = normalized_psd_shape(y_data,fs,[0.5 10]);
[P_fit_plot,~] = normalized_psd_shape(y_fit,fs,[0.5 10]);

phi_data = data_features.phase;
phi_fit = fit_features.phase;
phase_diff_unwrapped = n_lock*phi_data - m_lock*phi_fit;
phase_diff_wrapped = wrap_pi(phase_diff_unwrapped);
Delta0 = angle(mean(exp(1i*phase_diff_wrapped)));
R = abs(mean(exp(1i*phase_diff_wrapped)));
dphi_corrected = wrap_pi(phase_diff_wrapped-Delta0);
fprintf('Final long-transient phase locking R_{%d:%d} = %.4f\n', ...
    n_lock,m_lock,R);

% Save the fitted point in both coordinate systems together with the
% long-transient dynamical diagnostics used for interpretation.
paper_region_label = {paperRegionFromJ(best_j)};
trajectory_label = {raw_dynamics.classification};
fit_mode_label = {fit_mode};
best_result = table(fit_mode_label,best_j,best_P,best_C,best_p,best_G,best_d, ...
    basePar.r*basePar.v0,best_cost,best_info.R,R,f_target, ...
    raw_dynamics.f_dom, ...
    raw_dynamics.raw_std,basePar.r*raw_dynamics.raw_std, ...
    basePar.r*raw_dynamics.raw_min,basePar.r*raw_dynamics.raw_max, ...
    raw_dynamics.sustain_ratio,paper_region_label,trajectory_label, ...
    'VariableNames',{'FitMode','j','P','C','p','G','d','log_k0','FitCost', ...
    'Optimization_R','FinalLongTransient_R','MEG_TargetFrequency_Hz', ...
    'RawModelFrequency_Hz', ...
    'RawStd','PaperX_Std','PaperX_Min','PaperX_Max','SustainRatio', ...
    'PaperRegion','TrajectoryClass'});
best_output_file = sprintf( ...
    'Jansen_best_fit_%s_paper_coordinates.csv',output_tag);
try
    writetable(best_result,best_output_file);
    fprintf('Saved best-fit paper coordinates to %s\n',best_output_file);
catch ME
    warning('Could not write %s: %s',best_output_file,ME.message);
end

%% ========================================================================
%  8. Diagnostic figures
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

% ------------------------------------------------------------------------
% Separate amplitude/PSD/residual diagnostics.  Keeping these in a second
% figure avoids squeezing the original optimization summary.
diagPos = figPos;
diagPos(1) = figPos(1)+50;
diagPos(2) = figPos(2)+50;
figure('Position',diagPos,'Color','w');

subplot(2,2,1);
plot(t_exp,y_data_raw,'r','LineWidth',1.1); hold on;
plot(t_exp,EEG_fit_affine,'b','LineWidth',1.0);
xlabel('Time (s)'); ylabel('MEG scale');
title(sprintf('Affine Observation Fit: gain=%.3g, offset=%.3g', ...
    obs_gain,obs_offset));
legend('MEG','affine-aligned model','Location','best');
grid on;

subplot(2,2,2);
plot(t_exp,env_data_relative,'r','LineWidth',1.2); hold on;
plot(t_exp,env_fit_relative,'b','LineWidth',1.1);
xlabel('Time (s)'); ylabel('Envelope / mean envelope');
title(sprintf('Relative Hilbert Envelope: E_{env}=%.4f, \\rho=%.3f', ...
    best_info.E_env,best_info.rho_env));
legend('MEG envelope','model envelope','Location','best');
grid on;

subplot(2,2,3);
plot(f_psd_plot,P_data_plot,'r','LineWidth',1.2); hold on;
plot(f_psd_plot,P_fit_plot,'b','LineWidth',1.1);
xlabel('Frequency (Hz)'); ylabel('Normalized spectral power');
xlim([0.5 10]);
title(sprintf('PSD Shape: E_{PSD}=%.4f',best_info.E_psd));
legend('MEG PSD','model PSD','Location','best');
grid on;

subplot(2,2,4);
plot(t_exp,affine_residual,'k','LineWidth',0.9);
yline(0,'r--','LineWidth',1.0);
xlabel('Time (s)'); ylabel('MEG - aligned model');
title(sprintf('Observation Residual: NRMSE=%.4f',affine_nrmse));
grid on;

sgtitle({ ...
    'Amplitude, Spectrum, and Residual Diagnostics', ...
    sprintf('j=%.4f, P=%.4f; paper region %s', ...
    best_j,best_P,paperRegionFromJ(best_j))}, ...
    'FontWeight','bold');

% Raw dynamics must be inspected before assigning a bifurcation meaning.
rawPos = figPos;
rawPos(1) = figPos(1)+100;
rawPos(2) = figPos(2)+100;
figure('Position',rawPos,'Color','w');

subplot(3,1,1);
plot(t_exp,X_fit_paper,'k','LineWidth',1.0);
xlabel('Time (s)'); ylabel('Paper X=r(y_1-y_2)');
title(sprintf(['Unfiltered Model Output: f=%.3f Hz, std(X)=%.4g, ', ...
               'sustain=%.3f'], ...
    raw_dynamics.f_dom,basePar.r*raw_dynamics.raw_std, ...
    raw_dynamics.sustain_ratio));
grid on;

subplot(3,1,2);
plot(P_grid,paper_X_min_profile,'b-','LineWidth',1.3); hold on;
plot(P_grid,paper_X_max_profile,'r-','LineWidth',1.3);
xlabel('Paper input P'); ylabel('Paper X');
title(sprintf(['One-Seed Numerical X Envelope at j=%.4f ', ...
               '(not all branches)'],best_j));
legend('minimum X','maximum X','Location','best');
grid on;

subplot(3,1,3);
yyaxis left;
plot(P_grid,paper_X_std_profile,'b-','LineWidth',1.3);
ylabel('Paper-X standard deviation');
yyaxis right;
plot(P_grid,sustain_profile,'m-','LineWidth',1.3); hold on;
yline(signal_cfg.min_sustain_ratio,'k--');
yline(signal_cfg.max_sustain_ratio,'k--');
ylabel('Sustain ratio');
xlabel('Paper input P');
title(sprintf('Raw-Dynamics Check at Fixed j=%.4f',best_j));
grid on;

fprintf('\n=== Completed successfully ===\n');

%% ========================================================================
%  Local functions
%  ========================================================================

function par = defaultJansenParameters
% Standard parameter set from Jansen_Scenario_Study.m.

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
% Connectivity conditions:
% C1=C, C2=0.8C, C3=0.25C, C4=0.25C.
par.C1 = 1.00*par.C;
par.C2 = 0.80*par.C;
par.C3 = 0.25*par.C;
par.C4 = 0.25*par.C;
end

function dX = jansenRHS(~,X,par,p)
% Six-dimensional Jansen neural mass model.
%
% State vector:
% X = [y0;y1;y2;y3;y4;y5],
% y3 = dy0/dt, y4 = dy1/dt, y5 = dy2/dt.
%
% Mathematical system:
% y0' = y3
% y1' = y4
% y2' = y5
% y3' = A*a*S(y1-y2) - 2*a*y3 - a^2*y0
% y4' = A*a*[p + C2*S(C1*y0)] - 2*a*y4 - a^2*y1
% y5' = B*b*C4*S(C3*y0) - 2*b*y5 - b^2*y2
% EEG = y1-y2.

y0 = X(1); y1 = X(2); y2 = X(3);
y3 = X(4); y4 = X(5); y5 = X(6);

S0 = sigmoidFunction(y1-y2,par);
S1 = sigmoidFunction(par.C1*y0,par);
S2 = sigmoidFunction(par.C3*y0,par);

dX = zeros(6,1);
dX(1) = y3;
dX(2) = y4;
dX(3) = y5;
dX(4) = par.A*par.a*S0 ...
      - 2*par.a*y3 ...
      - par.a^2*y0;
dX(5) = par.A*par.a*(p + par.C2*S1) ...
      - 2*par.a*y4 ...
      - par.a^2*y1;
dX(6) = par.B*par.b*par.C4*S2 ...
      - 2*par.b*y5 ...
      - par.b^2*y2;
end

function S = sigmoidFunction(v,par)
% Sigmoid firing-rate law:
% S(v) = nuMax / [1 + exp(r(v0-v))].
S = par.nuMax ./ (1 + exp(par.r*(par.v0-v)));
end

function [r,info] = jansen_residual_weighted( ...
    theta,basePar,sim,t_exp,y_data,data_features,fs, ...
    f_target,f_scale,weights,n_lock,m_lock)

% The fitted coordinates are exactly those used in Touboul et al.:
% theta=[j,P].  The dimensional C and p are derived, never independently
% optimized, so every trial stays in the paper's fixed-(G,d) plane.
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

% Determine the model's dynamical behavior BEFORE applying the comparison
% filter.  Only this raw frequency can be interpreted using the paper's
% fixed points and limit-cycle branches.
raw_diag = trajectoryDiagnostics( ...
    EEG_fit,fs,data_features.signal_cfg);

if data_features.signal_cfg.require_sustained_oscillation && ...
        ~raw_diag.is_sustained
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

% pdata.PdataF is already narrow-band filtered.  Filter the simulated EEG
% only for waveform, envelope, PSD, and phase comparisons with the MEG.
EEG_fit_filtered = comparison_bandpass(EEG_fit,fs);
y_fit = normalize01(EEG_fit_filtered);
fit_features = build_signal_features( ...
    y_fit,fs,data_features.psd_band);

if length(fit_features.psd) ~= length(data_features.psd) || ...
        any(~isfinite(fit_features.psd)) || ...
        any(~isfinite(fit_features.envelope))
    [r,info] = penaltyResidual(data_features.residual_length);
    return;
end

% A. Waveform residual
% E_y = (1/N) sum_k [y_data(k)-y_fit(k)]^2.
r_waveform = sqrt(weights.waveform/length(y_data))*(y_data-y_fit);
E_y = mean((y_data-y_fit).^2);

% B. Relative Hilbert amplitude-envelope residual.  Each envelope is
% divided by its own mean; therefore E_env measures temporal modulation,
% not an invalid equality between MEG sensor units and model mV.
env_difference = data_features.envelope-fit_features.envelope;
env_denominator = max(sum(data_features.envelope.^2),eps);
r_envelope = sqrt(weights.envelope/env_denominator)*env_difference;
E_env = sum(env_difference.^2)/env_denominator;
rho_env = safe_correlation( ...
    data_features.envelope,fit_features.envelope);

% C. Normalized PSD-shape residual.  Square-root spectra give a bounded,
% symmetric Hellinger-type discrepancy for unit-sum spectral powers.
sqrt_psd_data = sqrt(max(data_features.psd,0));
sqrt_psd_fit = sqrt(max(fit_features.psd,0));
psd_difference = sqrt_psd_data-sqrt_psd_fit;
r_psd = sqrt(weights.psd)*psd_difference;
E_psd = sum(psd_difference.^2);

% D. Frequency residual.  This intentionally uses the unfiltered model
% output; otherwise a 3.6--4.4 Hz bandpass makes a 4-Hz score circular.
% E_f = [(f_raw_model-f_target)/f_scale]^2.
f_dom_raw = raw_diag.f_dom;
f_dom_filtered = get_dominant_freq( ...
    y_fit,fs,data_features.signal_cfg.raw_frequency_band);
r_freq = sqrt(weights.frequency)*(f_dom_raw-f_target)/f_scale;
E_f = ((f_dom_raw-f_target)/f_scale)^2;

% E. Phase-locking residual
% R_{n:m}=|mean(exp(i(n phi_data-m phi_fit)))|.
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
% Two-stage integration:
% 1) transient interval [0,T_trans];
% 2) observation interval t_exp.

ok = false;
X_fit = [];
EEG_fit = [];

t_start = tic;
stop_fcn = @(t,y,flag) stop_by_timeout( ...
    t,y,flag,t_start,sim.final_timeout);

ode_opts = odeset( ...
    'OutputFcn',stop_fcn, ...
    'RelTol',sim.RelTol, ...
    'AbsTol',sim.AbsTol, ...
    'MaxStep',sim.MaxStep);

try
    [~,Xtrans] = ode15s( ...
        @(t,X) jansenRHS(t,X,par,p), ...
        [0 sim.T_trans],sim.X0,ode_opts);

    if isempty(Xtrans) || toc(t_start) > sim.final_timeout
        return;
    end

    X0_steady = Xtrans(end,:)';

    [~,X_fit] = ode15s( ...
        @(t,X) jansenRHS(t,X,par,p), ...
        t_exp,X0_steady,ode_opts);

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
% Reconstruct full theta=[j,P] from the selected fitting mode.
% NaN entries in fixed_params are filled, in order, by theta_var.
theta_full = fixed_params;
var_idx = isnan(fixed_params);

if numel(theta_var) ~= sum(var_idx)
    error(['Number of supplied free parameters (%d) does not match ', ...
           'the number of NaN entries in fixed_params (%d).'], ...
        numel(theta_var),sum(var_idx));
end

theta_full(var_idx) = reshape(theta_var,1,[]);
end

function [C,p,G,d] = paperToDimensional(j,P,par)
% Touboul et al. dimensionless-to-dimensional parameter map:
%   j=(r*A*nuMax/a)C, P=(r*A/a)p, G=B/A, d=b/a.
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

function region = paperRegionFromJ(j)
% Vertical j-zones in Touboul et al. Figure 4/Table 4.  P and the initial
% condition still determine which attractor is actually observed.
if j < 5.38
    region = 'A: unique stable equilibrium';
elseif j < 10.05
    region = 'B: bistable equilibria';
elseif j < 12.10
    region = 'C: bistable/nonoscillatory deterministic regime';
elseif j < 12.38
    region = 'D: alpha and low-frequency cycles may coexist';
elseif j < 12.48
    region = 'E: mixed alpha/low-frequency cycle regime';
elseif j < 12.55
    region = 'F: low-frequency cycle can be sole attractor';
elseif j < 12.93
    region = 'G: low-frequency cycle can be sole attractor';
else
    region = 'H: connected low-frequency/theta/alpha cycle family';
end
end

function diag = trajectoryDiagnostics(sig,fs,cfg)
% Classify only what a finite trajectory can justify.  A sustained trace is
% a limit-cycle candidate; a true bifurcation/stability claim still requires
% equilibrium and periodic-orbit continuation.
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

has_amplitude = isfinite(diag.raw_std) && ...
    diag.raw_std >= cfg.min_raw_std;
has_stationary_amplitude = isfinite(diag.sustain_ratio) && ...
    diag.sustain_ratio >= cfg.min_sustain_ratio && ...
    diag.sustain_ratio <= cfg.max_sustain_ratio;
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
% Precompute the signal quantities used by the enhanced objective.
sig = sig(:);
features.phase = get_phase(sig);
features.envelope = relative_hilbert_envelope(sig);
[features.psd,features.psd_frequency] = ...
    normalized_psd_shape(sig,fs,psd_band);
features.psd_band = psd_band;

% Residual blocks: waveform N, envelope N, PSD K, frequency 1, phase 1.
features.residual_length = 2*length(sig)+length(features.psd)+2;
end

function envelope_relative = relative_hilbert_envelope(sig)
% Hilbert amplitude divided by its mean.  This retains temporal amplitude
% modulation while removing an arbitrary global observation scale.
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
% Toolbox-free, one-sided Hann-window periodogram normalized to unit sum.
% Only spectral shape is compared; absolute spectral power is intentionally
% excluded because MEG and the neural-mass output use different units.
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
% Correlation without requiring the Statistics Toolbox.
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

function [model_aligned,gain,offset] = affine_align_signal(model,data)
% Least-squares observation map: data ~= offset + gain*model.
model = model(:);
data = data(:);
model_centered = model-mean(model);
data_centered = data-mean(data);
denominator = model_centered'*model_centered;

if denominator > eps
    gain = (model_centered'*data_centered)/denominator;
else
    gain = 0;
end

offset = mean(data)-gain*mean(model);
model_aligned = offset+gain*model;
end

function y = comparison_bandpass(x,fs)
% Zero-phase comparison filter for the same narrow band represented by
% pdata.PdataF.  The direct bandpass call is used when Signal Processing
% Toolbox is available.  A zero-phase FFT taper is provided as a fallback.

x = x(:);

if exist('bandpass','file') == 2
    try
        y = bandpass(x,[3.6 4.4],fs);
        y = y(:);
        return;
    catch
        % Continue to the toolbox-free FFT implementation below.
    end
end

Nf = length(x);
freq = (0:Nf-1)'*fs/Nf;
abs_freq = min(freq,fs-freq);

% Smooth transitions: stop/pass/pass/stop = 3.4/3.6/4.4/4.6 Hz.
H = zeros(Nf,1);
H(abs_freq >= 3.6 & abs_freq <= 4.4) = 1;

idx_low = abs_freq > 3.4 & abs_freq < 3.6;
H(idx_low) = 0.5*(1-cos(pi*(abs_freq(idx_low)-3.4)/(3.6-3.4)));

idx_high = abs_freq > 4.4 & abs_freq < 4.6;
H(idx_high) = 0.5*(1+cos(pi*(abs_freq(idx_high)-4.4)/(4.6-4.4)));

y = real(ifft(fft(x).*H));
end

function f_dom = get_dominant_freq(sig,fs,band)
% Dominant non-DC frequency, optionally restricted to band=[fmin,fmax].
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

% Ignore the DC component explicitly.
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
% Toolbox-free analytic signal construction equivalent to Hilbert method.
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
