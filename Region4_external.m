clc; clearvars; close all;

% Parameters
M0 = 1e6; % Magnetisation amplitude [A/m]
P = 4; % Pole-pair number (integer ≥ 1)
Ri = 0.005; % Inner radius [m]
Ro = 0.015; % Outer radius [m]
Wr = 0.02; % Magnet height (total) [m]
N_r = 50; % Radial collocation points
N_z = 50; % Axial collocation points
theta0 = 0; % Azimuthal angle
k_min = 1e-10; % Minimum k search bound [1/m] 
k_max = 5000; % Maximum k search bound [1/m] 
MAX_TERMS = 200; % Maximum number of modes
TARGET_REL_ERR = 0.01; % 1% target RMS error
ABS_TOL = 1e-8; % Integral absolute tolerance
REL_TOL = 1e-4; % Integral relative tolerance
ORTHO_TOL = 1e-4; % Looser orthogonality tolerance
MIN_NORM = 1e-6; % Looser minimum vector norm threshold
MAX_TRIALS_PER_TERM = 100; % Trials per term

% Create collocation grid (ρ, z) for region IV
rho_vec = linspace(Ro, Ro*1.5, N_r);
z_vec = linspace(0, Wr/2, N_z);
[RR, ZZ] = ndgrid(rho_vec, z_vec);
ntotal = numel(RR);

% Run only for external rotor
rotor_type = 'external';
fprintf('\n===== Processing %s rotor configuration (Region IV) =====\n', rotor_type);

% Compute exact potential φ_exact on grid
fprintf('Evaluating exact potential on %d x %d grid...\n', N_r, N_z);
phi_exact = zeros(ntotal, 1);
volume_factor = (1 - P); % External rotor factor

warnState = warning('off', 'all');
for idx = 1:ntotal
    rho = RR(idx);
    z = ZZ(idx);
    phi_val = computePoint(rho, z, theta0, Ri, Ro, Wr, P, M0, volume_factor, ABS_TOL, REL_TOL);
    phi_exact(idx) = phi_val;
end
warning(warnState);

fprintf('Exact potential computed.\n');
norm_exact = norm(phi_exact);
fprintf('Exact potential RMS norm: %.4g\n', norm_exact);

% Scale for numerical stability
potential_scale = norm_exact;
phi_exact_scaled = phi_exact / potential_scale;

% Initialize variables
k_list = [];
type_list = [];
C_list_scaled = [];
B_normalized = [];
err_hist = [];
current_err = 1.0;
n_terms = 0;

fprintf('\nStarting adaptive greedy basis construction...\n');

for n = 1:MAX_TERMS
    % Compute current residual
    if isempty(B_normalized)
        r0 = phi_exact_scaled;
    else
        % Use least squares with regularization
        [U, S, V] = svd(B_normalized, 'econ');
        s = diag(S);
        rank = sum(s > 1e-12 * max(s));
        if rank > 0
            lambda = 1e-8 * max(s);
            s_inv = diag(s(1:rank) ./ (s(1:rank).^2 + lambda^2));
            C_current = V(:, 1:rank) * s_inv * (U(:, 1:rank)' * phi_exact_scaled);
            r0 = phi_exact_scaled - B_normalized * C_current;
        else
            r0 = phi_exact_scaled;
        end
    end
    
    norm_r0 = norm(r0);
    if n_terms == 0
        fprintf('Initial residual norm: %.6f\n', norm_r0);
    end
    
    % Adaptive k-space scanning based on current error
    if current_err > 0.5
        % High error - scan broadly
        k_scans = {
            logspace(log10(k_min), log10(1e-2), 100),
            logspace(log10(1e-2), log10(1), 200),
            logspace(log10(1), log10(100), 300),
            logspace(log10(100), log10(k_max), 200)
        };
    elseif current_err > 0.1
        % Medium error - focus on medium frequencies
        k_scans = {
            logspace(log10(1e-3), log10(10), 400),
            logspace(log10(10), log10(500), 400)
        };
    else
        % Low error - refine existing frequencies
        if ~isempty(k_list)
            % Focus around existing k values
            k_centers = unique(k_list);
            k_scans = {};
            for kc = k_centers
                if kc > 0
                    lb = max(k_min, 0.1 * kc);
                    ub = min(k_max, 10 * kc);
                    k_scans{end+1} = logspace(log10(lb), log10(ub), 200);
                end
            end
            % Also include some random exploration
            k_scans{end+1} = logspace(log10(k_min), log10(k_max), 300);
        else
            k_scans = {logspace(log10(k_min), log10(k_max), 500)};
        end
    end
    
    best_candidate = [];
    best_candidate_err = inf;
    
    fprintf('Term %d: Scanning %d frequency ranges...\n', n, length(k_scans));
    
    for scan_idx = 1:length(k_scans)
        k_scan = k_scans{scan_idx};
        
        for type_idx = 1:2
            for k_idx = 1:length(k_scan)
                k0 = k_scan(k_idx);
                
                % Skip if this k-type combination already exists
                if any(k_list == k0 & type_list == type_idx)
                    continue;
                end
                
                v = basisVectorIV(k0, type_idx, RR, ZZ, P);
                v_norm = norm(v);
                
                if v_norm < MIN_NORM
                    continue;
                end
                
                v_normalized = v / v_norm;
                
                % Check orthogonality with current basis
                if ~isempty(B_normalized)
                    proj = B_normalized' * v_normalized;
                    if any(abs(proj) > 0.99)  % Nearly parallel to existing basis
                        continue;
                    end
                end
                
                % Simple projection test - how much does this reduce residual?
                proj_r = dot(r0, v_normalized);
                candidate_err = sqrt(max(0, norm_r0^2 - proj_r^2)) / norm(phi_exact_scaled);
                
                if candidate_err < best_candidate_err
                    best_candidate_err = candidate_err;
                    best_candidate = struct('k', k0, 'type', type_idx, 'v', v, 'v_norm', v_norm);
                end
            end
        end
    end
    
    if isempty(best_candidate)
        fprintf('No candidate found in scan. Trying emergency measures...\n');
        
        % Emergency: try random k values
        emergency_ks = [logspace(log10(k_min), log10(k_max), 1000), ...
                       rand(1, 500) * (k_max - k_min) + k_min];
        
        for k0 = emergency_ks
            for type_idx = 1:2
                v = basisVectorIV(k0, type_idx, RR, ZZ, P);
                v_norm = norm(v);
                if v_norm > MIN_NORM
                    v_normalized = v / v_norm;
                    proj_r = dot(r0, v_normalized);
                    candidate_err = sqrt(max(0, norm_r0^2 - proj_r^2)) / norm(phi_exact_scaled);
                    
                    if candidate_err < best_candidate_err
                        best_candidate_err = candidate_err;
                        best_candidate = struct('k', k0, 'type', type_idx, 'v', v, 'v_norm', v_norm);
                    end
                end
            end
        end
    end
    
    if isempty(best_candidate)
        fprintf('No valid candidate found after emergency scan. Stopping.\n');
        break;
    end
    
    % Add the best candidate
    k_opt = best_candidate.k;
    type_opt = best_candidate.type;
    v_opt_normalized = best_candidate.v / best_candidate.v_norm;
    
    B_temp = [B_normalized, v_opt_normalized];
    
    % Solve for new coefficients
    [U, S, V] = svd(B_temp, 'econ');
    s = diag(S);
    rank = sum(s > 1e-12 * max(s));
    
    if rank > 0
        lambda = 1e-8 * max(s);
        s_inv = diag(s(1:rank) ./ (s(1:rank).^2 + lambda^2));
        C_temp = V(:, 1:rank) * s_inv * (U(:, 1:rank)' * phi_exact_scaled);
        residual = phi_exact_scaled - B_temp * C_temp;
        current_err_candidate = norm(residual);
    else
        C_temp = [];
        current_err_candidate = 1.0;
    end
    
    % Always accept if it improves error
    if current_err_candidate <= current_err || n_terms < 5  % Force first 5 terms
        k_list(end+1) = k_opt;
        type_list(end+1) = type_opt;
        B_normalized = B_temp;
        C_list_scaled = C_temp;
        prev_err = current_err;
        current_err = current_err_candidate;
        err_hist(end+1) = current_err;
        n_terms = n_terms + 1;
        
        error_reduction = prev_err - current_err;
        fprintf('Term %d: type=%d, k_n=%.6f, RelErr=%.6f%%, Δ=%.6f%%, max|C|=%.2e\n', ...
            n_terms, type_opt, k_opt, 100*current_err, 100*error_reduction, max(abs(C_temp)));
        
        % Check convergence
        if current_err < TARGET_REL_ERR
            fprintf('\n🎯 TARGET ACHIEVED: Error (%.6f%%) < target (%.4f%%) with %d terms!\n', ...
                100*current_err, 100*TARGET_REL_ERR, n_terms);
            break;
        end
        
        % Force at least 10 terms before considering stopping
        if n_terms >= 10 && error_reduction < 1e-6
            fprintf('Minimal improvement. Stopping.\n');
            break;
        end
    else
        fprintf('Candidate rejected (no improvement). Continuing search...\n');
    end
    
    % Don't stop early - keep going until we have reasonable terms
    if n_terms >= 50 && current_err < 0.05  % Good enough
        fprintf('Reasonable accuracy achieved with %d terms. Stopping.\n', n_terms);
        break;
    end
end

% Transform coefficients back to original space
fprintf('\nComputing final coefficients...\n');
C_list_final = zeros(size(C_list_scaled));

for i = 1:length(k_list)
    basis_func = basisVectorIV(k_list(i), type_list(i), RR, ZZ, P);
    basis_norm = norm(basis_func);
    if basis_norm > 0
        C_list_final(i) = C_list_scaled(i) * potential_scale / basis_norm;
    else
        C_list_final(i) = 0;
    end
end
C_list = C_list_final;

% Output results
fprintf('\n%s Rotor (Region IV): Results\n', upper(rotor_type));
fprintf('%-6s %-12s %-15s %-15s\n', 'Index', 'k_n [1/m]', 'Bessel Type', 'C(n)');
fprintf('------------------------------------------------\n');
for i = 1:length(k_list)
    bessel_type = {'J_P', 'Y_P'};
    fprintf('%-6d %-12.6f %-15s %-15.6g\n', i, k_list(i), bessel_type{type_list(i)}, C_list(i));
end
fprintf('------------------------------------------------\n');
fprintf('Final RMS error: %.6f%%\n', 100*current_err);
fprintf('Number of terms: %d\n', n_terms);

% Compute verification
fprintf('\nComputing verification...\n');
phi_approx = zeros(size(phi_exact));
for i = 1:length(k_list)
    basis_func = basisVectorIV(k_list(i), type_list(i), RR, ZZ, P);
    phi_approx = phi_approx + C_list(i) * basis_func;
end

abs_error = phi_exact - phi_approx;
rel_error = abs_error ./ (abs(phi_exact) + eps);
rms_abs_error = sqrt(mean(abs_error.^2));
rms_rel_error = sqrt(mean(rel_error.^2));

fprintf('\n=== VERIFICATION ===\n');
fprintf('RMS Absolute Error: %.6e\n', rms_abs_error);
fprintf('RMS Relative Error: %.6f%%\n', 100*rms_rel_error);
fprintf('Maximum Absolute Error: %.6e\n', max(abs(abs_error)));
fprintf('Maximum Relative Error: %.6f%%\n', 100*max(abs(rel_error)));

% Plot results
phi_exact_grid = reshape(phi_exact, size(RR));
phi_approx_grid = reshape(phi_approx, size(RR));
abs_error_grid = reshape(abs_error, size(RR));
rel_error_grid = reshape(rel_error, size(RR));

plotResults(RR, ZZ, phi_exact_grid, phi_approx_grid, abs_error_grid, rel_error_grid, ...
           k_list, type_list, C_list, err_hist, Ri, Ro, Wr, rotor_type);

% Save results
eigenvalues = k_list;
basis_types = type_list;
coefficients = C_list;

save('external_rotor_results.mat', 'eigenvalues', 'basis_types', 'coefficients', ...
     'rotor_type', 'P', 'Ri', 'Ro', 'Wr', 'phi_exact', 'phi_approx', ...
     'rms_abs_error', 'rms_rel_error', 'TARGET_REL_ERR');

fprintf('Results saved to external_rotor_results.mat\n');

%% Helper Functions
function phi = computePoint(rho, z, theta, Ri, Ro, Wr, P, M0, volume_factor, abs_tol, rel_tol)
phi = 0;
eps_dist = 1e-12;

try
    intVol = @(rp, tp, zp) cos(P*tp) ./ max(sqrt(rho^2 + rp.^2 - 2*rho*rp.*cos(theta-tp) + (z-zp).^2), eps_dist);
    intRi = @(tp, zp) cos(P*tp) ./ max(sqrt(rho^2 + Ri^2 - 2*rho*Ri*cos(theta-tp) + (z-zp).^2), eps_dist);
    intRo = @(tp, zp) cos(P*tp) ./ max(sqrt(rho^2 + Ro^2 - 2*rho*Ro*cos(theta-tp) + (z-zp).^2), eps_dist);
    
    I1 = integral3(intVol, Ri, Ro, 0, 2*pi, -Wr/2, Wr/2, 'AbsTol', abs_tol, 'RelTol', rel_tol);
    I2 = integral2(intRi, 0, 2*pi, -Wr/2, Wr/2, 'AbsTol', abs_tol, 'RelTol', rel_tol);
    I3 = integral2(intRo, 0, 2*pi, -Wr/2, Wr/2, 'AbsTol', abs_tol, 'RelTol', rel_tol);
    
    phi = M0/(4*pi) * (volume_factor * I1 - Ri*I2 + Ro*I3);
    
catch
    phi = 0;
end
end

function v = basisVectorIV(k, type, RR, ZZ, P)
rho_vals = RR(:);
z_vals = ZZ(:);
k = abs(k);

if k > 1e4
    v = zeros(size(rho_vals));
    return;
end

switch type
    case 1
        v = exp(-k * abs(z_vals)) .* besselj(P, k * rho_vals);
    case 2
        valid = rho_vals > 1e-12;
        v = zeros(size(rho_vals));
        v(valid) = exp(-k * abs(z_vals(valid))) .* bessely(P, k * rho_vals(valid));
        v(isinf(v) | isnan(v)) = 0;
end
end

function plotResults(RR, ZZ, phi_exact, phi_approx, abs_error, rel_error, ...
                    k_list, type_list, C_list, err_hist, Ri, Ro, Wr, rotor_type)
    
    font_size = 12;
    
    % 1. Potential Comparison
    fig1 = figure('Position', [100, 100, 1200, 800]);
    sgtitle(sprintf('%s Rotor - Results', upper(rotor_type)), 'FontSize', font_size+2);
    
    subplot(2,3,1);
    contourf(RR, ZZ, phi_exact, 30, 'LineColor', 'none');
    colorbar; axis equal tight;
    title('Exact Potential');
    xlabel('\rho [m]'); ylabel('z [m]');
    
    subplot(2,3,2);
    contourf(RR, ZZ, phi_approx, 30, 'LineColor', 'none');
    colorbar; axis equal tight;
    title('Approximate Potential');
    xlabel('\rho [m]'); ylabel('z [m]');
    
    subplot(2,3,3);
    contourf(RR, ZZ, phi_exact - phi_approx, 30, 'LineColor', 'none');
    colorbar; axis equal tight;
    title('Difference');
    xlabel('\rho [m]'); ylabel('z [m]');
    
    subplot(2,3,4);
    contourf(RR, ZZ, abs_error, 30, 'LineColor', 'none');
    colorbar; axis equal tight;
    title('Absolute Error');
    xlabel('\rho [m]'); ylabel('z [m]');
    
    subplot(2,3,5);
    contourf(RR, ZZ, rel_error * 100, 30, 'LineColor', 'none');
    colorbar; axis equal tight;
    title('Relative Error (%)');
    xlabel('\rho [m]'); ylabel('z [m]');
    
    % Convergence
    subplot(2,3,6);
    if ~isempty(err_hist)
        semilogy(1:length(err_hist), err_hist*100, 'o-', 'LineWidth', 2);
        hold on;
        yline(1, 'r--', 'Target 1%', 'LineWidth', 2);
        grid on;
        xlabel('Number of Terms');
        ylabel('Relative RMS Error (%)');
        title('Convergence History');
        legend('Error', 'Target');
    end
    
    saveas(fig1, sprintf('%s_results.png', rotor_type));
    
    % 2. Coefficient Analysis
    if length(k_list) > 1
        fig2 = figure('Position', [100, 100, 1000, 400]);
        
        subplot(1,2,1);
        stem(1:length(C_list), abs(C_list), 'filled', 'LineWidth', 2);
        set(gca, 'YScale', 'log');
        grid on;
        title('Coefficient Magnitudes');
        xlabel('Term Index'); ylabel('|C(n)|');
        
        subplot(1,2,2);
        scatter(k_list, abs(C_list), 50, type_list, 'filled');
        set(gca, 'XScale', 'log', 'YScale', 'log');
        colorbar; grid on;
        title('Coefficients vs Wave Number');
        xlabel('k_n [1/m]'); ylabel('|C(n)|');
        
        saveas(fig2, sprintf('%s_coefficients.png', rotor_type));
    end
    
    fprintf('Plots saved.\n');
end