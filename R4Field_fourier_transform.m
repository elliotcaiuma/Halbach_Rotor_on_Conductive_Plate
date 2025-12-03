% Store the result as a field for a 3D grid
clc; clearvars; close all;

%% Load the decomposition results
fprintf('Loading decomposition results...\n');
load('external_rotor_results.mat');

%% Parameters (should match those used in decomposition)
Ri = 0.005;      % Inner radius [m]
Ro = 0.015;      % Outer radius [m]
Wr = 0.02;       % Magnet height (total) [m]
P = 4;           % Pole-pair number
mu0 = 4*pi*1e-7; % Permeability of free space

%% Create grid for Region IV only (r >= Ro) with at least 100 points in each direction
N_r = 120;       % Radial points
N_z = 60;        % Half z-direction points (will become 119 total)
N_theta = 100;   % Azimuthal points

% Region IV only: outside rotor
rho_min = Ro;           % Start at outer radius
rho_max = Ro * 1.5;     % Further out
z_min = 0;              % Start at z = 0
z_max = Wr/2;           % Up to magnet half-height

rho_vec = linspace(rho_min, rho_max, N_r);
z_vec_pos = linspace(z_min, z_max, N_z);
theta_vec = linspace(0, 2*pi, N_theta);

%% Create full symmetric grid by flipping over z=0 (Region IV only)
fprintf('Creating symmetric grid for Region IV (r >= Ro)...\n');

% Create full z vector (negative and positive) - symmetric
z_vec_full = [-fliplr(z_vec_pos(2:end)), z_vec_pos]; % Skip duplicate at z=0
[RR_full, ZZ_full] = ndgrid(rho_vec, z_vec_full);

%% Create 3D grid for Cartesian components (Region IV only)
[RR_3D, ZZ_3D, THETA_3D] = ndgrid(rho_vec, z_vec_full, theta_vec);

%% Compute decomposed potential and field components (Region IV only)
fprintf('Computing x, y, z field components from decomposition coefficients (Region IV)...\n');

% Initialize field components
Bx_decomp = zeros(size(RR_3D));
By_decomp = zeros(size(RR_3D));
Bz_decomp = zeros(size(RR_3D));

% Precompute angular terms for efficiency
cos_theta = cos(THETA_3D);
sin_theta = sin(THETA_3D);

% Precompute frequently used values
abs_ZZ = abs(ZZ_3D);  % For even symmetry

% Vectorized computation for better efficiency
fprintf('Vectorizing basis function computations...\n');

% Group computations by basis type for better cache utilization
type1_idx = find(basis_types == 1);
type2_idx = find(basis_types == 2);

% Pre-allocate temporary arrays
temp_Bx = zeros(size(RR_3D));
temp_By = zeros(size(RR_3D));
temp_Bz = zeros(size(RR_3D));

% Process type 1 basis functions (J_P)
if ~isempty(type1_idx)
    fprintf('Processing %d J_P type basis functions...\n', length(type1_idx));
    
    for idx = 1:length(type1_idx)
        i = type1_idx(idx);
        k_val = abs(eigenvalues(i));
        C = coefficients(i);
        
        if k_val > 1e4, continue; end  % Skip high frequency components
        
        % Vectorized computation for J_P type
        k_rho = k_val * RR_3D;
        J_P = besselj(P, k_rho);
        J_P_minus1 = besselj(P-1, k_rho);
        J_P_plus1 = besselj(P+1, k_rho);
        
        exp_term = exp(-k_val * abs_ZZ);
        
        % Radial derivative
        dJ_drho = k_val * (J_P_minus1 - J_P_plus1) / 2;
        dbasis_drho = exp_term .* dJ_drho .* cos(P * THETA_3D);
        
        % Axial derivative with even symmetry
        sign_ZZ = sign(ZZ_3D);
        sign_ZZ(ZZ_3D == 0) = 0;
        dbasis_dz = -k_val * exp_term .* J_P .* cos(P * THETA_3D) .* sign_ZZ;
        
        % Accumulate contributions
        temp_Bx = temp_Bx - C * mu0 * dbasis_drho .* cos_theta;
        temp_By = temp_By - C * mu0 * dbasis_drho .* sin_theta;
        temp_Bz = temp_Bz - C * mu0 * dbasis_dz;
    end
end

% Process type 2 basis functions (Y_P)
if ~isempty(type2_idx)
    fprintf('Processing %d Y_P type basis functions...\n', length(type2_idx));
    
    for idx = 1:length(type2_idx)
        i = type2_idx(idx);
        k_val = abs(eigenvalues(i));
        C = coefficients(i);
        
        if k_val > 1e4, continue; end  % Skip high frequency components
        
        % Vectorized computation for Y_P type (only where rho > 0)
        valid_mask = RR_3D > 1e-12;
        k_rho = k_val * RR_3D;
        
        Y_P = zeros(size(RR_3D));
        Y_P_minus1 = zeros(size(RR_3D));
        Y_P_plus1 = zeros(size(RR_3D));
        
        Y_P(valid_mask) = bessely(P, k_rho(valid_mask));
        Y_P_minus1(valid_mask) = bessely(P-1, k_rho(valid_mask));
        Y_P_plus1(valid_mask) = bessely(P+1, k_rho(valid_mask));
        
        exp_term = exp(-k_val * abs_ZZ);
        
        % Radial derivative
        dY_drho = k_val * (Y_P_minus1 - Y_P_plus1) / 2;
        dbasis_drho = exp_term .* dY_drho .* cos(P * THETA_3D);
        
        % Axial derivative with even symmetry
        sign_ZZ = sign(ZZ_3D);
        sign_ZZ(ZZ_3D == 0) = 0;
        dbasis_dz = -k_val * exp_term .* Y_P .* cos(P * THETA_3D) .* sign_ZZ;
        
        % Accumulate contributions
        Bx_decomp = Bx_decomp - C * mu0 * dbasis_drho .* cos_theta;
        By_decomp = By_decomp - C * mu0 * dbasis_drho .* sin_theta;
        Bz_decomp = Bz_decomp - C * mu0 * dbasis_dz;
    end
end

% Combine contributions
Bx_decomp = Bx_decomp + temp_Bx;
By_decomp = By_decomp + temp_By;
Bz_decomp = Bz_decomp + temp_Bz;

% Compute magnitude of decomposed field
B_mag_3D = sqrt(Bx_decomp.^2 + By_decomp.^2 + Bz_decomp.^2);

%% Fix constant ZData warning by checking field variation
fprintf('Checking field component variations...\n');

% Check if any field components are constant (which causes contour warnings)
Bx_range = range(Bx_decomp(:));
By_range = range(By_decomp(:));
Bz_range = range(Bz_decomp(:));
B_mag_range = range(B_mag_3D(:));

fprintf('Field component ranges:\n');
fprintf('Bx range: %.3e T\n', Bx_range);
fprintf('By range: %.3e T\n', By_range);
fprintf('Bz range: %.3e T\n', Bz_range);
fprintf('|B| range: %.3e T\n', B_mag_range);

%% Output field components information (Region IV only)
fprintf('\n=== DECOMPOSED FIELD COMPONENTS OUTPUT (REGION IV ONLY) ===\n');
fprintf('Grid dimensions: %d x %d x %d (rho x z x theta)\n', size(RR_3D));
fprintf('Radial range: %.3f m to %.3f m\n', rho_min, rho_max);
fprintf('Axial range: %.3f m to %.3f m\n', min(z_vec_full), max(z_vec_full));
fprintf('Azimuthal points: %d\n', N_theta);
fprintf('Total grid points: %d\n', numel(RR_3D));

% Field component ranges in Region IV
fprintf('\nField component ranges:\n');
fprintf('Bx: [%.3e, %.3e] T\n', min(Bx_decomp(:)), max(Bx_decomp(:)));
fprintf('By: [%.3e, %.3e] T\n', min(By_decomp(:)), max(By_decomp(:)));
fprintf('Bz: [%.3e, %.3e] T\n', min(Bz_decomp(:)), max(Bz_decomp(:)));
fprintf('|B|: [%.3e, %.3e] T\n', min(B_mag_3D(:)), max(B_mag_3D(:)));

% Additional statistics
fprintf('\nField Statistics:\n');
fprintf('Bx RMS: %.3e T\n', sqrt(mean(Bx_decomp(:).^2)));
fprintf('By RMS: %.3e T\n', sqrt(mean(By_decomp(:).^2)));
fprintf('Bz RMS: %.3e T\n', sqrt(mean(Bz_decomp(:).^2)));
fprintf('|B| RMS: %.3e T\n', sqrt(mean(B_mag_3D(:).^2)));

% Decomposition information
fprintf('\nDecomposition Information:\n');
fprintf('Number of terms: %d\n', length(eigenvalues));
fprintf('Pole pairs: %d\n', P);
fprintf('Coefficient range: [%.3e, %.3e]\n', min(coefficients), max(coefficients));

% Save field components to file (Region IV only)
field_data.Bx = Bx_decomp;
field_data.By = By_decomp;
field_data.Bz = Bz_decomp;
field_data.B_mag = B_mag_3D;
field_data.RR = RR_3D;
field_data.ZZ = ZZ_3D;
field_data.THETA = THETA_3D;
field_data.rho_vec = rho_vec;
field_data.z_vec = z_vec_full;
field_data.theta_vec = theta_vec;
field_data.eigenvalues = eigenvalues;
field_data.basis_types = basis_types;
field_data.coefficients = coefficients;
field_data.P = P;
field_data.Ri = Ri;
field_data.Ro = Ro;
field_data.Wr = Wr;
field_data.region = 'IV (r >= Ro)';

save('decomposed_field_components_region_IV.mat', '-struct', 'field_data');
fprintf('\nField components saved to decomposed_field_components_region_IV.mat\n');

%% Create field components visualization (Region IV only)
createFieldComponentsPlots(RR_3D, ZZ_3D, THETA_3D, Bx_decomp, By_decomp, Bz_decomp, B_mag_3D, ...
                          Ri, Ro, Wr, P, eigenvalues, coefficients);

%% Efficient Fourier Transform of Field Components in X and Z Directions
fprintf('\n=== PERFORMING EFFICIENT FOURIER TRANSFORM IN X AND Z DIRECTIONS ===\n');

% Convert cylindrical coordinates to Cartesian coordinates for Region IV
fprintf('Converting cylindrical to Cartesian coordinates...\n');
X_3D = RR_3D .* cos(THETA_3D);
Y_3D = RR_3D .* sin(THETA_3D);
Z_3D = ZZ_3D;

% Define the region of interest bounds
x_min = -rho_max; x_max = rho_max;
y_min = -rho_max; y_max = rho_max;
z_min = -Wr/2; z_max = Wr/2;

fprintf('Region of interest bounds:\n');
fprintf('X: [%.3f, %.3f] m\n', x_min, x_max);
fprintf('Y: [%.3f, %.3f] m\n', y_min, y_max);
fprintf('Z: [%.3f, %.3f] m\n', z_min, z_max);

% Create optimized Cartesian grid for Fourier transform
fprintf('Creating optimized Cartesian grid for Fourier transform...\n');
N_x = 100;  % Reduced for faster computation
N_y = 100;  
N_z = 100;  

x_vec = linspace(x_min, x_max, N_x);
y_vec = linspace(y_min, y_max, N_y);
z_vec = linspace(z_min, z_max, N_z);

[X_grid, Y_grid, Z_grid] = ndgrid(x_vec, y_vec, z_vec);

%% Fast Interpolation using direct cylindrical to Cartesian mapping
fprintf('Performing fast cylindrical to Cartesian interpolation...\n');

% Initialize field components on Cartesian grid
Bx_cart = zeros(N_x, N_y, N_z);
By_cart = zeros(N_x, N_y, N_z);
Bz_cart = zeros(N_x, N_y, N_z);

% Convert Cartesian grid to cylindrical coordinates for direct lookup
fprintf('Converting Cartesian grid to cylindrical coordinates...\n');
RHO_grid = sqrt(X_grid.^2 + Y_grid.^2);
THETA_grid = atan2(Y_grid, X_grid);
THETA_grid(THETA_grid < 0) = THETA_grid(THETA_grid < 0) + 2*pi; % Wrap to [0, 2*pi]
Z_grid_sym = abs(Z_grid); % Use absolute z for even symmetry

% Find indices for nearest neighbor interpolation (much faster)
fprintf('Finding nearest neighbor indices...\n');
rho_idx = round((RHO_grid - rho_min) / (rho_max - rho_min) * (N_r - 1)) + 1;
rho_idx = max(1, min(N_r, rho_idx)); % Clamp to valid range

theta_idx = round(THETA_grid / (2*pi) * (N_theta - 1)) + 1;
theta_idx = max(1, min(N_theta, theta_idx)); % Clamp to valid range

z_idx = round((Z_grid_sym - z_min) / (z_max - z_min) * (length(z_vec_full) - 1)) + 1;
z_idx = max(1, min(length(z_vec_full), z_idx)); % Clamp to valid range

% Perform fast nearest neighbor interpolation
fprintf('Performing fast nearest neighbor interpolation...\n');
total_points = N_x * N_y * N_z;
points_processed = 0;

for k = 1:N_z
    for j = 1:N_y
        for i = 1:N_x
            r_idx = rho_idx(i,j,k);
            t_idx = theta_idx(i,j,k);
            z_idx_val = z_idx(i,j,k);
            
            % Direct lookup from cylindrical grid
            Bx_cart(i,j,k) = Bx_decomp(r_idx, z_idx_val, t_idx);
            By_cart(i,j,k) = By_decomp(r_idx, z_idx_val, t_idx);
            Bz_cart(i,j,k) = Bz_decomp(r_idx, z_idx_val, t_idx);
            
            points_processed = points_processed + 1;
        end
    end
    
    % Update progress every 10 z-slices
    if mod(k, 10) == 0 || k == N_z
        progress = floor(100 * k / N_z);
        fprintf('\rInterpolation Progress: [%-20s] %d%%', repmat('=', 1, progress/5), progress);
    end
end
fprintf('\n');

fprintf('Fast interpolation completed.\n');
fprintf('Non-zero points: Bx: %d, By: %d, Bz: %d\n', ...
        nnz(Bx_cart), nnz(By_cart), nnz(Bz_cart));

%% Apply even symmetry efficiently
fprintf('Applying efficient even symmetry enforcement...\n');

% For nearest neighbor, we already used abs(Z_grid), so it's already symmetric
Bx_cart_sym = Bx_cart;
By_cart_sym = By_cart;
Bz_cart_sym = Bz_cart;

fprintf('Even symmetry already enforced via absolute z coordinates.\n');

%% Perform optimized 2D Fourier Transform in X and Z directions with progress bar
fprintf('Performing optimized 2D Fourier transform...\n');

% Precompute frequency vectors
dx = x_vec(2) - x_vec(1);
dz = z_vec(2) - z_vec(1);

% Frequency vectors (spatial frequencies) in rad/m
kx_vec = 2*pi * (-N_x/2:N_x/2-1) / (N_x * dx);
kz_vec = 2*pi * (-N_z/2:N_z/2-1) / (N_z * dz);
kx_vec = kx_vec(1:N_x);
kz_vec = kz_vec(1:N_z);

% Initialize Fourier transformed fields with correct dimensions
Bx_ft = zeros(N_x, N_y, N_z);
By_ft = zeros(N_x, N_y, N_z);
Bz_ft = zeros(N_x, N_y, N_z);

% Create progress bar for Fourier transform
fprintf('Fourier Transform Progress: [');
ft_progress_chars = 0;

% Use sequential processing for reliability
fprintf('] Processing %d y-slices sequentially...\n', N_y);

% Start timer for Fourier transform
ft_start_time = tic;

% Sequential processing with visual progress bar
for y_idx = 1:N_y
    % Extract 2D slice at current y - FIXED DIMENSION ISSUE
    Bx_slice = squeeze(Bx_cart_sym(:, y_idx, :));
    By_slice = squeeze(By_cart_sym(:, y_idx, :));
    Bz_slice = squeeze(Bz_cart_sym(:, y_idx, :));
    
    % Ensure proper 2D shape and no NaN
    Bx_slice = reshape(Bx_slice, [N_x, N_z]);
    By_slice = reshape(By_slice, [N_x, N_z]);
    Bz_slice = reshape(Bz_slice, [N_x, N_z]);
    
    Bx_slice(isnan(Bx_slice)) = 0;
    By_slice(isnan(By_slice)) = 0;
    Bz_slice(isnan(Bz_slice)) = 0;
    
    % Apply ifftshift and fft2 with proper dimension handling
    Bx_ft_slice = fftshift(fft2(ifftshift(Bx_slice)));
    By_ft_slice = fftshift(fft2(ifftshift(By_slice)));
    Bz_ft_slice = fftshift(fft2(ifftshift(Bz_slice)));
    
    % FIXED: Properly assign 2D slice to 3D array with correct dimensions
    Bx_ft(:, y_idx, :) = reshape(Bx_ft_slice, [N_x, 1, N_z]);
    By_ft(:, y_idx, :) = reshape(By_ft_slice, [N_x, 1, N_z]);
    Bz_ft(:, y_idx, :) = reshape(Bz_ft_slice, [N_x, 1, N_z]);
    
    % Update progress bar every 5 slices
    if mod(y_idx, 5) == 0 || y_idx == N_y
        progress = floor(100 * y_idx / N_y);
        fprintf('\rFourier Transform Progress: [%-20s] %d%%', repmat('=', 1, progress/5), progress);
        
        % Estimate remaining time
        elapsed_time = toc(ft_start_time);
        avg_time_per_slice = elapsed_time / y_idx;
        remaining_slices = N_y - y_idx;
        estimated_remaining = avg_time_per_slice * remaining_slices;
        
        if y_idx < N_y
            fprintf(' (ETA: %.1f sec)', estimated_remaining);
        else
            fprintf(' (Completed)');
        end
    end
end
fprintf('\n');

ft_total_time = toc(ft_start_time);
fprintf('Fourier transform completed in %.1f seconds.\n', ft_total_time);

% Compute magnitude spectra efficiently
fprintf('Computing magnitude spectra...\n');
Bx_ft_mag = abs(Bx_ft);
By_ft_mag = abs(By_ft);
Bz_ft_mag = abs(Bz_ft);
B_ft_total_mag = sqrt(Bx_ft_mag.^2 + By_ft_mag.^2 + Bz_ft_mag.^2);

% Clean up any remaining invalid values
Bx_ft_mag = max(Bx_ft_mag, 0);
By_ft_mag = max(By_ft_mag, 0);
Bz_ft_mag = max(Bz_ft_mag, 0);
B_ft_total_mag = max(B_ft_total_mag, 0);

fprintf('Fourier transform processing completed.\n');

%% Output Fourier transform results
fprintf('\n=== FOURIER TRANSFORM RESULTS ===\n');
fprintf('Spatial grid size: %d x %d x %d\n', N_x, N_y, N_z);
fprintf('Frequency ranges:\n');
fprintf('kx: [%.3f, %.3f] rad/m\n', min(kx_vec), max(kx_vec));
fprintf('kz: [%.3f, %.3f] rad/m\n', min(kz_vec), max(kz_vec));
fprintf('Fourier component ranges:\n');
fprintf('|FFT(Bx)|: [%.3e, %.3e]\n', min(Bx_ft_mag(:)), max(Bx_ft_mag(:)));
fprintf('|FFT(By)|: [%.3e, %.3e]\n', min(By_ft_mag(:)), max(By_ft_mag(:)));
fprintf('|FFT(Bz)|: [%.3e, %.3e]\n', min(Bz_ft_mag(:)), max(Bz_ft_mag(:)));
fprintf('|FFT(B_total)|: [%.3e, %.3e]\n', min(B_ft_total_mag(:)), max(B_ft_total_mag(:)));

%% Save optimized Fourier transform results
fprintf('\nSaving optimized Fourier transform results...\n');
fourier_data.Bx_ft = Bx_ft;
fourier_data.By_ft = By_ft;
fourier_data.Bz_ft = Bz_ft;
fourier_data.Bx_ft_mag = Bx_ft_mag;
fourier_data.By_ft_mag = By_ft_mag;
fourier_data.Bz_ft_mag = Bz_ft_mag;
fourier_data.B_ft_total_mag = B_ft_total_mag;
fourier_data.kx_vec = kx_vec;
fourier_data.kz_vec = kz_vec;
fourier_data.X_grid = X_grid;
fourier_data.Y_grid = Y_grid;
fourier_data.Z_grid = Z_grid;
fourier_data.x_vec = x_vec;
fourier_data.y_vec = y_vec;
fourier_data.z_vec = z_vec;
fourier_data.Bx_cart = Bx_cart;
fourier_data.By_cart = By_cart;
fourier_data.Bz_cart = Bz_cart;
fourier_data.Bx_cart_sym = Bx_cart_sym;
fourier_data.By_cart_sym = By_cart_sym;
fourier_data.Bz_cart_sym = Bz_cart_sym;
fourier_data.region_bounds = [x_min, x_max, y_min, y_max, z_min, z_max];
fourier_data.transform_directions = 'X and Z';
fourier_data.notes = 'Fast nearest neighbor interpolation with even symmetry enforcement';
fourier_data.computation_time_seconds = ft_total_time;

save('fourier_transform_field_components.mat', '-struct', 'fourier_data', '-v7.3');
fprintf('Fourier transform results saved to fourier_transform_field_components.mat\n');

%% Create optimized Fourier transform visualization
createOptimizedFourierPlots(X_grid, Y_grid, Z_grid, Bx_cart_sym, By_cart_sym, Bz_cart_sym, ...
                           Bx_ft_mag, By_ft_mag, Bz_ft_mag, B_ft_total_mag, ...
                           kx_vec, kz_vec, x_vec, y_vec, z_vec);

%% Compute the Fourier-space coefficients for the conductive plate model WITH NUMERICAL STABILITY
fprintf('\n=== COMPUTING FOURIER-SPACE COEFFICIENTS WITH NUMERICAL STABILITY ===\n');

% Parameters from the problem
h_r = 0.02;        % Height of rotor above conductive plate [m]
d_p = 0.005;       % Thickness of conductive plate [m]
v_x = 0.05;        % Velocity of rotor along x-axis [m/s]
omega_m = 5;       % Angular velocity of rotor [rad/s]
sigma = 3.8e7;     % Conductivity of plate [S/m]

% Calculate magnetic source frequency
omega_e = P * omega_m;  % From equation (5.5)

fprintf('Model Parameters:\n');
fprintf('Rotor height h_r: %.3f m\n', h_r);
fprintf('Plate thickness d_p: %.3f m\n', d_p);
fprintf('Rotor velocity v_x: %.3f m/s\n', v_x);
fprintf('Angular velocity ω_m: %.3f rad/s\n', omega_m);
fprintf('Magnetic frequency ω_e: %.3f rad/s\n', omega_e);
fprintf('Conductivity σ: %.3e S/m\n', sigma);

%% Extract Fourier-transformed source field components at boundary y = -h_r
fprintf('\nExtracting Fourier-transformed source field at boundary y = -h_r...\n');

% Find the y-index closest to -h_r
[~, y_boundary_idx] = min(abs(y_vec - (-h_r)));
fprintf('Using y-slice %d (y = %.3f m) as boundary\n', y_boundary_idx, y_vec(y_boundary_idx));

% Extract the Fourier components at the boundary
Bx_ft_boundary = squeeze(Bx_ft(:, y_boundary_idx, :));
By_ft_boundary = squeeze(By_ft(:, y_boundary_idx, :));
Bz_ft_boundary = squeeze(Bz_ft(:, y_boundary_idx, :));

fprintf('Boundary field components extracted (size: %dx%d)\n', size(Bx_ft_boundary));

%% Initialize coefficient arrays
fprintf('Initializing coefficient arrays...\n');

alpha_x = zeros(size(Bx_ft_boundary));
beta_x = zeros(size(Bx_ft_boundary));
alpha_z = zeros(size(Bx_ft_boundary));
beta_z = zeros(size(Bx_ft_boundary));
upsilon = zeros(size(Bx_ft_boundary));
nu_val = zeros(size(Bx_ft_boundary)); % Using nu_val to avoid conflict with MATLAB function

%% Precompute frequently used terms with numerical stability
fprintf('Precomputing common terms with numerical stability...\n');

% Create 2D grids for kx and kz frequencies
[KX, KZ] = ndgrid(kx_vec, kz_vec);

% Compute k^2 = ξ^2 + ζ^2 (equation A4.1)
k_squared = KX.^2 + KZ.^2;
k = sqrt(k_squared);
k(k == 0) = 1e-12; % Avoid division by zero

% Compute γ^2 = ξ^2 + ζ^2 + iμ_0σ(ω_e - v_xξ) (equation A4.1)
gamma_squared = k_squared + 1i * mu0 * sigma * (omega_e - v_x * KX);
gamma = sqrt(gamma_squared);

% Numerical stability: Limit exponential arguments to prevent overflow
max_exp_arg = 50; % Maximum allowed argument for exponential functions

% Precompute exponential terms with bounds (equations A4.3-A4.4)
S_H = exp(min(k * h_r, max_exp_arg));      % Equation A4.3
T_H = exp(min(-k * h_r, max_exp_arg));     % Equation A4.3
S_D = exp(min(k * d_p, max_exp_arg));      % Equation A4.4
T_D = exp(min(-k * d_p, max_exp_arg));     % Equation A4.4

% Precompute denominator D with stability (equation A4.6)
D = 2 * k .* sinh(min(k * d_p, max_exp_arg));
D(abs(D) < 1e-20) = 1e-20; % Avoid division by zero

%% Compute coefficients υ and ν first with numerical stability (equations A4.12-A4.13)
fprintf('Computing υ and ν coefficients with numerical stability...\n');

% Use the complex Fourier transforms directly
Bx_ft_complex = Bx_ft_boundary;
By_ft_complex = By_ft_boundary;
Bz_ft_complex = Bz_ft_boundary;

% Compute υ (equation A4.12) with stability
denominator_uv = 2 * mu0 * k_squared;
denominator_uv(denominator_uv < 1e-20) = 1e-20; % Avoid division by zero

upsilon_numerator = -k .* By_ft_complex + 1i * KZ .* Bz_ft_complex + 1i * KX .* Bx_ft_complex;
upsilon = -T_H ./ denominator_uv .* upsilon_numerator;

% Compute ν (equation A4.13) with stability  
nu_numerator = -k .* By_ft_complex + 1i * KZ .* Bz_ft_complex + 1i * KX .* Bx_ft_complex;
nu_val = S_H ./ denominator_uv .* nu_numerator;

% Apply numerical bounds to prevent extreme values
max_coef_value = 1e10;
upsilon = complex(real(upsilon), imag(upsilon));
nu_val = complex(real(nu_val), imag(nu_val));
upsilon(abs(upsilon) > max_coef_value) = max_coef_value;
nu_val(abs(nu_val) > max_coef_value) = max_coef_value;

% Clean up any invalid values
upsilon(isnan(upsilon) | isinf(upsilon)) = 0;
nu_val(isnan(nu_val) | isinf(nu_val)) = 0;

fprintf('υ and ν coefficients computed with stability bounds.\n');

%% Compute U and V with stability (equations A4.5)
fprintf('Computing U and V coefficients with stability...\n');

U = upsilon .* S_H;
V = nu_val .* T_D .* T_H;

% Apply bounds
U(abs(U) > max_coef_value) = max_coef_value;
V(abs(V) > max_coef_value) = max_coef_value;

%% Compute α and β coefficients with enhanced numerical stability (equations A4.7-A4.11)
fprintf('Computing α and β coefficients with enhanced numerical stability...\n');

% Compute α_x (equation A4.7)
alpha_x_numerator = S_D .* (-Bz_ft_complex + 1i * mu0 * KZ .* U) - 1i * mu0 * KZ .* V;
alpha_x = alpha_x_numerator ./ (T_H .* D);

% Compute β_x (equation A4.8)  
beta_x_numerator = T_D .* (-Bz_ft_complex + 1i * mu0 * KZ .* U) - 1i * mu0 * KZ .* V;
beta_x = beta_x_numerator ./ (S_H .* D);

% α_y = β_y = 0 (equation A4.9) - no computation needed

% Compute α_z (equation A4.10)
alpha_z_numerator = S_D .* (Bx_ft_complex - 1i * mu0 * KX .* U) + 1i * mu0 * KX .* V;
alpha_z = alpha_z_numerator ./ (T_H .* D);

% Compute β_z (equation A4.11)
beta_z_numerator = T_D .* (Bx_ft_complex - 1i * mu0 * KX .* U) + 1i * mu0 * KX .* V;
beta_z = beta_z_numerator ./ (S_H .* D);

% Apply numerical bounds to all coefficients
coefficient_sets = {alpha_x, beta_x, alpha_z, beta_z};
for i = 1:length(coefficient_sets)
    coef = coefficient_sets{i};
    % Bound real and imaginary parts separately
    coef_real = real(coef);
    coef_imag = imag(coef);
    coef_real(coef_real > max_coef_value) = max_coef_value;
    coef_real(coef_real < -max_coef_value) = -max_coef_value;
    coef_imag(coef_imag > max_coef_value) = max_coef_value;
    coef_imag(coef_imag < -max_coef_value) = -max_coef_value;
    coefficient_sets{i} = complex(coef_real, coef_imag);
end
[alpha_x, beta_x, alpha_z, beta_z] = coefficient_sets{:};

% Final cleanup of any remaining invalid values
coefficients_to_clean = {alpha_x, beta_x, alpha_z, beta_z, upsilon, nu_val, U, V};
for i = 1:length(coefficients_to_clean)
    coef = coefficients_to_clean{i};
    coef(isnan(coef) | isinf(coef)) = 0;
    coefficients_to_clean{i} = coef;
end
[alpha_x, beta_x, alpha_z, beta_z, upsilon, nu_val, U, V] = coefficients_to_clean{:};

fprintf('All coefficients computed with numerical stability.\n');

%% Analyze and display coefficient statistics
fprintf('\n=== COEFFICIENT STATISTICS (WITH STABILITY) ===\n');

coefficient_names = {'α_x', 'β_x', 'α_z', 'β_z', 'υ', 'ν'};
coefficient_arrays = {alpha_x, beta_x, alpha_z, beta_z, upsilon, nu_val};

for i = 1:length(coefficient_names)
    coef = coefficient_arrays{i};
    coef_mag = abs(coef);
    coef_phase = angle(coef);
    
    fprintf('\n%s:\n', coefficient_names{i});
    fprintf('  Magnitude: [%.3e, %.3e]\n', min(coef_mag(:)), max(coef_mag(:)));
    fprintf('  Phase: [%.3f, %.3f] rad\n', min(coef_phase(:)), max(coef_phase(:)));
    fprintf('  RMS: %.3e\n', sqrt(mean(coef_mag(:).^2)));
    fprintf('  Non-zero points: %d/%d (%.1f%%)\n', ...
            nnz(coef_mag > 1e-10), numel(coef), 100*nnz(coef_mag > 1e-10)/numel(coef));
    
    % Check for problematic values
    if max(coef_mag(:)) > 1e15
        fprintf('  ⚠ WARNING: Extremely large values detected\n');
    elseif max(coef_mag(:)) < 1e-20
        fprintf('  ⚠ WARNING: Extremely small values detected\n');
    end
end

%% Save coefficient results with stability information
fprintf('\nSaving stable coefficient results...\n');

coefficient_data.alpha_x = alpha_x;
coefficient_data.beta_x = beta_x;
coefficient_data.alpha_z = alpha_z;
coefficient_data.beta_z = beta_z;
coefficient_data.upsilon = upsilon;
coefficient_data.nu_val = nu_val;
coefficient_data.U = U;
coefficient_data.V = V;
coefficient_data.k = k;
coefficient_data.gamma = gamma;
coefficient_data.KX = KX;
coefficient_data.KZ = KZ;
coefficient_data.kx_vec = kx_vec;
coefficient_data.kz_vec = kz_vec;
coefficient_data.Bx_ft_boundary = Bx_ft_boundary;
coefficient_data.By_ft_boundary = By_ft_boundary;
coefficient_data.Bz_ft_boundary = Bz_ft_boundary;
coefficient_data.model_parameters.h_r = h_r;
coefficient_data.model_parameters.d_p = d_p;
coefficient_data.model_parameters.v_x = v_x;
coefficient_data.model_parameters.omega_m = omega_m;
coefficient_data.model_parameters.omega_e = omega_e;
coefficient_data.model_parameters.sigma = sigma;
coefficient_data.model_parameters.P = P;
coefficient_data.model_parameters.mu0 = mu0;
coefficient_data.stability_parameters.max_exp_arg = max_exp_arg;
coefficient_data.stability_parameters.max_coef_value = max_coef_value;

save('fourier_coefficients_results_stable.mat', '-struct', 'coefficient_data', '-v7.3');
fprintf('Stable coefficient results saved to fourier_coefficients_results_stable.mat\n');

%% Create coefficient visualization plots
fprintf('Creating coefficient visualization plots...\n');
createStableCoefficientPlots(KX, KZ, alpha_x, beta_x, alpha_z, beta_z, upsilon, nu_val, kx_vec, kz_vec);

fprintf('\n=== STABLE COEFFICIENT COMPUTATION COMPLETED ===\n');
fprintf('All Fourier-space coefficients have been computed with numerical stability.\n');

%% Enhanced Helper Functions with Stability

function createFieldComponentsPlots(RR, ZZ, THETA, Bx, By, Bz, B_mag, Ri, Ro, Wr, P, eigenvalues, coefficients)
    % Create efficient visualization plots for field components
    
    font_size = 11;
    
    % Extract only necessary 2D slices for visualization
    theta0_idx = 1; % theta = 0
    
    % Create 2D slices efficiently
    RR_2D_0 = RR(:,:,theta0_idx);
    ZZ_2D_0 = ZZ(:,:,theta0_idx);
    Bx_2D_0 = Bx(:,:,theta0_idx);
    By_2D_0 = By(:,:,theta0_idx);
    Bz_2D_0 = Bz(:,:,theta0_idx);
    B_mag_2D_0 = B_mag(:,:,theta0_idx);
    
    % Check for constant data that causes contour warnings
    if range(Bx_2D_0(:)) < 1e-10
        fprintf('Warning: Bx component is nearly constant, using imagesc instead of contourf\n');
        use_contour = false;
    else
        use_contour = true;
    end
    
    % Create optimized figure with subplots
    fig1 = figure('Position', [50, 50, 1400, 1000]);
    sgtitle('Decomposed Field Components at \theta = 0° (Region IV Only)', 'FontSize', font_size+2, 'FontWeight', 'bold');
    
    % Use efficient plotting with optimized contour levels
    contour_levels = 20;  % Reduced for efficiency
    
    if use_contour
        % Bx component
        subplot(2,3,1);
        [~, h1] = contourf(RR_2D_0, ZZ_2D_0, Bx_2D_0, contour_levels, 'LineColor', 'none');
        colorbar; addROI(Ri, Ro, Wr, RR_2D_0);
        title('B_x Component [T]', 'FontSize', font_size);
        
        % By component  
        subplot(2,3,2);
        [~, h2] = contourf(RR_2D_0, ZZ_2D_0, By_2D_0, contour_levels, 'LineColor', 'none');
        colorbar; addROI(Ri, Ro, Wr, RR_2D_0);
        title('B_y Component [T]', 'FontSize', font_size);
        
        % Bz component
        subplot(2,3,3);
        [~, h3] = contourf(RR_2D_0, ZZ_2D_0, Bz_2D_0, contour_levels, 'LineColor', 'none');
        colorbar; addROI(Ri, Ro, Wr, RR_2D_0);
        title('B_z Component [T]', 'FontSize', font_size);
        
        % Field magnitude
        subplot(2,3,4);
        [~, h4] = contourf(RR_2D_0, ZZ_2D_0, B_mag_2D_0, contour_levels, 'LineColor', 'none');
        colorbar; addROI(Ri, Ro, Wr, RR_2D_0);
        title('|B| Magnitude [T]', 'FontSize', font_size);
    else
        % Use imagesc for constant or near-constant data
        subplot(2,3,1);
        imagesc(RR_2D_0(:,1), ZZ_2D_0(1,:), Bx_2D_0');
        axis equal tight; colorbar; addROI(Ri, Ro, Wr, RR_2D_0);
        title('B_x Component [T]', 'FontSize', font_size);
        
        subplot(2,3,2);
        imagesc(RR_2D_0(:,1), ZZ_2D_0(1,:), By_2D_0');
        axis equal tight; colorbar; addROI(Ri, Ro, Wr, RR_2D_0);
        title('B_y Component [T]', 'FontSize', font_size);
        
        subplot(2,3,3);
        imagesc(RR_2D_0(:,1), ZZ_2D_0(1,:), Bz_2D_0');
        axis equal tight; colorbar; addROI(Ri, Ro, Wr, RR_2D_0);
        title('B_z Component [T]', 'FontSize', font_size);
        
        subplot(2,3,4);
        imagesc(RR_2D_0(:,1), ZZ_2D_0(1,:), B_mag_2D_0');
        axis equal tight; colorbar; addROI(Ri, Ro, Wr, RR_2D_0);
        title('|B| Magnitude [T]', 'FontSize', font_size);
    end
    
    % Vector field (common to both)
    subplot(2,3,5);
    skip = 6;  % Increased skip for efficiency
    quiver(RR_2D_0(1:skip:end, 1:skip:end), ZZ_2D_0(1:skip:end, 1:skip:end), ...
           Bx_2D_0(1:skip:end, 1:skip:end), Bz_2D_0(1:skip:end, 1:skip:end), 2, 'b');
    addROI(Ri, Ro, Wr, RR_2D_0);
    title('Field Vectors (B_x, B_z)', 'FontSize', font_size);
    
    % Optimized statistics display
    subplot(2,3,6);
    stats_text = getFieldStatsText(Bx_2D_0, By_2D_0, Bz_2D_0, B_mag_2D_0, RR_2D_0, P, eigenvalues);
    text(0.1, 0.5, stats_text, 'FontSize', font_size-1, 'VerticalAlignment', 'top');
    axis off;
    
    saveas(fig1, 'field_components_region_IV.png');
    
    % Create efficient coefficient plots
    createEfficientCoefficientPlots(coefficients, eigenvalues);
    
    fprintf('\n=== FIELD COMPONENTS SUMMARY (REGION IV ONLY) ===\n');
    fprintf('All field components computed and saved for Region IV (r >= Ro).\n');
    fprintf('3D grid size: %d (rho) x %d (z) x %d (theta)\n', size(RR,1), size(RR,2), size(RR,3));
    fprintf('Data saved to: decomposed_field_components_region_IV.mat\n');
end

function addROI(Ri, Ro, Wr, RR_2D)
    % Helper function to add region of interest to plots
    hold on;
    rectangle('Position', [Ri, -Wr/2, Ro-Ri, Wr], 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '--');
    plot([min(RR_2D(:)) max(RR_2D(:))], [0 0], 'k--', 'LineWidth', 1);
    axis equal tight;
    xlabel('\rho [m]'); ylabel('z [m]');
    grid on;
end

function stats_text = getFieldStatsText(Bx, By, Bz, B_mag, RR, P, eigenvalues)
    % Efficient statistics text generation
    stats_text = sprintf(['Field Component Statistics\n' ...
                         'at \\theta = 0° (Region IV):\n' ...
                         'B_x range: [%.3e, %.3e] T\n' ...
                         'B_y range: [%.3e, %.3e] T\n' ...
                         'B_z range: [%.3e, %.3e] T\n' ...
                         '|B| range: [%.3e, %.3e] T\n' ...
                         'Radial range: [%.3f, %.3f] m\n' ...
                         'Pole pairs: %d\n' ...
                         'Number of terms: %d'], ...
                         min(Bx(:)), max(Bx(:)), ...
                         min(By(:)), max(By(:)), ...
                         min(Bz(:)), max(Bz(:)), ...
                         min(B_mag(:)), max(B_mag(:)), ...
                         min(RR(:)), max(RR(:)), ...
                         P, length(eigenvalues));
end

function createEfficientCoefficientPlots(coefficients, eigenvalues)
    % Efficient coefficient visualization
    fig2 = figure('Position', [100, 100, 1000, 600]);
    
    subplot(1,2,1);
    stem(1:length(coefficients), abs(coefficients), 'filled', 'LineWidth', 1.5);
    set(gca, 'YScale', 'log');
    grid on;
    title('Decomposition Coefficient Magnitudes');
    xlabel('Term Index'); ylabel('|C(n)|');
    
    subplot(1,2,2);
    scatter(eigenvalues, abs(coefficients), 30, 'filled');
    set(gca, 'XScale', 'log', 'YScale', 'log');
    grid on;
    title('Coefficients vs Wave Number');
    xlabel('k_n [1/m]'); ylabel('|C(n)|');
    
    saveas(fig2, 'coefficient_analysis.png');
end

function createOptimizedFourierPlots(X_grid, Y_grid, Z_grid, Bx_cart, By_cart, Bz_cart, ...
                                    Bx_ft_mag, By_ft_mag, Bz_ft_mag, B_ft_total_mag, ...
                                    kx_vec, kz_vec, x_vec, y_vec, z_vec)
    % Create efficient Fourier transform visualization
    
    font_size = 11;
    
    % Find central slices for visualization
    y_mid_idx = round(length(y_vec)/2);
    
    % Extract only necessary 2D slices
    Bx_spatial = squeeze(Bx_cart(:, y_mid_idx, :));
    By_spatial = squeeze(By_cart(:, y_mid_idx, :));
    Bz_spatial = squeeze(Bz_cart(:, y_mid_idx, :));
    B_total_spatial = sqrt(Bx_spatial.^2 + By_spatial.^2 + Bz_spatial.^2);
    
    Bx_ft_slice = squeeze(Bx_ft_mag(:, y_mid_idx, :));
    By_ft_slice = squeeze(By_ft_mag(:, y_mid_idx, :));
    Bz_ft_slice = squeeze(Bz_ft_mag(:, y_mid_idx, :));
    B_total_ft_slice = squeeze(B_ft_total_mag(:, y_mid_idx, :));
    
    % Create optimized spatial domain plot
    createSpatialDomainPlot(x_vec, z_vec, Bx_spatial, By_spatial, Bz_spatial, B_total_spatial, ...
                           Bx_cart, By_cart, Bz_cart, X_grid);
    
    % Create optimized frequency domain plot
    createFrequencyDomainPlot(kx_vec, kz_vec, Bx_ft_slice, By_ft_slice, Bz_ft_slice, B_total_ft_slice, ...
                             Bx_ft_mag, By_ft_mag, Bz_ft_mag, B_ft_total_mag);
    
    % Create efficient 1D spectra
    create1DSpectra(kx_vec, kz_vec, Bx_ft_slice, By_ft_slice, Bz_ft_slice, Bx_ft_mag, Bz_ft_mag);
    
    fprintf('\nOptimized Fourier transform plots saved.\n');
end

function createSpatialDomainPlot(x_vec, z_vec, Bx, By, Bz, B_total, Bx_cart, By_cart, Bz_cart, X_grid)
    fig1 = figure('Position', [50, 50, 1400, 1000]);
    sgtitle('Spatial Domain Field Components (Cartesian, Y=0 plane) - Even Symmetry', 'FontSize', 13, 'FontWeight', 'bold');
    
    subplot(2,3,1);
    imagesc(x_vec, z_vec, Bx');
    axis equal tight; colorbar;
    title('B_x Spatial Domain [T]', 'FontSize', 11);
    xlabel('x [m]'); ylabel('z [m]');
    
    subplot(2,3,2);
    imagesc(x_vec, z_vec, By');
    axis equal tight; colorbar;
    title('B_y Spatial Domain [T]', 'FontSize', 11);
    xlabel('x [m]'); ylabel('z [m]');
    
    subplot(2,3,3);
    imagesc(x_vec, z_vec, Bz');
    axis equal tight; colorbar;
    title('B_z Spatial Domain [T]', 'FontSize', 11);
    xlabel('x [m]'); ylabel('z [m]');
    
    subplot(2,3,4);
    imagesc(x_vec, z_vec, B_total');
    axis equal tight; colorbar;
    title('|B| Spatial Domain [T]', 'FontSize', 11);
    xlabel('x [m]'); ylabel('z [m]');
    
    % Efficient statistics
    subplot(2,3,5);
    stats_text = sprintf(['Spatial Domain Statistics\n' ...
                         'B_x: [%.3e, %.3e] T\n' ...
                         'B_y: [%.3e, %.3e] T\n' ...
                         'B_z: [%.3e, %.3e] T\n' ...
                         '|B|: [%.3e, %.3e] T\n' ...
                         'Grid: %d x %d x %d'], ...
                         min(Bx_cart(:)), max(Bx_cart(:)), ...
                         min(By_cart(:)), max(By_cart(:)), ...
                         min(Bz_cart(:)), max(Bz_cart(:)), ...
                         min(B_total(:)), max(B_total(:)), ...
                         size(X_grid,1), size(X_grid,2), size(X_grid,3));
    text(0.1, 0.5, stats_text, 'FontSize', 10, 'VerticalAlignment', 'top');
    axis off;
    
    saveas(fig1, 'spatial_domain_fields.png');
end

function createFrequencyDomainPlot(kx_vec, kz_vec, Bx_ft, By_ft, Bz_ft, B_total_ft, ...
                                  Bx_ft_mag, By_ft_mag, Bz_ft_mag, B_ft_total_mag)
    fig2 = figure('Position', [100, 100, 1400, 1000]);
    sgtitle('Fourier Transform Magnitude (X-Z Transform, Y=0 plane)', 'FontSize', 13, 'FontWeight', 'bold');
    
    % Use efficient log scaling
    Bx_ft_log = log10(Bx_ft' + 1e-20);
    By_ft_log = log10(By_ft' + 1e-20);
    Bz_ft_log = log10(Bz_ft' + 1e-20);
    B_total_ft_log = log10(B_total_ft' + 1e-20);
    
    subplot(2,3,1);
    imagesc(kx_vec, kz_vec, Bx_ft_log);
    axis equal tight; colorbar;
    title('log|FFT(B_x)|', 'FontSize', 11);
    xlabel('k_x [rad/m]'); ylabel('k_z [rad/m]');
    
    subplot(2,3,2);
    imagesc(kx_vec, kz_vec, By_ft_log);
    axis equal tight; colorbar;
    title('log|FFT(B_y)|', 'FontSize', 11);
    xlabel('k_x [rad/m]'); ylabel('k_z [rad/m]');
    
    subplot(2,3,3);
    imagesc(kx_vec, kz_vec, Bz_ft_log);
    axis equal tight; colorbar;
    title('log|FFT(B_z)|', 'FontSize', 11);
    xlabel('k_x [rad/m]'); ylabel('k_z [rad/m]');
    
    subplot(2,3,4);
    imagesc(kx_vec, kz_vec, B_total_ft_log);
    axis equal tight; colorbar;
    title('log|FFT(B_{total})|', 'FontSize', 11);
    xlabel('k_x [rad/m]'); ylabel('k_z [rad/m]');
    
    % Efficient frequency statistics
    subplot(2,3,5);
    ft_stats_text = sprintf(['Fourier Transform Statistics\n' ...
                            '|FFT(B_x)|: [%.3e, %.3e]\n' ...
                            '|FFT(B_y)|: [%.3e, %.3e]\n' ...
                            '|FFT(B_z)|: [%.3e, %.3e]\n' ...
                            '|FFT(B_{total})|: [%.3e, %.3e]\n' ...
                            'k_x range: [%.1f, %.1f] rad/m\n' ...
                            'k_z range: [%.1f, %.1f] rad/m'], ...
                            min(Bx_ft_mag(:)), max(Bx_ft_mag(:)), ...
                            min(By_ft_mag(:)), max(By_ft_mag(:)), ...
                            min(Bz_ft_mag(:)), max(Bz_ft_mag(:)), ...
                            min(B_ft_total_mag(:)), max(B_ft_total_mag(:)), ...
                            min(kx_vec), max(kx_vec), min(kz_vec), max(kz_vec));
    text(0.1, 0.5, ft_stats_text, 'FontSize', 10, 'VerticalAlignment', 'top');
    axis off;
    
    saveas(fig2, 'fourier_transform_fields.png');
end

function create1DSpectra(kx_vec, kz_vec, Bx_ft_slice, By_ft_slice, Bz_ft_slice, Bx_ft_mag, Bz_ft_mag)
    fig3 = figure('Position', [150, 150, 1200, 800]);
    
    % Extract central frequency cuts efficiently
    kx_center = round(length(kx_vec)/2);
    kz_center = round(length(kz_vec)/2);
    
    Bx_kx = Bx_ft_slice(:, kz_center);
    By_kx = By_ft_slice(:, kz_center);
    Bz_kx = Bz_ft_slice(:, kz_center);
    
    Bx_kz = Bx_ft_slice(kx_center, :);
    By_kz = By_ft_slice(kx_center, :);
    Bz_kz = Bz_ft_slice(kx_center, :);
    
    subplot(2,2,1);
    semilogy(kx_vec, Bx_kx, 'r-', 'LineWidth', 2); hold on;
    semilogy(kx_vec, By_kx, 'g-', 'LineWidth', 2);
    semilogy(kx_vec, Bz_kx, 'b-', 'LineWidth', 2);
    grid on; legend('B_x', 'B_y', 'B_z', 'Location', 'best');
    title('1D Fourier Spectrum vs k_x (k_z = 0)');
    xlabel('k_x [rad/m]'); ylabel('|FFT| Magnitude');
    
    subplot(2,2,2);
    semilogy(kz_vec, Bx_kz, 'r-', 'LineWidth', 2); hold on;
    semilogy(kz_vec, By_kz, 'g-', 'LineWidth', 2);
    semilogy(kz_vec, Bz_kz, 'b-', 'LineWidth', 2);
    grid on; legend('B_x', 'B_y', 'B_z', 'Location', 'best');
    title('1D Fourier Spectrum vs k_z (k_x = 0)');
    xlabel('k_z [rad/m]'); ylabel('|FFT| Magnitude');
    
    % Efficient dominant frequency calculation
    [~, Bx_max_idx] = max(Bx_ft_mag(:));
    [Bx_i, ~, Bx_k] = ind2sub(size(Bx_ft_mag), Bx_max_idx);
    
    [~, Bz_max_idx] = max(Bz_ft_mag(:));
    [Bz_i, ~, Bz_k] = ind2sub(size(Bz_ft_mag), Bz_max_idx);
    
    subplot(2,2,3);
    dominant_text = sprintf(['Dominant Spatial Frequencies:\n\n' ...
                            'B_x max at:\n' ...
                            'k_x = %.2f rad/m\n' ...
                            'k_z = %.2f rad/m\n\n' ...
                            'B_z max at:\n' ...
                            'k_x = %.2f rad/m\n' ...
                            'k_z = %.2f rad/m'], ...
                            kx_vec(Bx_i), kz_vec(Bx_k), ...
                            kx_vec(Bz_i), kz_vec(Bz_k));
    text(0.1, 0.5, dominant_text, 'FontSize', 11, 'VerticalAlignment', 'middle');
    axis off;
    
    saveas(fig3, 'frequency_spectra.png');
end

function createStableCoefficientPlots(KX, KZ, alpha_x, beta_x, alpha_z, beta_z, upsilon, nu_val, kx_vec, kz_vec)
    % Create visualization plots for the computed coefficients with stability checks
    
    font_size = 11;
    
    % Create figure for coefficient magnitudes
    fig1 = figure('Position', [50, 50, 1400, 1000]);
    sgtitle('Stable Fourier-Space Coefficient Magnitudes', 'FontSize', font_size+2, 'FontWeight', 'bold');
    
    coefficients = {alpha_x, beta_x, alpha_z, beta_z, upsilon, nu_val};
    titles = {'|α_x|', '|β_x|', '|α_z|', '|β_z|', '|υ|', '|ν|'};
    
    for i = 1:6
        subplot(2, 3, i);
        coef_data = coefficients{i};
        coef_mag = abs(coef_data);
        
        % Use log scale but handle zeros properly
        coef_log = log10(coef_mag + 1e-20);
        
        imagesc(kx_vec, kz_vec, coef_log');
        colorbar;
        title(titles{i}, 'FontSize', font_size);
        xlabel('k_x [rad/m]'); ylabel('k_z [rad/m]');
        axis equal tight;
    end
    
    saveas(fig1, 'stable_coefficient_magnitudes.png');
    
    % Create figure for coefficient quality assessment
    fig2 = figure('Position', [100, 100, 1200, 800]);
    sgtitle('Coefficient Quality Assessment', 'FontSize', font_size+2, 'FontWeight', 'bold');
    
    % Check coefficient validity
    subplot(2,3,1);
    valid_mask = isfinite(alpha_x) & ~isnan(alpha_x) & (abs(alpha_x) > 0);
    imagesc(kx_vec, kz_vec, double(valid_mask)');
    colorbar; title('α_x Validity Mask'); xlabel('k_x'); ylabel('k_z');
    
    subplot(2,3,2);
    histogram(log10(abs(alpha_x(:)) + 1e-20), 50);
    title('α_x Magnitude Distribution'); xlabel('log10(|α_x|)'); ylabel('Count');
    grid on;
    
    subplot(2,3,3);
    histogram(angle(alpha_x(:)), 50);
    title('α_x Phase Distribution'); xlabel('Phase [rad]'); ylabel('Count');
    grid on;
    
    % Statistics
    subplot(2,3,4);
    stats_text = sprintf(['Stable Coefficient Statistics:\n\n' ...
                         'α_x: [%.3e, %.3e]\n' ...
                         'β_x: [%.3e, %.3e]\n' ...
                         'α_z: [%.3e, %.3e]\n' ...
                         'β_z: [%.3e, %.3e]\n' ...
                         'υ: [%.3e, %.3e]\n' ...
                         'ν: [%.3e, %.3e]\n\n' ...
                         'Valid points: %.1f%%'], ...
                         min(abs(alpha_x(:))), max(abs(alpha_x(:))), ...
                         min(abs(beta_x(:))), max(abs(beta_x(:))), ...
                         min(abs(alpha_z(:))), max(abs(alpha_z(:))), ...
                         min(abs(beta_z(:))), max(abs(beta_z(:))), ...
                         min(abs(upsilon(:))), max(abs(upsilon(:))), ...
                         min(abs(nu_val(:))), max(abs(nu_val(:))), ...
                         100 * nnz(valid_mask) / numel(valid_mask));
    text(0.1, 0.5, stats_text, 'FontSize', font_size-1, 'VerticalAlignment', 'middle');
    axis off;
    
    % Dominant frequencies
    subplot(2,3,5);
    [~, alpha_max_idx] = max(abs(alpha_x(:)));
    [alpha_i, alpha_j] = ind2sub(size(alpha_x), alpha_max_idx);
    
    dominant_text = sprintf(['Dominant Frequencies:\n\n' ...
                            'α_x max at:\n' ...
                            'k_x = %.1f rad/m\n' ...
                            'k_z = %.1f rad/m\n\n' ...
                            'Max values:\n' ...
                            '|α_x| = %.3e\n' ...
                            '|β_x| = %.3e'], ...
                            kx_vec(alpha_i), kz_vec(alpha_j), ...
                            max(abs(alpha_x(:))), max(abs(beta_x(:))));
    text(0.1, 0.5, dominant_text, 'FontSize', font_size-1, 'VerticalAlignment', 'middle');
    axis off;
    
    saveas(fig2, 'coefficient_quality_assessment.png');
    
    fprintf('Stable coefficient visualization plots saved.\n');
end

fprintf('\n=== FULL STABLE COMPUTATION COMPLETED ===\n');
fprintf('All computations finished successfully with numerical stability!\n');
fprintf('Output files created:\n');
fprintf('  - decomposed_field_components_region_IV.mat\n');
fprintf('  - fourier_transform_field_components.mat\n');
fprintf('  - fourier_coefficients_results_stable.mat\n');
fprintf('  - Various visualization plots (.png)\n');