clear; clc; close all;

%  1D CONVERGING-DIVERGING NOZZLE SOLVER
%  Quasi-1D Euler Equations — SSP-RK2 Time Integration
%  Rusanov (Local Lax-Friedrichs) Flux
%  Working third order
%  only retro-fitted this old code for subsonic transonic and limiter may
%  not work - its a little finicky

%% USER SETTINGS
flow = 2; % 1 = subsonic, 2 = transonic (with shock)
order = 3; % 1 = first order, 2 = second order
limiter = 2; % 1 = no limiter,  2 = limiter on
CFL = 1.0; % standard cfl of 1 should be stable
tolerance = 1e-6;

max_iterations = 100000;

np_values = [100, 200, 400, 800, 1600, 3200];  % mesh sizes for convergence study
xmin = -4;
xmax = 4;

spectral = 1; % later knob

%% CONSTANTS
gam = 1.4;
gamm1 = gam - 1;
gamp1 = gam + 1;

%% INITIALIZE FLOW (sets globals: p0, rho0, astar, and shock params if flow=2)
Initialize(flow);

%% EXACT SOLUTION SETUP (subsonic exit conditions)
area_e = area(5.0);

mach_e = 0.4;
rho_e = 1.0;
p_e = 1.0/gam;

temp = 1 + 0.5*gamm1*mach_e^2;

p0 = p_e * temp^(gam/gamm1);
rho0 = rho_e * temp^(1/gamm1);

temp = 1/mach_e^2 * (2/gamp1*(1 + 0.5*gamm1*mach_e^2))^(gamp1/gamm1);

astar = area_e / sqrt(temp);

%% PLOT COLORS (green shades, one per mesh size)
green_colors = [linspace(0.5, 0, length(np_values))', ...
                linspace(1.0, 0.5, length(np_values))', ...
                linspace(0.5, 0, length(np_values))'];
figure; hold on;

%% MESH CONVERGENCE LOOP
for idx = 1:length(np_values)

    np = np_values(idx);
    dx = (xmax - xmin) / (np - 1);
    x  = linspace(xmin, xmax, np);

    %% EXACT SOLUTION
    q_exact = zeros(3, np);
    for i = 1:np
        q_exact(:, i) = ExactSolu(x(i), flow);
    end

    %% INITIAL CONDITION — uniform freestream
    rho_guess = q_exact(1, 1);
    rhou_guess = q_exact(2, 1);
    E_guess = q_exact(3, 1);

    Q = [rho_guess * ones(1, np);
         rhou_guess * ones(1, np);
         E_guess * ones(1, np)];

    Q(:, 1) = q_exact(:, 1);
    Q(:, end) = q_exact(:, end);

    %% ITERATION LOOP — SSP-RK2
    residual = inf;
    iteration = 0;

    while residual > tolerance && iteration < max_iterations

        % Enforce exact boundary conditions every step
        Q(:, 1) = q_exact(:, 1);
        Q(:, end) = q_exact(:, end);
        Q(:, end-2:end) = q_exact(:, end-2:end);

        dt = computeDT(Q, CFL, dx, gam);

        % RK2 Step 1
        [qL, qR] = reconstruct(order, limiter, Q, q_exact);
        F = computeFluxes(qL, qR, gam, dx);
        S = computeSource(Q, gam, dx, x);
        Q_star = Q + dt * (-F + S);
        
        % RK2 Step 2
        [qL, qR] = reconstruct(order, limiter, Q_star, q_exact);
        F2 = computeFluxes(qL, qR, gam, dx);
        S2 = computeSource(Q_star, gam, dx, x);
        Q_new = 0.5 * (Q + Q_star + dt * (-F2 + S2));

        % Residual of middle quarter
        mid = (np/2 - np/4):(np/2 + np/4);
        residual = max(abs(S2(1, mid) - F2(1, mid)), [], 'all');

        Q_new(:, end-2:end) = q_exact(:, end-2:end);
        Q = Q_new;

        iteration = iteration + 1;
        if mod(iteration, 100) == 0
            disp(['np = ', num2str(np), ...
                ' iter = ', num2str(iteration), ...
                ' residual = ', num2str(residual)]);
        end
    end

    Q_all{idx} = Q;  

    %% DERIVED QUANTITIES FOR PLOTTING
    Q(4, :) = Q(2, :) ./ Q(1, :); % velocity
    q_exact(4, :) = q_exact(2, :) ./ q_exact(1, :);

    color = green_colors(idx, :);

    subplot(4,1,1); hold on;
    plot(x, Q(1,:), 'Color', color, 'DisplayName', ['np = ' num2str(np)]);
    xlabel('x'); ylabel('Mass'); title(['Flow ' num2str(flow) ' — Mass']);

    subplot(4,1,2); hold on;
    plot(x, Q(2,:), 'Color', color, 'DisplayName', ['np = ' num2str(np)]);
    xlabel('x'); ylabel('Momentum'); title(['Flow ' num2str(flow) ' — Momentum']);

    subplot(4,1,3); hold on;
    plot(x, Q(3,:), 'Color', color, 'DisplayName', ['np = ' num2str(np)]);
    xlabel('x'); ylabel('Energy'); title(['Flow ' num2str(flow) ' — Energy']);

    subplot(4,1,4); hold on;
    plot(x, Q(4,:), 'Color', color, 'DisplayName', ['np = ' num2str(np)]);
    xlabel('x'); ylabel('Velocity'); title(['Flow ' num2str(flow) ' — Velocity']);


end

%% ORDER OF ACCURACY SUMMARY
disp(' ')
disp(['flow=', num2str(flow), ' order=', num2str(order)])
disp('np L2 Error Order')

errors = zeros(1, length(np_values));
for idx = 1:length(np_values)
    np = np_values(idx);
    dx = (xmax - xmin) / (np - 1);
    x = linspace(xmin, xmax, np);
    q_exact_local = zeros(3, np);
    for i = 1:np
        q_exact_local(:,i) = ExactSolu(x(i), flow);
    end
    errors(idx) = sqrt(dx * sum((Q_all{idx}(1,:) - q_exact_local(1,:)).^2));
end

for idx = 1:length(np_values)
    if idx == 1
        disp([np_values(idx), errors(idx)])
    else
        p = log(errors(idx-1)/errors(idx)) / log(np_values(idx)/np_values(idx-1));
        disp([np_values(idx), errors(idx), p])
    end
end

%% OVERLAY EXACT SOLUTION
subplot(4,1,1); plot(x, q_exact(1,:), 'k-', 'DisplayName', 'Exact'); legend();
subplot(4,1,2); plot(x, q_exact(2,:), 'k-', 'DisplayName', 'Exact'); legend();
subplot(4,1,3); plot(x, q_exact(3,:), 'k-', 'DisplayName', 'Exact'); legend();
subplot(4,1,4); plot(x, q_exact(4,:), 'k-', 'DisplayName', 'Exact'); legend();



%%  FUNCTIONS


function dt = computeDT(Q, CFL, dx, gam)
    rho = Q(1,:);
    u = Q(2,:) ./ rho;
    E = Q(3,:);
    p = (gam - 1) * (E - 0.5 * rho .* u.^2);
    c = sqrt(gam * p ./ rho);
    dt = min(CFL * dx ./ (abs(u) + c));
end
function [qL, qR] = reconstruct(order, limiter, Q, q_exact)
    np = size(Q, 2);
    n_faces = np + 1;
    qL = zeros(3, n_faces);
    qR = zeros(3, n_faces);

    qL(:, 1) = q_exact(:, 1);
    qR(:, 1) = q_exact(:, 1);
    qL(:, n_faces) = q_exact(:, end);
    qR(:, n_faces) = q_exact(:, end);

    if order == 1
        for i = 2:np
            qL(:, i) = Q(:, i-1);
            qR(:, i) = Q(:, i);
        end

    elseif order == 2
        for i = 2:np
            if i == 2
                qL(:, i) = Q(:, i-1);
            else
                dm = Q(:, i-1) - Q(:, i-2);
                dp = Q(:, i)   - Q(:, i-1);
                qL(:, i) = Q(:, i-1) + 0.5 * muscl_slope(dm, dp, limiter);
            end
            if i == np
                qR(:, i) = Q(:, i);
            else
                dm = Q(:, i)   - Q(:, i-1);
                dp = Q(:, i+1) - Q(:, i);
                qR(:, i) = Q(:, i) - 0.5 * muscl_slope(dm, dp, limiter);
            end
        end

    elseif order == 3
        kappa = 1/3;

        for i = 2:np
            % LEFT state from cell i-1
            if i == 2
                qL(:, i) = Q(:, i-1);
            elseif i == 3
                dm = Q(:, i-1) - Q(:, i-2);
                dp = Q(:, i)   - Q(:, i-1);
                qL(:, i) = Q(:, i-1) + 0.5 * muscl_slope(dm, dp, limiter);
            else
                dm = Q(:, i-1) - Q(:, i-2);
                dp = Q(:, i)   - Q(:, i-1);
                slope_unlimited = 0.25*((1-kappa)*dm + (1+kappa)*dp);
                slope_limited   = muscl_slope(dm, dp, limiter);
                % if limiter==1, slope_limited == slope_unlimited
                % if limiter==2, clamp to minmod of dm,dp
                if limiter == 2
                    qL(:, i) = Q(:, i-1) + sign_clamp(slope_unlimited, slope_limited);
                else
                    qL(:, i) = Q(:, i-1) + slope_unlimited;
                end
            end

            % RIGHT state from cell i
            if i == np
                qR(:, i) = Q(:, i);
            elseif i == np-1
                dm = Q(:, i)   - Q(:, i-1);
                dp = Q(:, i+1) - Q(:, i);
                qR(:, i) = Q(:, i) - 0.5 * muscl_slope(dm, dp, limiter);
            else
                dm = Q(:, i)   - Q(:, i-1);
                dp = Q(:, i+1) - Q(:, i);
                slope_unlimited = 0.25*((1+kappa)*dm + (1-kappa)*dp);
                slope_limited   = muscl_slope(dm, dp, limiter);
                if limiter == 2
                    qR(:, i) = Q(:, i) - sign_clamp(slope_unlimited, slope_limited);
                else
                    qR(:, i) = Q(:, i) - slope_unlimited;
                end
            end
        end
    end
end

function out = sign_clamp(a, b)
    out = zeros(size(a));
    for k = 1:length(a)
        if b(k) == 0 || a(k) * b(k) <= 0
            out(k) = 0; 
        else
            out(k) = sign(a(k)) * min(abs(a(k)), abs(b(k)));
        end
    end
end

function slope = muscl_slope(dm, dp, limiter)
    if limiter == 1
        slope = 0.5 * (dm + dp);
    elseif limiter == 2
        slope = zeros(size(dm));
        for k = 1:length(dm)
            if dm(k) * dp(k) > 0
                slope(k) = sign(dm(k)) * min(abs(dm(k)), abs(dp(k)));
            end
        end
    end
end

function source = computeSource(Q, gam, dx, x)
    np = size(Q, 2);
    source = zeros(size(Q));
    for i = 1:np
        rho = Q(1, i);
        u = Q(2, i) / rho;
        E = Q(3, i);
        p = (gam - 1) * (E - 0.5 * rho * u^2);
        A = area(x(i));
        dA_dx = darea(x(i));
        source(1, i) = -rho * u * dA_dx / A;
        source(2, i) = -rho * u^2 * dA_dx / A;
        source(3, i) = -u * (E + p) * dA_dx / A;
    end
end

function da = darea(x)
    if x > 0
        da =  0.198086 * 2 * log(2) * x / 0.6^2 * exp(-log(2) * (x/0.6)^2);
    else
        da =  0.661514 * 2 * log(2) * x / 0.6^2 * exp(-log(2) * (x/0.6)^2);
    end
end

function fluxes = computeFluxes(qL, qR, gam, dx)
    n_faces = size(qL, 2);
    np = n_faces - 1;
    F_face  = zeros(3, n_faces);

    % boundary fluxes
    F_face(:, 1) = rusanov(qL(:,1),   qR(:,1),   gam);
    F_face(:, end) = rusanov(qL(:,end), qR(:,end), gam);

    for i = 2:np % do at the interfaces first
        F_face(:, i) = rusanov(qL(:,i), qR(:,i), gam);
    end
    for i = 1:np % fluxes at cell centers
        fluxes(:, i) = (F_face(:, i+1) - F_face(:, i)) / dx;
    end
end

function flux = rusanov(qL, qR, gam)
    rhoL = qL(1); uL = qL(2)/rhoL; EL = qL(3);
    pL = (gam-1) * (EL - 0.5*rhoL*uL^2);
    cL = sqrt(gam * pL / rhoL);

    rhoR = qR(1); uR = qR(2)/rhoR; ER = qR(3);
    pR = (gam-1) * (ER - 0.5*rhoR*uR^2);
    cR = sqrt(gam * pR / rhoR);

    FL = [rhoL*uL; rhoL*uL^2 + pL; uL*(EL+pL)];
    FR = [rhoR*uR; rhoR*uR^2 + pR; uR*(ER+pR)];

    s_max = max(abs(uL)+cL, abs(uR)+cR);
    flux = 0.5*(FL + FR) + 0.5*s_max*(qL - qR);
end

%%  PROVIDED FUNCTIONS (from Dr. Wang)
function aa = area(x)
    if x > 0
        aa = 0.536572 - 0.198086 * exp(-log(2) * (x/0.6)^2);
    else
        aa = 1.0 - 0.661514 * exp(-log(2) * (x/0.6)^2);
    end
end
function q = ExactSolu(x, flow)
    global x_shock p0 rho0 astar p02 astar2
    gam = 1.4;
    gamm1 = gam - 1;
    gamp1 = gam + 1;
    q = zeros(3, 1);

    if flow == 1   % subsonic
        temp = (area(x) / astar)^2;
        mach = fzero(@(m) MachArea(m, temp), 0.05);
        temp = 1 + 0.5*gamm1*mach^2;
        rho  = rho0 / temp^(1/gamm1);
        p = p0   / temp^(gam/gamm1);
        c = sqrt(gam * p / rho);
        u = abs(mach) * c;
        q = [rho; rho*u; p/gamm1 + 0.5*rho*u^2];

    elseif flow == 2   % transonic with shock
        T0 = p0 / rho0;
        if x <= x_shock
            A = area(x);
            aas2 = (A / astar)^2;
            if aas2 < 1.01
                guess = (x >= 0) * 1.1 + (x < 0) * 0.99;
            else
                guess = (x > 0) * 2.0  + (x <= 0) * 0.1;
            end
            machs = fzero(@(m) MachArea(m, aas2), guess);
            temp  = 1 + 0.5*gamm1*machs^2;
            p1 = p0   / temp^(gam/gamm1);
            rho1 = rho0 / temp^(1/gamm1);
            c = sqrt(gam * p1 / rho1);
            u1 = abs(machs) * c;
        else
            aas2  = (area(x) / astar2)^2;
            mache = fzero(@(m) MachArea(m, aas2), 0.05);
            temp = 1 + 0.5*gamm1*mache^2;
            p1 = p02 / temp^(gam/gamm1);
            T1 = T0  / temp;
            rho1 = p1  / T1;
            c = sqrt(gam * T1);
            u1 = abs(mache) * c;
        end
        q = [rho1; rho1*u1; p1/gamm1 + 0.5*rho1*u1^2];
    end
end
function [] = Initialize(flow)
    global x_shock p0 rho0 astar p02 astar2
    gam = 1.4;
    gamm1 = gam - 1;
    gamp1 = gam + 1;

    if flow == 1   % subsonic
        area_e = area(5.0);
        mach_e = 0.4;
        rho_e = 1.0;
        p_e = 1.0 / gam;
        temp = 1 + 0.5*gamm1*mach_e^2;
        p0 = p_e   * temp^(gam/gamm1);
        rho0 = rho_e * temp^(1/gamm1);
        temp = 1/mach_e^2 * (2/gamp1*(1+0.5*gamm1*mach_e^2))^(gamp1/gamm1);
        astar = area_e / sqrt(temp);

    elseif flow == 2   % transonic — locate shock iteratively
        mach_i = 0.2006533;
        rho_i = 1.0;
        p_i = 1.0 / gam;
        p_e = 0.6071752;
        temp = 1 + 0.5*gamm1*mach_i^2;
        p0 = p_i   * temp^(gam/gamm1);
        rho0 = rho_i * temp^(1/gamm1);
        astar = area(0);   % throat

        x_shock = 0.5;
        x_shock_small = 0.0;
        x_shock_large = 1.0;
        pe = 0.0;

        while abs(p_e - pe) > 1e-5
            A = area(x_shock);
            aas2 = (A / astar)^2;
            machs = fzero(@(m) MachArea(m, aas2), 2.0);
            temp = 1 + 0.5*gamm1*machs^2;
            p1 = p0 / temp^(gam/gamm1);

            p2 = p1 * (1 + 2*gam/gamp1*(machs^2 - 1));
            mach2 = sqrt((1 + 0.5*gamm1*machs^2) / (gam*machs^2 - 0.5*gamm1));
            aas2 = 1/mach2^2 * (2/gamp1*(1+0.5*gamm1*mach2^2))^(gamp1/gamm1);
            astar2 = sqrt(A^2 / aas2);

            temp = 1 + 0.5*gamm1*mach2^2;
            p02 = p2 * temp^(gam/gamm1);
            aas2 = (area(4) / astar2)^2;
            mache = fzero(@(m) MachArea(m, aas2), 0.05);
            temp = 1 + 0.5*gamm1*mache^2;
            pe = p02 / temp^(gam/gamm1);

            if pe > p_e
                x_shock_small = x_shock;
                x_shock = 0.5 * (x_shock + x_shock_large);
            else
                x_shock_large = x_shock;
                x_shock = 0.5 * (x_shock + x_shock_small);
            end
        end
        disp(['Shock located at x = ' num2str(x_shock)]);
    end
end
function aas = MachArea(mach, aas2)
    gam = 1.4;
    gamp1 = gam + 1;
    gamm1 = gam - 1;
    aas = 1/mach^2 * (2/gamp1*(1+0.5*gamm1*mach^2))^(gamp1/gamm1) - aas2;
end