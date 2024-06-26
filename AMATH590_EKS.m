%% Ensemble Kalman Sampler for Low-Dimensional Parameter Space Example

% Goal: Find u1, u2 knowing y data

%% Set Up
% Problem Specific Information
x = [0.25; 0.75];                                       % Measurement Location
y = [27.5; 79.7];                                       % Observed Data
p = @(x,u1,u2) u2.*x + exp(-u1).*(-1/2*(x).^2 + x/2);   % Component functions for operator G
G = @(x1,x2,u1,u2) [p(x1,u1,u2); p(x2,u1,u2)];          % Define operator G as G(u,x)
Gx = @(u1,u2) G(x(1),x(2),u1,u2);                       % Evaluate G at input x point to get G(u)

J = [1000,1100,1200,1300,1400,1500,1700,1900,2100,2300,2500,2750,3000,3250,3500,3750,4000,4300,4600,4900,5200,5500];
nJ = length(J);
err_EKS = zeros(1,nJ);
for i = 1:nJ
    [~,errs] = EKS(J(i),Gx,y);
    err_EKS(i) = mean(errs(75:100));
end

%%
plot(log(J),log(err_EKS))
hold on
plot(log(J),log(160*(1./J).^(0.5)))
hold off

%%
function [u_EKS, err_EKS] = EKS(J,Gx,y)
    % Ensemble Kalman Algoritm Set Up
    n_max = 100;                                % Total number of iterations (matches choice from paper)
    u1_ensemble = normrnd(0,1,[1,J]);           % u1 ~ N[0,1]
    u2_ensemble = 90 + 20*rand(1,J);            % u2 ~ U[90,110]
    u_ensemble = [u1_ensemble; u2_ensemble];    % [u1,u2] ensemble tuples

    % Distribution Information
    I2 = eye(2);                        % dim(u1)xdim(u2) matrix; I2 ensures u1 and u2 are iid
    sigma = 10;                         % Noise parameter
    Gamma0 = sigma.^2*I2;               % Prior Distribution
    Gamma = 0.1^2 * I2;                 % Distribution
    eps = 10^-7;                        % Perturbation value to avoid ill-conditioned time-steps
    dt0 = 1000;                         % Initial time step (Adjust this as needed)

    % EKS: Iterations to Converge to an Approximation of the True Distribution

    u_n = u_ensemble;               % Initialize ensemble of particles at step 0
    err_EKS = zeros(1,n_max);       % Initialize matrix of y error values
    us_EKS = zeros(2,J,n_max);      % Initialize matrix storing information from each iterate

    for i = 1:n_max
        G_un = Gx(u_n(1,:),u_n(2,:));               % Compute the y-values predicted by the ensemble values
        Gbar = mean(G_un,2);                        % Update the average y-value predicted by the ensemble 
        in_prod = (G_un - Gbar)'*(Gamma\(G_un-y));  % Compute the inner product of the ensemble relative to its average versus the ensemble relative to the true y
        dtn = dt0/(norm(in_prod,"fro")+eps);        % Compute adaptive time step
        mean_adj = dtn/J*u_n*in_prod;               % Compute the term used to adjust the means to match the true distribution
    
        ubar = mean(u_n,2);                         % Compute the average ensemble coefficient values
        cov_U = 1/J*(u_n-ubar)*(u_n-ubar)';         % Compute the covariance matrix of the ensemble
        u_n_star = (I2 + dtn*cov_U*(Gamma0\I2))\(u_n - mean_adj); % Update the mean-improvement iterate

        % Compute errors and boundaries for each iterate
        err_EKS(i) = mean(vecnorm((y-G_un).*(Gamma\(y-G_un)),1,1));

        xi = normrnd(0,1,[2,J]);                    % Include random noise to perturb the data
        L = chol(cov_U);                            % Compute the square root of the covariance matrix using its Cholesky factorization
        u_n = u_n_star + sqrt(2*dtn)*L*xi;          % Compute the next ensemble iterate
    end
    u_EKS = u_n;
end