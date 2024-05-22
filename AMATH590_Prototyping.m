%% Ensemble Kalman Sampler for Low-Dimensional Parameter Space Example

% Goal: Find u1, u2 knowing y data

%% Set Up
% Problem Specific Information
x = [0.25; 0.75];                                   % Measurement Location
y = [27.5; 79.7];                                   % Observed Data
p = @(x,u1,u2) u2.*x + exp(-u1).*(-1/2*(x).^2 + x/2);   % Component functions for operator G
G = @(x1,x2,u1,u2) [p(x1,u1,u2); p(x2,u1,u2)];      % Define operator G as G(u,x)
Gx = @(u1,u2) G(x(1),x(2),u1,u2);                   % Evaluate G at input x point to get G(u)

% Ensemble Kalman Algoritm Set Up
n_max = 40;                         % Total number of iterations (matches choice from paper)
J = 10^3;                           % Number of ensemble particles
u1_ensemble = normrnd(0,1,[1,J]);   % u1 ~ N[0,1]
u2_ensemble = 90 + 20*rand(1,J);    % u2 ~ U[90,110]
u_ensemble = [u1_ensemble; u2_ensemble];

% Distribution Information
I2 = eye(2);
sigma = 10;
Gamma0 = sigma.^2*I2;               % Prior Distribution
Gamma = 0.1^2 * I2;                 % Distribution
eps = 10^-7;                        % Perturbation value to avoid ill-conditioned time-steps
dt0 = 1000;                            % Initial time step

% Plot Initial Ensemble Distribution
figure(1)
plot(u_ensemble(1,:),u_ensemble(2,:),'.')
xlim([-3,3])
ylim([85,115])
title('Initial Ensemble Distribution')
xlabel('u1');
ylabel('u2');
grid on

%% EKS: Iterate to Converge to an Approximation of the True Distribution
u_n = u_ensemble;
err_EKS = zeros(1,n_max);
us_EKS = zeros(2,J,n_max);

for i = 1:n_max
    G_un = Gx(u_n(1,:),u_n(2,:));               % Compute the y-values predicted by the ensemble values
    Gbar = mean(G_un,2);                        % Update the average y-value predicted by the ensemble 
    in_prod = (G_un - Gbar)'*(Gamma\(G_un-y));  % Compute the inner product of the ensemble relative to its average versus the ensemble relative to the true y
    dtn = dt0/(norm(in_prod,"fro")+eps);        % Compute adaptive time step
    mean_adj = dtn/J*u_n*in_prod;               % Compute the term used to adjust the means to match the true distribution
    
    ubar = mean(u_n,2);                      % Compute the average ensemble coefficient values
    cov_U = 1/J*(u_n-ubar)*(u_n-ubar)';         % Compute the covariance matrix of the ensemble
    u_n_star = (I2 + dtn*cov_U*(Gamma0\I2))\(u_n - mean_adj); % Update the mean-improvement iterate

    us_EKS(:,:,i) = u_n;
    err_EKS(i) = mean(vecnorm((y-G_un).*(Gamma\(y-G_un)),1,1));
    bounds_EKS{i} = boundary(u_n(1,:)',u_n(2,:)',0);

    xi = normrnd(0,1,[2,J]);                    % Include random noise to perturb the data
    L = chol(cov_U);                            % Compute the square root of the covariance matrix using its Cholesky factorization
    u_n = u_n_star + sqrt(2*dtn)*L*xi;          % Compute the next ensemble iterate
end
u_EKS = u_n;

%% EKI Herty and Visconti: Iterate to Converge to an Approximation of the True Distribution
% This is another method against which we benchmark the EKS
u_n = u_ensemble;
err_HV = zeros(1,n_max);

for i = 1:n_max
    G_un = Gx(u_n(1,:),u_n(2,:));               % Compute the y-values predicted by the ensemble values
    Gbar = mean(G_un,2);                        % Update the average y-value predicted by the ensemble 
    in_prod = (G_un - Gbar)'*(Gamma\(G_un-y));  % Compute the inner product of the ensemble relative to its average versus the ensemble relative to the true y
    dtn = dt0/(norm(in_prod,"fro")+eps);        % Compute adaptive time step
    mean_adj = dtn/J*u_n*in_prod;               % Compute the term used to adjust the means to match the true distribution
    
    ubar = mean(u_n,2);                         % Compute the average ensemble coefficient values
    cov_U = I2;                                 % Compute the covariance matrix of the ensemble
    
    u_n_star = (I2 + dtn*cov_U*(Gamma0\I2))\(u_n - mean_adj); % Update the mean-improvement iterate

    err_HV(i) = mean(vecnorm((y-G_un).*(Gamma\(y-G_un)),1,1));
    
    xi = normrnd(0,1,[2,J]);                    % Include random noise to perturb the data
    L = chol(cov_U);                            % Compute the square root of the covariance matrix using its Cholesky factorization
    u_n = u_n_star + sqrt(2*dtn)*L*xi;          % Compute the next ensemble iterate
end
u_HV = u_n;


%% Plot the Final Ensemble Distribution
figure(2)
plot(u_EKS(1,:),u_EKS(2,:),'.')
hold on
plot(u_HV(1,:),u_HV(2,:),'.')
hold off
xlim([-4,-1.5])
ylim([103,106])
grid on
title('Final Ensemble Distribution')
legend('EKS','Herty and Visconti EKI');
xlabel('u1');
ylabel('u2');

%% Plot the Rough Area in Which the Data lives at each iterate
figure(3)
plot(u_EKS(1,:),u_EKS(2,:),'.','DisplayName','Final Distribution')
title('Distribution Domain of the Ensemble Points vs. Iterations')
hold on
for i = 3:3:n_max
    us1 = us_EKS(1,:,i);
    us2 = us_EKS(2,:,i);
    k = bounds_EKS{i};
    txt = ['N = ',num2str(i)]; 
    plot(us1(k),us2(k),'DisplayName',txt)
end
hold off
legend show
xlabel('u_1');
ylabel('u_2');

%% Plot Error/Convergence over Iterations
figure(4)
semilogy(1:n_max,err_EKS)
hold on
semilogy(1:n_max,err_HV)
hold off
grid on
title('Ensemble Error vs. Iterates')
xlabel('Iterate [N]');
ylabel('Error 1/J \Sigma_{j = 1}^J |y - G(u_j)|_\Gamma^2')