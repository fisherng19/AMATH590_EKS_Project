%% Ensemble Kalman Sampler for Low-Dimensional Parameter Space Example

% Goal: Find u1, u2 knowing y data

%% Set Up
% Problem Specific Information
x = [0.25; 0.75];                                   % Measurement Location
y = [27.5; 79.7];                                   % Observed Data
p = @(x,u1,u2) u2.*x + exp(-u1).*(-x.^2/2 + x/2);   % Component functions for operator G
G = @(x1,x2,u1,u2) [p(x1,u1,u2); p(x2,u1,u2)];      % Define operator G as G(u,x)
Gx = @(u1,u2) G(x(1),x(2),u1,u2);                   % Evaluate G at input x point to get G(u)

% Ensemble Kalman Sampling Information
n_max = 30;                         % Total number of iterations (matches choice from paper)
J = 10^3;                           % Number of ensemble particles
u1_ensemble = normrnd(0,1,[1,J]);   % u1 ~ N[0,1]
u2_ensemble = 90 + 20*rand(1,J);    % u2 ~ U[90,110]

I2 = eye(2);
sigma = 10;
Gamma0 = sigma.^2*I2;               % Prior Distribution
Gamma = 0.1^2 * I2;                 % Distribution
eps = 10^-8;                        % Perturbation value to avoid ill-conditioned time-steps
dt0 = 1;                            % Initial time step

u_n = [u1_ensemble; u2_ensemble];

% Plot Initial Ensemble Distribution
figure(1)
plot(u_n(1,:),u_n(2,:),'.')
xlim([-3,3])
ylim([85,115])
title('Initial Ensemble Distribution')
xlabel('u1');
ylabel('u2');

%% Iterate to Converge to an Approximation of the True Distribution
Gbar = 1/J*sum(Gx(u1_ensemble,u2_ensemble),2);      % Do I have to update Gbar every time?

for i = 1:n_max
    G_un = Gx(u_n(1,:),u_n(2,:));
    Gbar = 1/J*sum(G_un,2);                     % Not sure if this should be updated every step
    in_prod = (G_un - Gbar)'*(Gamma\(G_un-y));  % Not sure about definition of inner product with weighted kernel matrix
    dtn = dt0/(norm(in_prod,"fro")+eps);        % Not sure if I am taking the right entry on the bottom
    mean_adj = dtn/J*u_n*in_prod';
    
    ubar = 1/J*sum(u_n,2);
    cov_U = 1/J*(u_n-ubar)*(u_n-ubar)';
    
    u_n_star = (I2 + dtn*cov_U*(Gamma0\I2))\(u_n - mean_adj); % Still have questions about this weird equation

    xi = normrnd(0,1,[2,J]);
    u_n = u_n_star + sqrt(2*dtn*cov_U)*xi;
end

% Plot the Final Ensemble Distribution
figure(2)
plot(u_n(1,:),u_n(2,:),'.')
xlim([-3,3])
ylim([85,115])