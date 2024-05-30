%% Ensemble Kalman Sampler for Low-Dimensional Parameter Space Example

% Goal: Find u1, u2 knowing y data

%% Set Up
Ex_num = 2;         % Choose the example number to run

n_max = 500;         % Total number of iterations (matches choice from paper)
J = 500;           % Number of ensemble particles

%% Example Set-ups

% Linear Example (Self-Generated)
% This linear example is of the form G(u) = <u,x> = u1 x1 + u2 x2 + ... + ud xd 
if Ex_num == 1
    % Define the measurement data
    d = 2;                                      % Dimension of spatial vector
    x = rand(d,1);                              % Generate a random test point
    u_dagger = randi([1,5],d,1);                % Generate a random truth
    y = x.*u_dagger + normrnd(0,0.1,[d,1]);     % Generate a measurement at x under true u perturbed by noise
    
    % Define the operator
    p = @(x,u) x.*u;                            % Define the operator component-wise
    G = @(x,u) cell2mat(arrayfun(@(c) p(x,u(:,c)),1:size(u,2),'UniformOutput',false));
    Gx = @(u) G(x,u);                           % Define the operator evaluated at the given x
    
    % Ensemble Information
    u_ensemble = normrnd(3,2,[d,J]);            % Create an ensemble vector
    dt0 = 50;                                  % Define an initial time step empirically

% Nonlinear Example (Paper)
% This nonlinear example is from the main EKS paper to reproduce results
elseif Ex_num == 2
    % Define the measurement data
    d = 2;                                      % Dimension of spatial vector
    x = [0.25; 0.75];                           % Measurement Location

    
    % Define the operator
    p_ij = @(x,u) u(2).*x + exp(-u(1)).*(-1/2*(x).^2 + x/2);    % Define the operator component-wise
    p = @(x,u) arrayfun(@(c) p_ij(x,u(:,c)),1:size(u,2));       % Define a column-wise evaluator for input matrix of ensemble points
    G = @(x,u) [p(x(1),u); p(x(2),u)];                          % Define the operator by its p(x) components
    Gx = @(u) G(x,u);                                           % Define the operator evaluated at the given x

    u_dagger = [normrnd(0,1,[1,1]) ; 90 + 20*rand(1,1)];
    y = G(x,u_dagger) + normrnd(0,0.1,[d,1]);                           % Observed Data
    % Ensemble information
    u1_ensemble = normrnd(0,1,[1,J]);           % u1 ~ N[0,1]
    u2_ensemble = 90 + 20*rand(1,J);            % u2 ~ U[90,110]
    dt0 = 1000;                                 % Initial time step (Adjust this as needed)
    u_ensemble = [u1_ensemble; u2_ensemble];    % [u1,u2] ensemble tuples
elseif Ex_num == 3
    % Define the measurement data
    d = 2^8;            % Number of unknown parameters u    
    K = sqrt(d);        % Number of lattice points
    idx = 1:1:K;        % Indices of lattice points
    [X,Y] = meshgrid(idx,idx);
    u_ensemble = normrnd(0,1,[d,J]);
    %eta = normrnd(0,0.1^2,[K,1]);
    %y = G;
    Tau = 3;
    alpha = 2;
end

%% Ensemble Kalman Algorithm Set Up

% Distribution Information
Id = eye(d);                        % dim(u1)xdim(u2) matrix; Id ensures ui are iid
sigma = 10;                         % Noise parameter
Gamma0 = sigma.^2*Id;               % Prior Distribution
Gamma = 0.1^2 * Id;                 % Distribution
eps = 10^-7;                        % Perturbation value to avoid ill-conditioned time-steps

%% EKS: Iterations to Converge to an Approximation of the True Distribution
% The EKS method is the primary method we use to generate an approximation
% to the true distribution

u_n = u_ensemble;               % Initialize ensemble of particles at step 0
err_EKS = zeros(1,n_max);       % Initialize vector of y error values
us_EKS = zeros(d,J,n_max);      % Initialize matrix storing information from each iterate
t_n = zeros(1,n_max);           % Initialize vector of time values

for i = 2:n_max
    % Intermediate step
    G_un = Gx(u_n);
    Gbar = mean(G_un,2);                        % Update the average y-value predicted by the ensemble 
    in_prod = (G_un - Gbar)'*(Gamma\(G_un-y));  % Compute the inner product of the ensemble relative to its average versus the ensemble relative to the true y
    dtn = dt0/(norm(in_prod,"fro")+eps);        % Compute adaptive time step
    mean_adj = dtn/J*u_n*in_prod;               % Compute the term used to adjust the means to match the true distribution
    ubar = mean(u_n,2);                         % Compute the average ensemble coefficient values
    cov_U = 1/J*(u_n-ubar)*(u_n-ubar)';         % Compute the covariance matrix of the ensemble
    u_n_star = (Id + dtn*cov_U*(Gamma0\Id))\(u_n - mean_adj); % Update the mean-improvement iterate

    % Compute errors for each iterate
    us_EKS(:,:,i) = u_n;                        % Store the u_n values
    err_EKS(i) = mean(vecnorm((y-G_un).*(Gamma\(y-G_un)),1,1));     % Compute the errors of G(u) relative to y
    t_n(i) = t_n(i-1)+dtn;                      % Compute the current time value

    % Final step
    xi = normrnd(0,1,[d,J]);                    % Include random noise to perturb the data
    L = chol(cov_U);                            % Compute the square root of the covariance matrix using its Cholesky factorization
    u_n = u_n_star + sqrt(2*dtn)*L*xi;          % Compute the next ensemble iterate
end
u_EKS = u_n;        % Compute final ensemble prediction

%% EKI Herty and Visconti: Iterate to Converge to an Approximation of the True Distribution
% This is another method against which we benchmark the EKS; the difference
% is that the covariance matrix is replaced by the identity matrix, leading
% to more variance retained in the data

u_n = u_ensemble;               % Initialize ensemble of particles at step 0
err_HV = zeros(1,n_max);        % Initialize vector of y error values
for i = 2:n_max
    % Intermediate step
    G_un = Gx(u_n);
    Gbar = mean(G_un,2);                        % Update the average y-value predicted by the ensemble 
    in_prod = (G_un - Gbar)'*(Gamma\(G_un-y));  % Compute the inner product of the ensemble relative to its average versus the ensemble relative to the true y
    dtn = dt0/(norm(in_prod,"fro")+eps);        % Compute adaptive time step
    mean_adj = dtn/J*u_n*in_prod;               % Compute the term used to adjust the means to match the true distribution
    ubar = mean(u_n,2);                         % Compute the average ensemble coefficient values
    cov_U = Id;                                 % Compute the covariance matrix of the ensemble
    u_n_star = (Id + dtn*cov_U*(Gamma0\Id))\(u_n - mean_adj); % Update the mean-improvement iterate

    % Compute error values for each iterate
    err_HV(i) = mean(vecnorm((y-G_un).*(Gamma\(y-G_un)),1,1));
    
    % Final step
    xi = normrnd(0,1,[d,J]);                    % Include random noise to perturb the data
    L = chol(cov_U);                            % Compute the square root of the covariance matrix using its Cholesky factorization
    u_n = u_n_star + sqrt(2*dtn)*L*xi;          % Compute the next ensemble iterate
end
u_HV = u_n;         % Compute final ensemble prediction

%% EKI
u_n = u_ensemble;               % Initialize ensemble of particles at step 0
err_EKI = zeros(1,n_max);        % Initialize vector of y error values
for i = 2:n_max
    G_un = Gx(u_n);
    Gbar = mean(G_un,2);                        % Update the average y-value predicted by the ensemble 
    in_prod = (G_un - Gbar)'*(Gamma\(G_un-y));  % Compute the inner product of the ensemble relative to its average versus the ensemble relative to the true y
    dtn = dt0/(norm(in_prod,"fro")+eps);        % Compute adaptive time step
    mean_adj = dtn/J*u_n*in_prod;               % Compute the term used to adjust the means to match the true distribution
    ubar = mean(u_n,2);                         % Compute the average ensemble coefficient values
    cov_U = Id;                                 % Compute the covariance matrix of the ensemble
    u_n = u_n - mean_adj;                       % Update the mean-improvement iterate

    % Compute error values for each iterate
    err_EKI(i) = mean(vecnorm((y-G_un).*(Gamma\(y-G_un)),1,1));
end
u_EKI = u_n;         % Compute final ensemble prediction

%% Plot the Initial and Final Ensemble Distributions Side-by-Side
figure(1)
% subplot(1,2,1);
% plot(u_ensemble(1,:),u_ensemble(2,:),'.','MarkerSize',7.5,'DisplayName','Initial Distribution')
% if Ex_num == 1
%     hold on
%     plot(u_dagger(1),u_dagger(2),'.k','MarkerSize',15,'DisplayName','$u^\dagger$');
%     hold off
% end
% if Ex_num == 2
%     xlim([-3,3])
%     ylim([60,120])
% end
% title('Initial Ensemble Distribution','Interpreter','latex')
% lgd1a = legend;
% legend show
% set(lgd1a,'Interpreter','latex');
% xlabel('$u_1$','Interpreter','latex');
% ylabel('$u_2$','Interpreter','latex');
% grid on
% pbaspect([1 1 1])

% subplot(1,2,2);
clf;
plot(u_EKS(1,:),u_EKS(2,:),'.','MarkerSize',7.5,'DisplayName','EKS');
hold on;
plot(u_HV(1,:),u_HV(2,:),'.','MarkerSize',5,'DisplayName','Noisy EKI (Herty and Visconti)')
plot(u_EKI(1,:),u_EKI(2,:),'.','MarkerSize',10,'DisplayName','EKI (Noise-Free)')
if Ex_num == 1
    plot(u_dagger(1),u_dagger(2),'.k','MarkerSize',15,'DisplayName','$u^\dagger$');
end
if Ex_num == 2
    % xlim([-4,4])
    % ylim([90,110])
    plot(u_dagger(1),u_dagger(2),'.k','MarkerSize',15,'DisplayName','$u^\dagger$');
end
grid on
title('Final Ensemble Distribution','Interpreter','latex')
lgd1b = legend;
legend show
set(lgd1b,'Interpreter','latex');
xlabel('$u_1$','Interpreter','latex');
ylabel('$u_2$','Interpreter','latex');
pbaspect([1 1 1])
hold off

saveas(gcf,['Ex',num2str(Ex_num),'_Initial_vs_Final.png'])

%% Plot the Rough Area in Which the Data lives at each iterate
figure(2)
% plot(u_ensemble(1,:),u_ensemble(2,:),'.','DisplayName','N = 0')
% title('Distribution Domain of the Ensemble Points vs. Iterations','Interpreter','latex')
skip = 3;
idx = skip:skip:n_max;
n_idx = length(idx);
us1 = us_EKS(1,:,skip);
us2 = us_EKS(2,:,skip);
scatter(us1,us2,'.','MarkerFaceAlpha',(i/skip-1)/n_idx,'DisplayName',['N = ',num2str(i)])
xl = gcf().CurrentAxes.XLim;
yl = gcf().CurrentAxes.YLim;
for i = idx
    us1 = us_EKS(1,:,i);
    us2 = us_EKS(2,:,i);
    scatter(us1,us2,'.','MarkerFaceAlpha',(i/skip-1)/n_idx,'DisplayName','Ensemble Points')
    xlim(xl);
    ylim(yl);
    hold on;
    plot(mean(us1),mean(us2),'.','Color',"#A2142F", 'MarkerSize',25,'DisplayName','$\bar{u}$');
    if Ex_num == 1
        plot(u_dagger(1),u_dagger(2),'.k','MarkerSize',25,'DisplayName','$u^{\dagger}$');
        legend show
        legend('Interpreter','latex','FontSize',15)
    end
    title("EKS for G(u)=u_1x_1+u_2x_2 with J="+J,'FontSize',15);
    xlabel("u_1"); ylabel("u_2");
    hold off;
    exportgraphics(gcf,"Linear_"+J+"_mean.gif",Append=true)
end

% lgd = legend;
% legend show
% set(lgd,'Interpreter','latex');
% xlabel('u_1','Interpreter','latex');
% ylabel('u_2','Interpreter','latex');
% grid on
% hold off

% saveas(gcf,['Ex',num2str(Ex_num),'_EKS_Distribution.png'])

%% Plot the Error/Convergence over Iterations
figure(3)
semilogy(1:n_max,err_EKS,'LineWidth', 2.5, 'DisplayName', 'Ensemble Kalman Sampler')
hold on
semilogy(1:n_max,err_HV,'LineWidth', 2.5, 'DisplayName', 'Ensemble Kalman Inversion (Herty and Visconti)')
semilogy(1:n_max,err_EKI,'LineWidth', 2.5, 'DisplayName', 'Ensemble Kalman Inversion (Noise-Free)')
hold off
grid on
title('Ensemble Error vs. Iterates', 'Interpreter','latex')
xlabel('Iterate [N]','Interpreter','latex');
ylabel('Error $\frac{1}{J} \Sigma_{j = 1}^J |y - G(u_j)|_\Gamma^2$','Interpreter','latex')
lgd = legend;
set(lgd,'Interpreter','latex');
pbaspect([1.5 1 1])

saveas(gcf,['Ex',num2str(Ex_num),'_EKS_Error.png'])

%% Plot the Error vs. Time
figure(4)
semilogy(t_n,err_EKS,'LineWidth',2.5)
grid on
hold on
semilogy(t_n,exp(-2*t_n))
hold off
title('Ensemble Error vs. Iterates','Interpreter','latex')
xlabel('Time [t = \Sigma_{j=1}^n \Delta t_j]');
ylabel('Error $\frac{1}{J} \Sigma_{j = 1}^J |y - G(u_j)|_\Gamma^2$','Interpreter','latex')
%legend('Ensemble Kalman Sampler','Ensemble Kalman Inversion (Herty & Visconti)')

%% Compute the Error of the Ensemble Distribution Relative to the True u
if Ex_num == 1
    disp(norm(mean(u_EKS,2) - u_dagger,2))  % Not sure if I am computing this correctly
    disp(norm((u_EKS - u_dagger)/J,2))
end