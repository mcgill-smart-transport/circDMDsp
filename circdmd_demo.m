clear,clc
close all


%% load data traffic speed

load speed.mat
load decrease

idx = decrease(:,5)+1;  % sensor id

timewindow = 288;          % 288 timestamps/day
    
tau = timewindow*3;        % tau: delay embeddinng length (hyper-parameter, needs to be tuned carefully)

Tpre = timewindow*7;

Xtr = data(idx,1+timewindow*2:timewindow*16);   % training dataset (like 2 weeks)

Xpre = data(idx,timewindow*16+1:timewindow*16+Tpre);    % testing dataset (1 week)


%% circdmdsp
delta = 0;      % delta = 0, use optimal threshold to determine the rank r in svd
sp = 1;         % sp = 1, use sparse-promoting strategy to select the dominant modes
gamma = 500;    % sparsity parameter in sparse-promoting stage
dt = 1/12;      % time resolution (hours) here is 5 mins

tic
[Phi, Vand, b, lambda, Period, Xhat, Xpre_hat] = circdmd_sp(Xtr, tau, delta, Tpre, dt, sp, gamma);
toc

%%
% training results
rmse_tr = sqrt(norm(Xhat - Xtr, 'fro').^2/numel(Xtr))
mae_tr = sum(abs(Xhat - Xtr),"all")/numel(Xtr)

% testing results
mae_pre = sum(abs(Xpre_hat - Xpre),"all")/numel(Xpre)
rmse_pre = sqrt(norm(Xpre_hat - Xpre, 'fro').^2/numel(Xpre))


%%
figure
imagesc(Xpre_hat)
colormap(flipud(turbo))
clim([0 70])
title('prediction')

figure
imagesc(Xpre)
colormap(flipud(turbo))
clim([0 70])
title('data')

