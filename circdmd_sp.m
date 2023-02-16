
function [Phi, Vand, b, lambda, Period, Xhat, Xpre_hat] = circdmd_sp(X, tau, delta, Tpre, dt, sp, gamma)
%% anti-circulant dynamic mode decomposition with sparsity promoting (circDMDsp)

% @ Xudong Wang (xudong.wang2@mail.mcgill.ca)
% 2023-02

% input:
% X: training data (N by T)
% tau: delay embedding length in anti-circulant matrix (scalar)
% delta: rank selection method (0: optimal threshold, other: energy)
% Tpre: prediction length (scalar)
% sp: 0 not use sparse-promoting, 1: use sparse-promoting
% gamma: parameter in sp to balance the number of modes


% output:
% Phi: dynamic modes (Ntau by r)
% Vand: Vandermonde matrix (r by T_psi)
% b: amplitudes (r)
% lambda: eigenvalues of dynamical matrix (r)
% Period: oscillation periods (r)
% Xhat: reconstruction (N by T)
% Xhat_pre: prediction (N by Tpre)



%% Construct anti-circulant matrix 

[N,Tx] = size(X);

C = zeros(N*tau,Tx);

% Eq. (11)
for i = 1:tau
    C(N*(i-1)+1:N*i,:) = circshift(X,-i,2);
end


T = size(C,2);
Pt = eye(T);
idx = [T,1:T-1];
Pt = Pt(:,idx);


%% DMD main

% calculate svd of CP matrix using the method of snapshots
[U, S, V] = svd_snapshot(C*Pt, 1);


% Determine the rank  r using energy
if delta >0 && delta <= 1

    sum_sigma = cumsum(diag(S))./sum(S,'all');
    r = find(sum_sigma > delta, 1);

% Determine the rank r using optimal
elseif delta == 0

    beta = min(size(C,1)/size(C,2), size(C,2)/size(C,1));
    ome = 0.56*beta^3 - 0.95*beta^2 + 1.82*beta +1.43;
    tau  = median(diag(S))*ome;
    r = sum(diag(S) > tau);

end

disp('Rank done')


% truncate to rank-r
Ur = U(:, 1:r); 
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);


Atilde = Ur' * C * Vr / Sr;  % low-rank dynamics (eq. (17))

[W, Lambda] = eig(Atilde);   % W: Atilde modes (eq. (18))
lambda = diag(Lambda);       % discrete-time eigenvalues
 
% estimation dynamics （Vandermode matrix)
Vand = zeros(r,T);

for i = 1:T
    Vand(:,i) = lambda.^(i-1);
end

disp('DMD done')

%% Sparsity-promoting algorithm or not

if sp == 1

    G = Sr*Vr';
    P = (W'*W).*conj(Vand*Vand');
    q = conj(diag(Vand*Vr*Sr'*W));
    s = trace(G'*G);

        
    % Set options for dmdsp
    options = struct('rho',0.001,'maxiter',10000,'eps_abs',1.e-6,'eps_rel',1.e-4);
    
     
    answer = dmdsp(P,q,s,gamma,options);
   

    % DMD modes (projected modes)
    Phi = Ur*W;
    b = answer.xpol;

else
    
    % DMD modes (exact modes)
    Phi = C * Vr / Sr * W;
    b = Phi\C(:,T);         % first timestamp is the last column of C
    
end


omega = log(lambda)/dt; % continuous-time eigenvalues

f = imag(omega)/(2*pi); % frequency
Period = 1./f;   % period

disp('Modes done')


%% prediction

i = 1+T:Tpre+T;
Vand2 = zeros(r, Tpre);
Vand2(:,i-T) = lambda.^(i-1);
Vand_pre = [Vand Vand2];

% here we can use the first N rows of Phi to conduct the results (faster)
Xdmd = real((Phi(1:N,:)*diag(b)*Vand_pre)); 

% or we can use Ntau rows of Phi to conduct the results 
% Xdmd = real(circulant2mat(Phi*diag(b)*Vand_pre, tau)); 

% reconstruction
Xhat = Xdmd(:,1:T);

% prediction
Xpre_hat = Xdmd(:,T+1:T+Tpre);



%% sort modes
[~,idx] = sort(b,'descend');
Phi = Phi(:,idx);
Vand = Vand(idx,:);
b = b(idx);
lambda = lambda(idx);
Period = Period(idx);

