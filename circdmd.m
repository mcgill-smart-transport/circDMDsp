
function [Phi, Vand, b, lambda, Period, Xhat] = circdmd(X, K, delta, predict, dt)
%%

[N,T] = size(X);

C = zeros(N*K,T);

for i = 1:K
    C(N*(i-1)+1:N*i,:) = X;
    X = circshift(X, [0,-1]);
end


T2 = size(C,2);
P = eye(T2);
idx = [2:T2,1];
P = P(:,idx);


[U, S, V] = svd(C, 'econ');

disp('SVD done')

%%  determine the SVD rank r

% using energy
if delta >0 && delta <= 1

    sum_sigma = cumsum(diag(S))./sum(S,'all');
    r = find(sum_sigma > delta, 1);

% using optimal
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


Atilde = Ur' * C* P * Vr / Sr; % low-rank dynamics

[W, D] = eig(Atilde);   % W: Atilde modes

disp('Atilde eigendecompose done')

%%  DMD modes

Phi = C * P * Vr / Sr * W; 

% Phi = Ur * W;

lambda = diag(D); % discrete-time eigenvalues

omega = log(lambda)/dt; % continuous-time eigenvalues

f = imag(omega)/(2*pi); % frequency
Period = 1./f;   % period

disp('Modes done')

%% Compute DMD mode amplitudes b
x1 = C(:, 1);
b = Phi\x1;

%% temporal
T = size(C, 2)+predict; 

Vand = zeros(r,T);

for i = 1:T
    Vand(:,i) = lambda.^(i-1);
end

disp('Temporal done')

%%
Xdmd = Phi*diag(b)*Vand; 
Xhat = real(circulant2mat(Xdmd, K));
Xhat(Xhat<0) = 0;

%%
[~,idx] = sort(b,'descend');
Phi = Phi(:,idx);
Vand = Vand(idx,:);
b = b(idx);
lambda = lambda(idx);
Period = Period(idx);

