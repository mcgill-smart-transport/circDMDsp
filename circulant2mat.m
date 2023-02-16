function mat = circulant2mat(C,K)

% C is a circulant matrix
% K is the delay embedding length

[NK,T] = size(C);
N = NK/K;

mat = zeros(N,T);

for i = 1:K

    temp = circshift(C(N*(i-1)+1:N*i,:), i-1, 2);

    mat = mat + temp;

end

mat = mat/K;