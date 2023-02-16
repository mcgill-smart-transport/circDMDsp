function [U, Sigma, V] = svd_snapshot(Z, state)

[N, T] = size(Z);
U = [];

if T < N

    ATA = Z'*Z;

   [~, Sigma2, V] = svd(ATA);

   Sigma = sqrt(Sigma2);

   if state 
    
       U = Z * (V * diag(1./diag(Sigma)));

   end

else

    [V, Sigma, U] = svd_snapshot(Z', state);

    return;
end



