function beta = CM_spectral_estimator(Y,X,GM)
    % CM_SPECTRAL_ESTIMATOR Return the spectral estimator proposed in
    % Chetverikov and Manresa (2022).
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y          : NxT array of balanced panel data outcome;
    % X          : NxTxK array of balanced panel data covariates; 
    % GM         : product of the total numbers of factors (outcome and 
    %              covariate models).
    %
    % OUTPUT:
    % -------
    % beta       : Kx1 array of common slope coefficients.
    %
    % REFERENCE:
    % ----------
    % Chetverikov, D. and Manresa, E. (2022), Spectral and post-spectral 
    % estimators for grouped panel data models.
    
    [~,~,K] = size(X);
    L = f(Y,X,GM,zeros(K,1));
    S = zeros(K,1);
    Omega = zeros(K);
    e = zeros(K,1);
    for k=1:K
        e_k = e;
        e_k(k) = 1;
        S(k) = (f(Y,X,GM,e_k)-f(Y,X,GM,-e_k))/2;
        Omega(k,k) = (f(Y,X,GM,e_k)+f(Y,X,GM,-e_k))/2-L;
    end
    for k=1:K
        e_k = e;
        e_k(k) = 1;
         for l=(k+1):K
            e_l = e;
            e_l(l) = 1;
            Omega(k,l) = (f(Y,X,GM,e_k+e_l)-Omega(k,k)-Omega(l,l)- ...
                S(k)-S(l)-L)/2;
            Omega(l,k) = Omega(k,l);
         end
    end
    beta = -(Omega\S)/2;
end

function f_val = f(Y,X,GM,b)
    [N,T,K] = size(X);
    res =  Y-sum(X.*reshape(kron(b',ones([N,T])),N,T,K),3);
    A = sum((permute(res-permute(res,[3 2 1]),[1 3 2])).^2,3)/(N*T);
    [~, spectrum] = eigs(A,2*(GM+1),'largestabs');
    f_val = sum(diag(spectrum));
end

