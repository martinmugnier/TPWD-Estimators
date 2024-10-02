function[beta,Gamma,obj] = MW_nucnormreg_estimator_prox(Y,X,psi,beta_init,...
                 tol,MaxIter,verbose)
    % MW_NUCNORMREG_ESTIMATOR_PROX Return the nuclear-norm regularized 
    % estimator proposed in Moon and Weidner (2018). Computation uses 
    % alternative optimization and the proximal gradient algorithm proposed 
    % in Mazumder et al. (2010).
    %
    % Author: Martin Mugnier
    % Email: martin.mugnier (at) psemail (dot) eu
    % (last update: August 2024)
    %
    % INPUTS:
    % -------
    % Y          : NxT array of balanced panel data outcome;
    % X          : NxTxK array of balanced panel data covariates;
    % psi        : scalar regularization parameter;
    % beta_init  : Kx1 array of initialization slope coefficients;
    % tol        : scalar tolerance criterion for convergence;
    % MaxIter    : maximum number of iterations;
    % verbose    : display optimization details if true. 
    %
    % OUTPUTS:
    % --------
    % beta       : Kx1 array of estimated common slope coefficients;
    % Gamma      : NxT array of estimated unobserved effects;
    % obj        : objective function value at the reported estimates.
    %
    % REFERENCES:
    % -----------
    % Moon, H. R. and Weidner, M. (2018), Nuclear Norm Regularized 
    % Estimation of Panel Regression Models.
    % Mazumder R., Hastie T., and Tibshirani R. (2010), Spectral 
    % Regularization Algorithms for Learning Large Incomplete Matrices. 
    % Jouranal of Machine Learning Research 1(11):2287-2322.

    
    if nargin<5
        tol = 1e-9; %default choice for precision.
    end   
    
    if nargin<6
        MaxIter = 1000; %default choice for maximum number of iterations 
    end   
    
    if nargin<7
        verbose = 0; %default choice is no display of optimization details.
    end 
    
    [N,T,K] = size(X);
    obj = 1e9;
    beta = beta_init;
    j=0;
    while true
        Z = Y-sum(X.*reshape(kron(beta',ones([N,T])),N,T,K),3);
        [U,Sig,V] = svd(Z,'econ');
        threshSig = Sig-sqrt(N*T)*psi;
        threshSig = threshSig.*(threshSig>0);
        Gamma = U*threshSig*V';
        nuc_norm_Gamma = sum(threshSig(:));
        beta_new = reshape(permute(X,[2,1,3]),N*T,K)...
                    \reshape((Y-Gamma)',N*T,1);
        obj_new = sum((Y-sum(X.*reshape(kron(beta_new',ones([N,T])),...
            N,T,K),3)-Gamma).^2,'all')/(2*N*T)+psi*nuc_norm_Gamma/sqrt(N*T);
        obj_diff = obj-obj_new;
        if obj_diff<=tol || j>MaxIter 
            break;
        else
            beta = beta_new;
            obj = obj_new;
            j = j+1;
        end
    end
    if verbose
        if j==MaxIter
            disp('Maximum number of iterations reached.');
        else
            disp('Successful convergence after '+string(j)+' iterations.');
        end
        disp('Objective value at current estimate: '+string(obj)+'.');
    end
end