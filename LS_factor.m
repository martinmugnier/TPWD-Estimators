%====================================================================================
% LEAST SQUARES ESTIMATION OF LINEAR PANEL DATA MODELS WITH INTERACTIVE FIXED EFFECTS
%====================================================================================
%
% AUTHOR OF THIS CODE: 
%=====================
%
% Martin Weidner, University College London, and CeMMaP
% Email: m.weidner (at) ucl (dot) ac (dot) uk
% (last update: December 2013)
%
% DISCLAIMER:
%============
%
% This code is offered with no guarantees. Not all features of this code
% were properly tested. Please let me know if you find any bugs or
% encounter any problems while using this code. All feedback is
% appreciated.
%
% REFERENCES:
%============
%
% For a description of the model and the least squares estimator see e.g.
% Bai (2009, "Panel data models with interactive fixed effects"), or
% Moon and Weidner (two working papers:
% "Dynamic Linear Panel Regression Models with Interactive Fixed Effects"
% "Linear Regression for Panel with Unknown Number of Factors as Interactive Fixed Effects").
%
% THREE DIFFERENT COMPUTATION METHODS ARE IMPLEMENTED:
%=====================================================
%
% METHOD 1: (recommended default method)
%----------
% iterate the following two steps until convergence:
% #Step 1: for given beta compute update for lambda and f as
%          principal components of Y-beta*X
% #Step 2: for given lambda and f update beta by runing a pooled OLS regression of
%          M_lambda*Y*M_f (or equivalently of just Y itself) on M_lambda*X*M_f.
% #The procedure is repeated multiple times with different starting values.
%
%
% METHOD 2: 
%----------
% #The profile objective function (after profiling out lambda and f) is
%  optimized over beta using "fminunc"
% #The procedure is repeated multiple times with different starting values.
%
%
% METHOD 3: (described in Bai, 2009)
%----------
% iterate the following two steps until convergence:
% #Step 1: for given beta compute update for lambda and f as
%          principal components of Y-beta*X (same as in method 1)
% #Step 2: for given lambda and f run a pooled OLS regression of
%          Y-lambda*f' on X to update beta.
% #The procedure is repeated multiple times with different starting values.
%
%
% COMMENTS:
%==========
% # Another method would be to use Step 1 as in Method 1&3, but to 
%   replace step 2 with a regression of Y on either M_lambda*X or X*M_f, i.e.
%   to only project out either lambda or f in the step 2 regression.
%   Bai (2009) mentions this method and refers to Ahn, Lee, and Schmidt (2001),
%   Kiefer (1980) and Sargan (1964) for this. We have not tested this
%   alternative method, but we suspect that Method 1 performs better in
%   terms of speed of convergence.
%
% # This alternative method and the method proposed by Bai (2009) --- i.e.
%   "method 3" here --- have the property of reducing the LS objective function in
%   each step. This is not true for Method 1 and may be seen as a
%   disadvantage of Method 1. However, we found this to be a nice feature,
%   because we use this property of Method 1 as a stopping rule:
%   if the LS objective function does not improve, then we know we are
%   "far away" from a proper minimum, so we stop the iteration
%   and begin the iteration with another randomly chosen starting value.
%   Note that multiple runs with different starting values are required
%   anyways for all methods (because the LS objective function may have
%   multiple local minima).
%
% # We recommend method 1, because each iteration step is fast (quicker
%   than method 2, which needs to compute a gradient (and Hessian?) in each
%   step, involving multiple evaluations of the objective function) and its
%   rate of convergence in our tests was very good (faster than method 3).
%   However, we have not much explored the relative sensativity of the
%   different methods towards the choice of starting value. Note that by
%   choosing the quickest method (method 1) one can try out more different
%   starting values of the procedure in the same amount of time.
%   Nevertheless, it may well be that method 2 or 3 or the alternative
%   method described above perform better in certain situations.


function [beta,exitflag,lambda,f,Vbeta1,Vbeta2,Vbeta3,bcorr1,bcorr2,bcorr3]=LS_factor(Y,X,R,report,precision_beta,method,start,repMIN,repMAX,M1,M2)
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % REQUIRED INPUT PARAMETERS: 
 %     Y = NxT matrix of outcomes
 %     X = KxNxT multi-matrix of regressors
 %     R = positive integer 
 %         ... number of interactive fixed effects in the estimation
 %
 %     Comment: we assume a balanced panel,
 %              i.e. all elements of Y and X are known.
 %
 % OPTIONAL INPUT PARAMETERS: 
 %     report = 'silent' ... the program is running silently
 %            = 'report' ... the program reports what it is doing
 %     precision_beta = defines stopping criteria for numerical optimization,
 %                      namely optimization is stopped when difference in beta
 %                      relative to previous opimtization step is smaller than
 %                      "precision_beta" (uniformly over all K components of beta)
 %            NOTE: the actual precision in beta will typically be lower
 %            than precision_beta, depending on the convergence rate of the
 %            procedure.
 %     method = 'm1' or 'm2' or 'm3' ... which optimization method is used 
 %                              (described above)
 %     start = Kx1 vector, first starting value for numerical optimization
 %     repMIN = positive integer
 %              ... minimal number of runs of optimization with different starting point
 %     repMAX = positive integer
 %              ... maximal number of runs of optimization (in case numerical optimization
 %              doesn't terminate properly, we do multiple runs even for repMIN=1)
 %     M1 = positive integer 
 %          ... bandwidth for bias correction for dynamic bias (bcorr1),
 %              M1=number of lags of correlation between regressors and
 %              errors that is corrected for in dynamic bias correction
 %     M2 = non-negative integer 
 %          ... bandwidth for bias correction for time-serial correlation (bcorr3),
 %              M2=0 only corrects for time-series heteroscedasticity,
 %              while M2>0 corrects for time-correlation in erros up to lag M2 
 %
 % OUTPUT PARAMETERS:
 %     beta = parameter estimate 
 %     exitflag = 1 if iteration algorithm properly converged at optimal beta
 %              = -1 if iteration algorithm did not properly converge at optimal beta
 %     lambda= estimate for factor loading
 %     f = estimate for factors
 %     Vbeta1 = estimated variance-covariance matrix of beta,
 %              assuming homoscedasticity of errors in both dimensions
 %     Vbeta2 = estimated variance-covariance matrix of beta,
 %              assuming heteroscedasticity of errors in both dimensions
 %     Vbeta3 = estimated variance-covariance matrix of beta,
 %              allowing for time-serial correlation up to lag M2
 %              (i.e. if M2==0, then Vbeta2==Vbeta3)
 %     bcorr1,2,3 = estimate for the three different bias components
 %            (needs to be subtracted from beta to correct for the bias)
 %                  bcorr1 = bias due to pre-determined regressors
 %                  bcorr2 = bias due to cross-sectional heteroscedasticity
 %                           of errors
 %                  bcorr3 = bias due to time-serial heteroscedasticity and
 %                           time-serial correlation of errors
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
  %COMMENT: We assume that all provided input parameters have values and dimensions
  %as descibed above. The program could be impoved by checking that this is
  %indeed the case.
 
  K=size(X,1);   %number of regressors
  N=size(X,2);   %cross-sectional dimension
  T=size(X,3);   %time-serial dimension  
 
  %Input parameters that are not provided are given default parameters as
  %follows:
  
  if nargin<4
    report = 'report'; %default choice is to report what is going on  
  end    
  if nargin<5
    precision_beta = 10^-8;  
  end    
  if nargin<6
    method='m1';  %default computation method is Method 1 described above
  end
  if nargin<7
    start=zeros(K,1);
  end  
  if nargin<8
    repMIN=30;
  end  
  if nargin<9
    repMAX=10*repMIN;
  end  
  if nargin<10
    M1=1;
  end   
  if nargin<11
    M2=0;
  end     
 

  
  %if N<T we permute N and T in order to simplify computation of
  %eigenvectors and eigenvalues (we only solve the eigenvalue
  %problems for TxT matrices and we make sure that T<=N)
  trans=0;   
  if N<T
    trans=1;  %dummy variable to remember that we exchanged N and T dims.
    NN=N; N=T; T=NN;
    Y=Y';
    X=permute(X,[1,3,2]);    
  end
  
  %NUMERICAL OPTIMIZATION TO OBTAIN beta:
  
  beta=inf*ones(length(start),1);   %best beta found so far
  obj0=inf;    %objective function at optimal beta found so far
  count=0;     %how many minimization runs properly converged so far
  exitflag=-1; %no proper solution found, yet 
  for i=1:repMAX  
    if count<repMIN  
      
      %CHOOSE STARTING VALUE FOR OPTIMIZATION
      if i==1
        st=start;  %first starting value is given by user (or =0 by default)
      else   
        st=start+randn(length(start),1); %choose random starting values up from second run
           %COMMENT: this is a very simple way of choosing different starting
           %values. One might want to modify this deopending on the problem
           %one consideres.
      end
      
      %REPORT TO USER:
      if report=='report'
        disp([10 '  Program LS_factor now starting optimization run number: ' num2str(i) ' / ' num2str(repMAX)]);
        disp(['    number of optimization runs that converged so far: ' num2str(count) ' / ' num2str(repMIN)]);
        disp(['    starting value for current run = ' num2str(st')]);
      end    
      
      %RUN ACTUAL OPTIMIZATION OVER beta:
      if method=='m1'
        [para,obj,ef]=minimize_obj_method1(Y,X,R,st,precision_beta);
      elseif method=='m2'
        [para,obj,ef]=minimize_obj_method2(Y,X,R,st,precision_beta);
      elseif method=='m3'
        [para,obj,ef]=minimize_obj_method3(Y,X,R,st,precision_beta);
      end          
      
      %REPORT TO USER:
      if report=='report'
        if ef>0
          disp(['    Method ' num2str(method) ' converged at beta = ' num2str(para')]);
        else    
          disp(['    Method ' num2str(method) ' did NOT converge. Stopped at beta = ' num2str(para')]);
        end  
        if obj < obj0
          disp(['    Final Objective = ' num2str(obj) ' ==> NEW BEST MINIMUM FOUND']); 
        else
          disp(['    Final Objective = ' num2str(obj) ' > Best Objective so Far = ' num2str(obj0) ]); 
        end    
      end    
        
      %UPDATE ESTIMATOR, IN CASE BETTER SOLUTION FOUND:
      if (obj < obj0)
        obj0=obj;
        beta=para;      %new "global minimum" found
        if ef>0
          exitflag=1;      %optimal beta corresponds to point where 
                           %iteration algorithm properly converged 
        else
          exitflag=-1;
        end
      end
     
      %UPDATE COUNTER OF "good" SOLUTIONS:
      if ef>0   %if method properly converged, then
        count=count+1;   %count how many "good" solutions found
      end
    end  
  end %end of calculation of beta-estimator
    

  %CALCULATE lambda AND f FOR THE OPTIMAL beta:
  if nargout>2   %if user also asked for calculation of lambda and f:
    res1=Y;
    for k=1:K
      res1=res1-beta(k)*squeeze(X(k,:,:));
    end
    [V,D]=eig(res1'*res1);
    [Dsort,Ind]=sort(diag(D)); %sort eigenvalues, just to be sure
    f=V(:,Ind(T-R+1:T));       %eigenvectors corresponding to largest eigenvalues
    for r=1:R
      f(:,r)=f(:,r)/norm(f(:,r));
      if mean(f(:,r))<0
        f(:,r)=-f(:,r);
      end  
    end
    lambda=res1*f;
    res=res1-lambda*f';  %estimate for the residuals
   
    if trans==1     %need to undo the interchange of N and T now
      save=lambda; lambda=f; f=save;    
      res=res';
      NN=N; N=T; T=NN;
      Y=Y';
      X=permute(X,[1,3,2]);    
    end  
  end
  
  %CALCULATE VARIANCE-COVARIANCE MATRIX OF beta:
  if nargout>4   %user also asked for calculation of variance covariance matrix
    Pf=f*inv(f'*f)*f';  
    Plambda=lambda*inv(lambda'*lambda)*lambda';  
    Mf=eye(T)-Pf;
    Mlambda=eye(N)-Plambda;      
    W=zeros(K,K);
    Omega=zeros(K,K);
    for k1=1:K
    for k2=1:K
      Xk1=Mlambda*squeeze(X(k1,:,:))*Mf;
      Xk2=Mlambda*squeeze(X(k2,:,:))*Mf;
      W(k1,k2)=1/N/T*trace(Mf*squeeze(X(k1,:,:))'*Mlambda*squeeze(X(k2,:,:)));  %Hessian
      Omega(k1,k2)=1/N/T*(Xk1(:).*Xk2(:))'*(res(:).^2); %Variance of Score
      Omega2(k1,k2)=1/N/T*trace(trunc((res.*Xk1)'*(res.*Xk2),M2+1,M2+1));
    end
    end
    sigma2=trace(res'*res)/N/T;
    Vbeta1=inv(W)*sigma2/N/T;
    Vbeta2=inv(W)*Omega*inv(W)/N/T;
    Vbeta3=inv(W)*Omega2*inv(W)/N/T;
  end
  
  
  if nargout>7   %user also asked for calculation of bias estimators
    for k=1:K
      XX=squeeze(X(k,:,:));  
      B1(k)=1/sqrt(N*T)*trace(Pf*trunc(res'*XX,0,M1+1));
      B2(k)=1/sqrt(N*T)*trace(XX'*Mlambda*trunc(res*res',1,1)*lambda*inv(lambda'*lambda)*inv(f'*f)*f');
      B3(k)=1/sqrt(N*T)*trace(trunc(res'*res,M2+1,M2+1)*Mf*XX'*lambda*inv(lambda'*lambda)*inv(f'*f)*f'); 
    end
    den=zeros(K);
    for k1=1:K
    for k2=1:K
      den(k1,k2)=1/N/T*trace(Mf*squeeze(X(k1,:,:))'*Mlambda*squeeze(X(k2,:,:)));
    end
    end
    bcorr1=-inv(den)*B1'/sqrt(N*T);
    bcorr2=-inv(den)*B2'/sqrt(N*T);
    bcorr3=-inv(den)*B3'/sqrt(N*T); 
  end

return;

function [beta,obj,ef]=minimize_obj_method1(Y,X,R,st,precision_beta)
  %INPUT: Y  = NxT
  %       X  = KxNxT
  %       st = Kx1 ... starting value for optimization over regression paramter beta
  %       precision_beta = defines stopping criteria, namely optimization
  %                        is stopped when difference in beta after one
  %                        optimization step is smaller than precision_beta
  %                        (uniformly over all K components of beta)
  %OUTPUT: beta = Kx1 ... optimal beta that was found
  %        obj  = value of LS-objective function at optimal beta
  %        ef (exitflag) = 1  if procedure properly terminated (according
  %                           to "precision_beta" criteria)
  %                      = -1 if procedure failed to converge (namely if
  %                           objective function did not improve during
  %                           last step --- in principle the procedure
  %                           could still converge afterwards, but if the
  %                           objective function does not improve, then it
  %                           seems more promising to stop the optimization
  %                           and restart with a different starting value)
 
  N=size(Y,1);
  T=size(Y,2);
  K=size(X,1);
  
  SST=trace(Y*Y')/N/T;
  
  beta=st;  %starting value for beta-minimization;
  beta_old=st+inf;
  obj=inf;
  diff_obj=-inf;
  while ( max(abs(beta-beta_old))>precision_beta ) && (diff_obj<=SST*10^-10)
    %two stopping criteria for iteration of STEP 1 and STEP 2 below:
    %(1) stop if each component of "beta-beta_old" is smaller than "precision_beta"
    %    ==> this is the good case when we have found a proper local minimum
    %(2) stop if diff_obj>SST*10^-8, i.e. if we made no progress in the objective
    %    function during the last iteration
    %    ==> this is the bad case, where the iteration with that particular
    %    starting value is likely not to converge
      
    %---- STEP 1: CALCULATE FACTORS AND FACTOR LOADINGS FOR GIVEN beta BY PRINCIPAL COMPONENTS: ----  
    res=get_residuals(Y,X,beta);
    [lambda,f]=principal_components(res,R);
    res=res-lambda*f';        %residuals after subtracting lambda*f'
    obj_old=obj;              %save old objective function
    obj=trace(res'*res)/N/T;  %LS objective function
    diff_obj=obj-obj_old;     %we hopefully make progress in minimizing objective function, 
                              %i.e. "diff_obj" should better be negative

    if diff_obj<=0                                       
    %---- STEP 2: CALCULATE OLS ESTIMATOR FROM REGRESSING M_lambda*Y*M_f (or just Y) on M_lambda*X*M_f: ----  
      YY=Y(:);                      %flatten Y, i.e. YY now NTx1 vector
         %alternatively, we could define YY as follows, but it should not
         %matter:
            %YY=Y-lambda*(lambda\Y);        %project lambda out of Y
            %YY=( YY'-f*(f\YY') )';         %project f out of Y
            %YY=YY(:);                      %flatten Y, i.e. YY now NTx1 vector            
      for k=1:K
        xx=squeeze(X(k,:,:));        
        xx=xx-lambda*(lambda\xx);    %project lambda out of X
        xx=( xx'-f*(f\xx') )';       %project f out of X
        XX(:,k)=xx(:);               %flatten X, i.e. XX becomes NTxK matrix
      end    
      beta_old=beta;                 %store old beta
      beta=pinv(XX'*XX)*XX'*YY;      %calculate OLS estimator
          %Here, we use the pseudo-inverse, in case XX"*XX is not invertible
          %to avoide error messages at some points of the optimization.
          %However, at the optimium we should have that beta=XX\YY.
    
    
    end
  end
  
  if  diff_obj<=0
    ef=1;       %good solution found
  else
    ef=-1;      %no good solution found
  end  
    
  obj=LS_obj(beta,Y,X,R); %calculate objective function for this beta
return   

function [para,obj,exitflag]=minimize_obj_method2(Y,X,R,st,precision_beta)
  %inputs and outputs as in "minimize_obj_method1"

  os = optimset('LargeScale','off','MaxFunEvals',10^5,'MaxIter',10^5,'TolX',precision_beta);      
  str=evalc('[para,obj,exitflag]=fminunc(@LS_obj,st,os,Y,X,R)');
return      

function [beta,obj,ef]=minimize_obj_method3(Y,X,R,st,precision_beta)
  %inputs and outputs as in "minimize_obj_method1"
 
  N=size(Y,1);
  T=size(Y,2);
  K=size(X,1);

  for k=1:K
    xx=squeeze(X(k,:,:)); 
    XX(:,k)=xx(:);     %flatten X, i.e. XX becomes NTxK matrix (needed for step 2 below)
  end   
  
  beta=st;  %starting value for beta-minimization;
  beta_old=st+inf;
  obj=0;
  diff_obj=-inf;
  i=0;
  SST=trace(Y*Y')/N/T;
  obj_save=inf*ones(1,1000); %we save the objective functions from the previous 1000 iterations
  while ( max(abs(beta-beta_old))>precision_beta ) && (abs(diff_obj)>=10^-7*SST)    
    %In case the method does not converge (i.e. if |beta-beta_old| does not 
    %become sufficiently small) we need a second stopping criterion, 
    %which here we choose relatively conservately, 
    %namely, we stop if the objectve function did not improve
    %over the last 1000 iterations by at least 10^-7*trace(Y*Y')/N/T.
      
    %---- STEP 1: CALCULATE FACTORS AND FACTOR LOADINGS FOR GIVEN beta BY PRINCIPAL COMPONENTS: ----  
    res=get_residuals(Y,X,beta);
    [lambda,f]=principal_components(res,R);
    res=res-lambda*f';        %residuals after subtracting lambda*f'
    obj=trace(res'*res)/N/T;  %LS objective function   
    obj_save(mod(i,1000)+1)=obj;
    diff_obj=obj-obj_save(mod(i+1,1000)+1); 
        %Difference between current objective fct. and objectice fct. from
        %1000 iterations ago. In this method (as opposed to method 1) this
        %difference is negative by construction.
    
    %---- STEP 2: CALCULATE OLS ESTIMATOR FROM REGRESSING Y-lambda*f' on X: ----  
    YY=Y-lambda*f';                %redisuals after proncipal components are subtracted from Y
    YY=YY(:);                      %flatten Y, i.e. YY now NTx1 vector   
    beta_old=beta;                 %store old beta
    beta=XX\YY;                    %calculate OLS estimator
  
    i=i+1; %count the number of iterations
  end
  
  if max(abs(beta-beta_old))<=precision_beta
    ef=1;       %good solution found
  else
    ef=-1;      %no good solution found
  end  
    
  obj=LS_obj(beta,Y,X,R); %calculate objective function for this beta
return       


function obj=LS_obj(beta,Y,X,R)  
   %Calculate LS objective, i.e. SSR/N/T of Y-beta*X after subtracting R
   %largest principal components.
   %INPUT: Kx1 beta, NxT Y, KxNxT X, integer R>=0
   %OUTPUT: scalar objective function
   %
   %COMMENT: within "LS_factor" it is guaranteed that T<=N, so below we
   %diagonalize a TxT matrix (not an NxN) matrix. When using this function
   %outside "LS_factor" one should check whether T<N or N>T and switch
   %dimensions accordingly, if neccessary.
   
   res=get_residuals(Y,X,beta);   
   ev=sort(eig(res'*res));
   obj=sum(ev(1:size(Y,2)-R))/size(Y,1)/size(Y,2);
return;

function res=get_residuals(Y,X,beta)
  %calculate residuals Y-beta*X
  %INPUT: Y = NxT, X = KxNxT, beta=Kx1
  %OUTPUT: res = NxT
  
  res=Y;
  for k=1:size(X,1)
    res=res-beta(k)*squeeze(X(k,:,:));
  end
return

function [lambda,f]=principal_components(res,R)   
  %Extract the "R" leading principal components out of the NxT matrix "res".
  %Output: NxR matrix lambda, TxR matrix f (=factor loadings and factors) 
  %
  %COMMENT: within "LS_factor" it is guaranteed that T<=N, so below we
  %diagonalize a TxT matrix (not an NxN) matrix. When using this function
  %outside "LS_factor" one should check whether T<N or N>T and switch
  %dimensions accordingly, if neccessary.
  
  T=size(res,2);             %time dimensions
  [V,D]=eig(res'*res);       %calculate eigenvalues and eigenvectors of TxT matrix
  [Dsort,Ind]=sort(diag(D)); %sort eigenvalues, just to be sure
  f=V(:,Ind(T-R+1:T));       %the eigenvectors corresponding to the R largest eigenvalues
  for r=1:R
    f(:,r)=f(:,r)/norm(f(:,r));  %normalize each principal component (= each factor)
    if mean(f(:,r))<0            
      f(:,r)=-f(:,r);            %f(1,r) is normalized to be positive
    end  
  end
  lambda=res*f;                  %principal components in the other dimensions (= factor loadings)
return



function AT=trunc(A,M1,M2)  
   %truncates a symmetric matrix A to the (M1+M2-1) first diagonals
   NN=size(A,1);
   AT=zeros(NN);
   for i=1:NN
     for j=max(i-M1+1,1):min(i+M2-1,NN);
       AT(i,j)=A(i,j);
     end  
   end
 return;