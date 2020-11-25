%
% This programme simulates test statistics for heteroskedasticity in the CLRM
% It also plots the distribution of the test statistics and their
% asymptotic distribution
% It then calculates size-adjusted critical values to create power curves
% in another script

%% PARAMETER SPECIFICATION
clc; clear;  % start with a clean sheet
Rep = 1000; % number of Monte Carlo replications
%Not more because that takes a lot of time and does not really improve the results
seed = 12347; 
rng(seed);  % set random seed
N = 1000; % nr of observations
k = 2;      % nr of variables in X
m = 2;      % nr of variables in Z

% CLRM parameters
beta    = [1;1]; 
sigma   = 0.9;
sigma2  = sigma^2;

%% SPECIFY CLRM
% set gamma_DGP at 0 for  Ho to find size and adjusted critical values
gamma_DGP = 0.0;
gammaDGP  = gamma_DGP*ones(m,1);   % make it a vector since Z:Nxm 
 % with this specification both Z's do not influence the variance 

CVasymp = chi2inv(0.95,m);    %define the asymptotic critical value

%CRITICAL VALUES FOR ASYMPTOTIC N
       CVWald = CVasymp;
       CVLR   = CVasymp; 
       CVLM   = CVasymp;
       CVLMOPG= CVasymp;

%SPECIFY CLRM
iota = ones(N,1);  % col of ones
CovZhalf = [1 0.0; 0.0 1];       % contemporeneous correlation in Z

x = normrnd(0,1,N,k-1);       % define normally distributed random x
X = [iota x];                 % include constant in X matrix
Z = normrnd(0,1,N,m)*CovZhalf; % define normally distributed random Z
Zav = mean(Z)';
Z = Z - iota*Zav';             % move Z matrix to zero-mean
     
ZMiZ=Z'*Z-N*(Zav*Zav');
ZMZinv= inv(ZMiZ);

std = sigma*exp(Z*gammaDGP/2); % std of u, heteroskedastic if gammaDGP=/=0
  
% Creating matrices to collect the results of replications
    LRtest    = zeros(Rep,1);               
    Waldtest  = zeros(Rep,1);              
    LMtest    = zeros(Rep,1);              
    LMOPGtest = zeros(Rep,1);

 % options for optimization procedure fminunc
  options = optimset('LargeScale','off', ...   % options for optimization
               'TolFun',1e-10, ...             
               'Hessian','off', ...            % Use analytic Hessian
               'HessUpdate','bfgs', ...        % algorithm to update the Hessian
               'GradObj','off', ...            % Use analyic derivatives
               'DerivativeCheck','off', ...    % check derivative off
               'Algorithm','quasi-newton', ... % set algorithm
               'Display','off');             % print iterations to screen

%MONTE CARLO SIMULATIONS
for j=1:Rep
    u = normrnd(0,std,N,1);   % N disturbance terms with std as earlier defined 
    y = X*beta + u;           % CLRM with heteroskedastic errors if gammaDGP=/=0
    
    bhatrestr= X\y;                     % OLS estimate
    erestr = y-X*bhatrestr;             % OLS residual
    sigma2restr = erestr'*erestr/(N-k); % OLS sigma
    thetarestr = [bhatrestr;sigma2restr;zeros(m,1)];
    
    theta0start =  thetarestr ;
   
  %Estimate unrestricted model with Maximum Likelihood
    
    [thetaML,loglik,exitflg,outpt,mygrad,myhess] = fminunc(@(thet)loglikNormalHetsk(thet,y,X,Z),theta0start,options);
    thetaunrestr = thetaML;
    gammahat= thetaML(k+2:k+m+1);  % the last m elements of (beta' sigma2 gamma')'
    invmyhess = inv(myhess);
    Iunr22inv =inv(invmyhess(k+2:k+m+1,k+2:k+m+1)); % lower mxm matrix of hessian (for gamma's)
  
  %Calculate restricted score and hessian for the LM test
  [HessNrestr,scorerestr] = HessianNormalLLhet(thetarestr,y,X,Z);
  
   % Wald Test  
   Waldtest(j) = gammahat'*Iunr22inv*gammahat;
      
   % LR test
   likunrestr = -loglik;
   likrestr   = -loglikNormalHetsk(thetarestr,y,X,Z);
   LRtest(j)=2*(likunrestr-likrestr);

   % LM test
   % Hessian based
   LMtest(j) = scorerestr'*inv(HessNrestr)*scorerestr;
   
   % OPG based
   f = iota-(erestr.*erestr)/sigma2restr;
   LMOPGtest(j) = 0.5*f'*Z*ZMZinv*Z'*f;
   
end

% Rejection probabilities for the 4 tests using the specified critical values
rejectP = 100*[mean(Waldtest>CVWald) mean(LRtest >CVLR) mean(LMtest >CVLM) mean(LMOPGtest >CVLMOPG)];

%% PRINT REJECTION PROBABILITIES UNDER ASYMPTOTIC CRITICAL VALUES (SIZE)
disp([ 'Average values of the test statistics']);
disp([mean(Waldtest) mean(LRtest) mean(LMtest) mean(LMOPGtest)]);

disp(['Critical value used for all tests     ', num2str(CVasymp)]);
disp('Monte Carlo rejection probabilities, 5% nominal level');
disp(['Wald             : ' num2str(100*mean(Waldtest>CVWald)) '%']);
disp(['LR               : ' num2str(100*mean(LRtest >CVLR)) '%']);
disp(['LM               : ' num2str(100*mean(LMtest >CVLM)) '%']);
disp(['LM_OPG version   : ' num2str(100*mean(LMOPGtest >CVLMOPG)) '%']);


%% CALCULATE ADJUSTED SIZES FOR N VALUES
%  the true value = hypothesized value : H0 is true DGP
% calculating the MC based critical value, sort result and take 95% element
% getting the adjusted critical values under H0 that we 
% need for comparing size adjusted power of the tests
     Waldtest  = sort(Waldtest);
     LRtest    = sort(LRtest);
     LMtest    = sort(LMtest);
     LMOPGtest = sort(LMOPGtest);
 indx95 = round(0.95*Rep);
    disp(['Asymptotic critical value ' num2str(CVasymp)]);
    fprintf('    Monte Carlo based 5 percent critical values based on %3d replications \n', Rep)
    disp(['    CVWald  = ' num2str(Waldtest(indx95)) ';']);
    disp(['    CVLR    = ' num2str(LRtest(indx95)) ';']);
    disp(['    CVLM    = ' num2str(LMtest(indx95)) ';']);
    disp(['    CVLMOPG = ' num2str(LMOPGtest(indx95)) ';']);
 
 %% PLOT DISTRIBUTION OF TEST STATISTICS AND CHI^2(2)
 
 Xpoints=linspace(0,10.0,100);
 
 Chisqdata = chi2rnd(2,[Rep,1]);
 kd_Chisq = ksdensity(Chisqdata, Xpoints);
 kd_Waldtest =ksdensity(Waldtest,Xpoints);
 kd_LRtest   =ksdensity(LRtest,Xpoints);
 kd_LMtest   =ksdensity(LMtest,Xpoints);
 kd_LMOPGtest=ksdensity(LMOPGtest,Xpoints);
 
plot(Xpoints,[kd_Chisq',kd_Waldtest',kd_LRtest',kd_LMtest',kd_LMOPGtest']');
xlabel('test statistic');
ylabel('pdf');
legend('Chi^2','Wald','LR','LM','LMOPG');
 
 