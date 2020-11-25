%
% This programme simulates test statistics for heteroskedasticity in the CLRM
% It then plots size-adjusted power curves for these test statistics and
% shows confidence intervals

%% PARAMETER SPECIFICATION
clc; clear;  % start with a clean sheet
Rep = 1000; % number of Monte Carlo replications,
%Not more because that takes a lot of time and does not really improve the results
seed = 12347; 
rng(seed);  % set random seed
N = 15; % nr of observations
k = 2;      % nr of variables in X
m = 2;      % nr of variables in Z

zoom = 0; %0 for gamma range -1 to 1 and 1 for gamma range 0.6 to 1
% CLRM parameters
beta    = [1;1]; 
sigma   = 0.9;
sigma2  = sigma^2;

if zoom == 0
    gammalow  = -1; 
    steps = 0.1;
else
    gammalow = 0.6;
    steps = 0.05;
end
gammahigh =  1;
gammalternatives = gammalow:steps:gammahigh; % set range of gamma alternatives for power curve
nralternatives = length(gammalternatives);
PowerCurve = zeros(nralternatives,4); % one column for every test

%% FOR LOOP FOR POWER CURVES
for i=1:nralternatives
%loop over gamma alternatives
% set gamma_DGP under alternative hypothesis and 
% percentage of Ho rejections is power
gamma_DGP = gammalternatives(i); % degree of heteroskedasticity
gammaDGP  = gamma_DGP*ones(m,1);   % make it a vector since Z:Nxm 
 % with this specification both Z's influence the variance equally 

CVasymp = chi2inv(0.95,m);    %define the asymptotic critical value

%CRITICAL VALUES FOR LOWER N
    if N == 15;
        CVWald  = 11.7208;
        CVLR    = 9.4026;
        CVLM    = 4.4525;
        CVLMOPG = 4.363;
    elseif N==25;
        CVWald  = 8.7878;
        CVLR    = 8.3173;
        CVLM    = 5.0674;
        CVLMOPG = 5.4544;
    elseif N==100;
        CVWald  = 6.1071;
        CVLR    = 6.1353;
        CVLM    = 5.7765;
        CVLMOPG = 5.9306;
    end
%end
    
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
    y = X*beta + std.*u;           % CLRM with heteroskedastic errors if gammaDGP=/=0
    
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

PowerCurve(i,1) = mean(Waldtest>CVWald);
PowerCurve(i,2) = mean(LRtest >CVLR);
PowerCurve(i,3) = mean(LMtest >CVLM);
PowerCurve(i,4) = mean(LMOPGtest >CVLMOPG);
end

%% PRINT REJECTION PROBABILITIES UNDER ADJUSTED CRITICAL VALUES
disp([ 'Average values of the test statistics']);
disp([mean(Waldtest) mean(LRtest) mean(LMtest) mean(LMOPGtest)]);

disp(['Critical values used      ', num2str([CVWald,CVLR,CVLM,CVLMOPG]), '(Size adjusted)']);
disp('Monte Carlo rejection probabilities, 5% nominal level');
disp(['Wald             : ' num2str(100*mean(Waldtest>CVWald)) '%']);
disp(['LR               : ' num2str(100*mean(LRtest >CVLR)) '%']);
disp(['LM               : ' num2str(100*mean(LMtest >CVLM)) '%']);
disp(['LM_OPG version   : ' num2str(100*mean(LMOPGtest >CVLMOPG)) '%']);


%% PLOT POWER CURVES FOR ALL TESTS TOGETHER AND FOR SEPARATE TESTS
if zoom == 0
    plot(gammalternatives, PowerCurve)   % plot the powercurve of 4 tests combined
    legend('Wald','LR','LM','LM OPG')
    disp([ ' H0:gamma=0  =/= gamma Ha ='  num2str(gamma_DGP)]);
    figure
        plot(gammalternatives, PowerCurve(:,2))   % plot the powercurve of LR test
    legend('LR')
    axis([-1 1 0 1])
    figure
            plot(gammalternatives, PowerCurve(:,1))   % plot the powercurve of LR test
    legend('Wald')
    axis([-1 1 0 1])
    figure
        plot(gammalternatives, PowerCurve(:,3))   % plot the powercurve of LM test
    legend('LM')
    axis([-1 1 0 1])
    figure
        plot(gammalternatives, PowerCurve(:,4))   % plot the powercurve of LM test
    legend('LM OPG')
    axis([-1 1 0 1])
else
        plot(gammalternatives, PowerCurve(:,(2:3)))   % plot the powercurve of 4 tests combined
    legend('LR','LM')
    disp([ ' H0:gamma=0  =/= gamma Ha ='  num2str(gamma_DGP)]);
    figure
        plot(gammalternatives, PowerCurve(:,2))   % plot the powercurve of LR test
    legend('LR')
    axis([0.6 1 0 1])
    figure
            plot(gammalternatives, PowerCurve(:,1))   % plot the powercurve of LR test
    legend('Wald')
    axis([0.6 1 0 1])
    figure
        plot(gammalternatives, PowerCurve(:,3))   % plot the powercurve of LM test
    legend('LM')
    axis([0.6 1 0 1])
    figure
        plot(gammalternatives, PowerCurve(:,4))   % plot the powercurve of LM test
    legend('LM OPG')
    axis([0.6 1 0 1])
end


%% CONFIDENCE REGIONS (we do not use these in our analysis

QWald = @(gam1,gam2,I22inv) [(gammahat(1)-gam1) (gammahat(2)-gam2)]*I22inv*[(gammahat(1)-gam1) (gammahat(2)-gam2)]';

gam1grid= -1.0:0.05:1.5;   lg1=length(gam1grid); 
gam2grid= -1.0:0.05:1.5;   lg2=length(gam2grid);

gam1grid = gammahat(1)+gam1grid;
gam2grid = gammahat(2)+gam2grid;

QW=zeros(lg1,lg2);
 
for ii=1:lg1
    for jj=1:lg2
        QW(ii,jj) = QWald(gam1grid(ii),gam2grid(jj),Iunr22inv);
    end
end
cntrlevels = [0.01, CVasymp, CVWald, QWald(0,0,Iunr22inv)];

 figure
%  color stuff
        caxis([-5 15]);
        colormap('summer'); 
        brighten(0.8);

[C,h] = contourf(gam2grid,gam1grid,QW,cntrlevels,'--','ShowText','on');
title('Confidence Region Gamma');

figure
ezcontour(@(g1,g2)QWald(g1,g2,Iunr22inv),[-1,1,-1,1],100);


 fprintf('    Monte Carlo based on %3d replications \n', Rep);
 fprintf('    Parameter Values :   N = %4d;  beta = (%3.1f,%3.1f)^T; sigm2 = %4.2f; gamma = (%3.1f,%3.1f)^T \n \n', N,beta,sigma2,gammaDGP);

 
