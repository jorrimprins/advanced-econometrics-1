function [LN,scoreN] = loglikNormalHetsk(thetaloc,yloc,Xloc,Zloc)
% minus loglikelihood for Normal regression model with heteroskedasticity,
%  with analytical score vector
% ...loc stands for local variable

[Nloc kloc]= size(Xloc);   %  determine N k and m from the inputmatrices
[dum mloc]= size(Zloc);

  betaloc = thetaloc(1:kloc);    % theta  = (beta', sigma2, gamma')'
sigma2loc = thetaloc(kloc+1);
 gammaloc = thetaloc((kloc+2):(kloc+mloc+1));

varyinv = exp(-Zloc*gammaloc)/sigma2loc;     %  these are the diagonal elements of the covariance matrix
eloc=(yloc-Xloc*betaloc);                    %  residuals based on beta

LN = 0.5*(eloc'*(varyinv.*eloc)) +0.5*Nloc*log(sigma2loc)+ 0.5*sum(Zloc*gammaloc);  % minus the log likelihood

%  in case gradients are used: the analytic gradients are:
if nargout > 1
   scoreN = [-Xloc'*(varyinv.*eloc);
       -0.5*eloc'*(varyinv.*eloc)/sigma2loc+0.5*Nloc/sigma2loc;
       -0.5*Zloc'*(eloc.*(varyinv.*eloc))+ 0.5*sum(Zloc)'];
end

end

