function [HessianHo,scoreN] = HessianNormalLLhet(thetaloc,yloc,Xloc,Zloc)
% 
% ...loc stands for local variable

[Nloc kloc]= size(Xloc);   %  determine N k and m from the inputmatrices
[dum mloc]= size(Zloc);

  betaloc = thetaloc(1:kloc);    % theta  = (beta', sigma2, gamma')'
sigma2loc = thetaloc(kloc+1);
 gammaloc = thetaloc((kloc+2):(kloc+mloc+1));
iZloc=Nloc*mean(Zloc);

% note that the X'X and Z'Z blocks do not change over replications
 HessianHo =[ Xloc'*Xloc/sigma2loc, zeros(kloc,mloc+1);   
             zeros(1,kloc), 0.5*Nloc/sigma2loc^2,iZloc/(2*sigma2loc);
             zeros(mloc,kloc),iZloc'/(2*sigma2loc),Zloc'*Zloc/2];
 

%  in case gradients are used: the analytic gradients are:
if nargout > 1
 varyinv = exp(-Zloc*gammaloc)/sigma2loc;     %  these are the diagonal elements of the covariance matrix
 eloc=(yloc-Xloc*betaloc);                    %  residuals based on beta
  scoreN = [-Xloc'*(varyinv.*eloc);
       -0.5*eloc'*(varyinv.*eloc)/sigma2loc+0.5*Nloc/sigma2loc;
       -0.5*Zloc'*(eloc.*(varyinv.*eloc))+ 0.5*sum(Zloc)'];
end
% if nargout > 2  % but then HessN should be given as part of the output
%    HessN =;  
% end
end