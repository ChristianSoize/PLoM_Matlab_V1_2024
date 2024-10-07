function [RqqLower_ww0,RqqUpper_ww0] = sub_conditional_confidence_interval(mw,mq,N,MatRqq,MatRww,Rww0,pc)

%------------------------------------------------------------------------------------------------------------------------------------
%
%  Copyright: Christian Soize, Universite Gustave Eiffel, 31 May 2024
%
%  Software     : Probabilistic Learning on Manifolds (PLoM) 
%  Function name: sub_conditional_confidence_interval
%  Subject      : Let QQ  = (QQ_1,...,QQ_mq)   be the vector-valued random quantity of interest
%                 Let WW  = (WW_1,...,WW_mw)   be the vector-valued control random variable
%                 Let ww0 = (ww0_1,...,ww0_mw) be a value of the vector-valued control parameter
%                 We consider the family of conditional real-valued random variables: QQ_k | WW = ww0,  for k = 1, ..., mq
%                 For a given value pc of the probability level, and for all k = 1,...,mq, this function computes 
%                 the lower bound  qqLower_ww0_k  and the upper bound  qqUpper_ww0_k  of the conditional confidence interval of 
%                 QQ_k | WW = ww0,  for k = 1, ..., mq.
%
%  Publications: 
%               [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
%                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
%               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
%                         American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020).   
%               [3] C. Soize, R. Ghanem, Probabilistic-learning-based stochastic surrogate model from small incomplete datasets 
%                         for nonlinear dynamical systems, Computer Methods in Applied Mechanics and Engineering, 
%                         doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024). 
%               [ ] For the conditional statistics formula, see the Appendix of paper [3]
%
%
%--- INPUTS
%          mw                     : dimension of MatRww(mw,N) and Rww_0(mw,1) 
%          mq                     : dimension of MatRqq(mq,N)
%          N                      : number of realizations
%          MatRqq(mq,N)           : N realizations of the vector-valued random quantity of interest QQ = (QQ_1,...,QQ_mq)
%          MatRww(mw,N)           : N realizations of the vector-valued control random variable WW = (WW_1,...,WW_mw)
%          Rww0(mw,1)             : a given value ww0 = (ww0_1,...,ww0_mw) of the vector-valued control parameter
%          pc                     : the probability level (for instance 0.98)
%
%--- OUPUTS
%          RqqLower_ww0(mq,1)     : For k=1,...,mq, RqqLower_ww0(k,1) is the lower bound  qqLower_ww0_k  of the conditional 
%                                   confidence interval of QQ_k | WW = ww0.   
%          RqqUpper_ww0(mq,1)     : For k=1,...,mq, RqqUpper_ww0(k,1) is the upper bound  qqUpper_ww0_k  of the conditional 
%                                   confidence interval of QQ_k | WW = ww0.   

%--- Pre-computation   
nx       = mq + mw;
sx       = (4/(N*(2+nx)))^(1/(nx+4));     % Silverman bandwidth for XX = (QQ,WW)
cox      = 1/(2*sx*sx);
co2x     = 1/(sqrt(2)*sx);

Rmean_qq = mean(MatRqq,2);                % Rmean_qq(mq,1),MatRqq(mq,N)
Rstd_qq  = std(MatRqq,0,2);               % Rstd_qq(mq,1),MatRqq(mq,N) 
Rmean_ww = mean(MatRww,2);                % Rmean_ww(mw,1),MatRww(mw,N) 
Rstd_ww  = std(MatRww,0,2);               % Rstd_ww(mw,1),MatRww(mw,N) 
for k=1:mq
   if Rstd_qq(k) == 0
      Rstd_qq(k) = 1;
   end
end
for k=1:mw
   if Rstd_ww(k) == 0
      Rstd_ww(k) = 1;
   end
end

Rmaxq = max(MatRqq, [], 2);                                         % MatRqq(mq,N), Rmaxq(mq,1)
Rminq = min(MatRqq, [], 2);                                         % MatRqq(mq,N), Rminq(mq,1)   

% Realizations of the normalized random variable QQtilde(mq,1), WWtilde(mw,1), and corresponding Rwwtilde_0(mw,1)
MatRqqtilde   = (MatRqq - Rmean_qq)./Rstd_qq;  % MatRqqtilde(mq,N),MatRqq(mq,N),Rmean_qq(mq,1),Rstd_qq(mq,1)
clear MatRqq
MatRwwtilde   = (MatRww - Rmean_ww)./Rstd_ww;  % MatRwwtilde(mw,N),MatRww(mw,N),Rmean_ww(mw,1),Rstd_ww(mw,1)
Rwwtilde0    = (Rww0 - Rmean_ww)./Rstd_ww;

%--- Conditional cdf of QQ_k | RWW = Rww0  for k = 1,...,mq  
nbpoint       = 200;
coeff         = 0.2;
MatRqstar     = zeros(mq,nbpoint);    % MatRqstar(mq,nbpoint): nbpoint qstar points in abscissa for computation of the cond cdf
for iq = 1:mq    
    maxq  = Rmaxq(iq);
    minq  = Rminq(iq);
    delta = (maxq - minq);
    maxq  = maxq + coeff*delta;
    minq  = minq - coeff*delta;
    MatRqstar(iq,:) = linspace(minq,maxq,nbpoint);                                    % MatRqstar(mq,nbpoint)    
end
                                                                                       % MatRqstartilde(mq,nbpoint),MatRqstar(mq,nbpoint)
MatRqstartilde = (MatRqstar - Rmean_qq)./Rstd_qq; % Rmean_qq(mq,1),Rstd_qq(mq,1)
MatRexpo       = MatRwwtilde - Rwwtilde0;         % MatRexpo(mw,N),MatRwwtilde(mw,N),Rwwtilde0(mw,1)
MatRS          = exp(-cox*(sum(MatRexpo.^2,1)));  % MatRS(1,N)
den            = sum(MatRS,2);
   
ArrayB = repmat(MatRqqtilde,1,1,nbpoint);         % ArrayB(mq,N,nbpoint),MatRqqtilde(mq,N) 
clear MatRqqtilde
ArrayB = permute(ArrayB,[1 3 2]);                 % ArrayB(mq,nbpoint,N)
ArrayB = ArrayB - repmat(MatRqstartilde,1,1,N);
clear MatRqstartilde
ArrayB = (-co2x)*ArrayB;
ArrayB = erf(ArrayB);
ArrayB = 0.5*ArrayB + 0.5;
MatRB  = reshape(ArrayB,mq*nbpoint,N);            % MatRB(mq*nbpoint,N)
clear ArrayB
MatRA  = MatRB*MatRS.';                           % MatRA(mq*nbpoint,1),MatRB(mq*nbpoint,N),MatRS(1,N)
clear MatRS
MatRcdfcondTemp = reshape(MatRA,mq,nbpoint);      % MatRcdfcondTemp(mq,nbpoint)
MatRcdfcond     = MatRcdfcondTemp.'/den;          % MatRcdfcond(nbpoint,mq)
clear MatRcdfcondTemp

%--- Computing the confidence interval of QQ_k | WW = ww_0 for k = 1,...,mq and given p_c
RqqLower_ww0 = zeros(mq,1);
RqqUpper_ww0 = zeros(mq,1);
for iq = 1:mq
    Rcdfcond_iq      = MatRcdfcond(:,iq);                    % MatRcdfcond(nbpoint,mq) 
    [~,ib_upper]     = min(abs(Rcdfcond_iq - pc));
    [~,ib_lower]     = min(abs(Rcdfcond_iq - (1 - pc)));
    RqqLower_ww0(iq) = MatRqstar(iq,ib_lower);               % MatRqstar(mq,nbpoint)  
    RqqUpper_ww0(iq) = MatRqstar(iq,ib_upper);
end   
return
end
