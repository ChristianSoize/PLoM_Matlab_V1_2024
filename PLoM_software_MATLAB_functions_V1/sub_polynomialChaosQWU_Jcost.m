function [J] = sub_polynomialChaosQWU_Jcost(MatRsample,n_y,Jmax,NnbMC0,n_q,nbqqC,nbpoint,MatRgamma_bar,MatRchol,MatRb,Ralpha_scale_yy, ...
                                 RQQmean,MatRVectEig1s2,Ind_qqC,MatRqq_ar0C,MatRxipointOVLC,RbwOVLC,MatRones) 
                                           
%----------------------------------------------------------------------------------------------------------------------------------------------
%          Copyright C. Soize, 05 October 2024
%          OVERLAPPING COEFFICIENT
%----------------------------------------------------------------------------------------------------------------------------------------------

% WARNING: since a minimization algorithm is used and since we want to maximize the overlap;
%          the signe "-" (minus) is introduced
%---- INPUT
%         MatRsample(n_y,Jmax)
%         MatRgamma_bar(n_y,Jmax)
%         MatRchol(n_y,n_y)
%         MatRb(Jmax,NnbMC0)  
%         Ralpha_scale_yy(n_y,1)
%         RQQmean(n_q,1)
%         MatRVectEig1s2(n_q,n_y) with n_y = n_q
%         Ind_qqC(nbqqC,1)
%         MatRqq_ar0C(nbqqC,NnbMC0) 
%         MatRxipointOVLC(nbqqC,nbpoint) 
%         RbwOVLC(nbqqC,1)
%         MatRones(n_y,Jmax) = ones(n_y,Jmax)

%--- OUTPUT
%         J = cost function

MatRtilde  = MatRgamma_bar.*(MatRones + MatRsample);            % MatRtilde(n_y,Jmax),MatRgamma_bar(n_y,Jmax),MatRsample(n_y,Jmax) 
MatRFtemp  = chol(MatRtilde*(MatRtilde'));                      % MatRFtemp(n_y,n_y)
MatRAtemp  = (inv(MatRFtemp))';                                 % MatRAtemp(n_y,n_y)
MatRhat    = MatRAtemp*MatRtilde;                               % MatRhat(n_y,Jmax),MatRtilde(n_y,Jmax)
MatRgamma  = MatRchol'*MatRhat;                                 % MatRgamma(n_y,Jmax)
                                                                      
MatRy      = MatRgamma*MatRb;                                   % MatRyy(n_y,NnbMC0),MatRy(n_y,NnbMC0),MatRgamma(n_y,Jmax),MatRb(Jmax,NnbMC0)          
MatRyy     = Ralpha_scale_yy.*MatRy;                            % Ralpha_scale_yy(n_y,1),MatRVectEig1s2(n_q,n_y)
MatRqq     = repmat(RQQmean,1,NnbMC0)+MatRVectEig1s2*MatRyy;        
                                                                % MatRqq_ar0(n_q,NnbMC0),MatRqq(n_q,NnbMC0),Ind_qqC(nbqqC,1)
MatRqqC    = MatRqq(Ind_qqC,:);
[RerrorOVL] = sub_polynomialChaosQWU_OVL(nbqqC,NnbMC0,MatRqq_ar0C,NnbMC0,MatRqqC,nbpoint,MatRxipointOVLC,RbwOVLC);
J  = - sum(RerrorOVL)/nbqqC; 
return
      