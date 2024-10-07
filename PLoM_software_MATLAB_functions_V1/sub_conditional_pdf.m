function  [nbpoint,Rq,Rpdfqq_ww0] = sub_conditional_pdf(mw,N,MatRqq,MatRww,Rww0,nbpoint0)      
   
   %------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 31 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_conditional_pdf
   %  Subject      : Let QQ be the real-valued random quantity of interest
   %                 Let WW  = (WW_1,...,WW_mw)   be the vector-valued control random variable
   %                 Let ww0 = (ww0_1,...,ww0_mw) be a value of the vector-valued control parameter
   %                 We consider the family of conditional real-valued random variables: QQ | WW = ww0
   %                 This function computes the conditional pdf  
   %                 pdfqq_ww0 =  pdf of {QQ | WW = ww0} estimated in nbpoint
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
   %          N                      : number of realizations
   %          MatRqq(1,N)            : N realizations of the real-valued random quantity of interest QQ
   %          MatRww(mw,N)           : N realizations of the vector-valued control random variable WW = (WW_1,...,WW_mw)
   %          Rww0(mw,1)             : a given value ww0 = (ww0_1,...,ww0_mw) of the vector-valued control parameter
   %          nbpoint0               : initial number of points from which the number of points will be used for estimating the pdf
   %                                   (for instance nbpoint0 = 200)
   %--- OUPUTS
   %          nbpoint                : number of points in which the conditional pdf is computed 
   %          Rq(nbpoint,1)          : discrete graph of the pdf: (Rq,Rpdfqq_ww0)
   %          Rpdfqq_ww0(nbpoint,1)  : Rpdfqq_ww0(j,1) is the conditional pdf of {QQ | WW = ww0} at point j = 1,...,nbpoint
   
   %--- Checking dimension for MatRqq(1,N) 
   n1temp = size(MatRqq,1);
   if n1temp ~= 1
      error('STOP in sub_conditional_pdf: the first dimension of MatRqq must be one')
   end
   
   %---  Pre-computation 
   nx     = 1 + mw;
   sx     = (4/(N*(2+nx)))^(1/(nx+4));       % Silverman bandwidth for XX = (QQ,WW)
   cox    = 1/(2*sx*sx);
   coefsx = 1/(sx*sqrt(2*pi));

   mean_qq  = mean(MatRqq,2);                % MatRqq(1,N)
   std_qq   = std(MatRqq,0,2);               % MatRqq(1,N) 
   Rmean_ww = mean(MatRww,2);                % Rmean_ww(mw,1),MatRww(mw,N) 
   Rstd_ww  = std(MatRww,0,2);               % Rstd_ww(mw,1),MatRww(mw,N) 
   if std_qq == 0
      std_qq = 1;
   end
   for k=1:mw
      if Rstd_ww(k) == 0
         Rstd_ww(k) = 1;
      end
   end

   %--- Realizations of the normalized random variable QQtilde, WWtilde(mw,1), and corresponding Rwwtilde_0(mw,1)
   MatRqqtilde     = (MatRqq - mean_qq)./std_qq;                   % MatRqqtilde(1,N),MatRqq(1,N),mean_qq,std_qq
   MatRwwtilde     = (MatRww - Rmean_ww)./Rstd_ww;                 % MatRwwtilde(mw,N),MatRww(mw,N),Rmean_ww(mw,1),Rstd_ww(mw,1)
   Rww0tilde       = (Rww0 - Rmean_ww)./Rstd_ww;                   % Rww0tilde(mw,1)  
    
   %--- Conditional pdf of QQ|WW=Rww_0 in nbpoint loaded in Rpdfqq_ww0(nbpoint,1)
        % maxq            = max(MatRqq, [], 2);                           % MatRqq(1,N)
        % minq            = min(MatRqq, [], 2);                           % MatRqq(1,N)
        % delta           = (maxq - minq);
        % coeff           = 0.05;
        % maxq            = maxq + coeff*delta;
        % minq            = minq - coeff*delta;
   coeff = 5;
   maxq  = mean_qq + coeff*std_qq;
   minq  = mean_qq - coeff*std_qq;
   MatRabscq       = linspace(minq,maxq,nbpoint0);                 % MatRabscq(1,nbpoint0)    
   nbpoint         = size(MatRabscq,2);                            % MatRabscq(1,nbpoint)
   Rq              = MatRabscq';                                   % Rq(nbpoint,1)
   MatRexpo        = MatRwwtilde - repmat(Rww0tilde,1,N);          % MatRexpo(mw,N),MatRwwtilde(mw,N),Rww0tilde(mw,1) 
   MatRtempw       = cox*(sum(MatRexpo.^2,1));                     % MatRtempw(1,N) 
   MatRS           = exp(-MatRtempw);                              % MatRS(1,N) 
   den             = sum(MatRS,2); 
   Rpdfqq_ww0      = zeros(nbpoint,1);                             % Rpdfqq_ww0(nbpoint,1)
   MatRabscqtilde  = (MatRabscq - mean_qq)/std_qq;                 % MatRabscqtilde(1,nbpoint);
   for ib = 1:nbpoint
       qtilde_ib        = MatRabscqtilde(1,ib);                    % MatRabscqtilde(1,nbpoint);
       MatRexpo_ib      = MatRqqtilde - repmat(qtilde_ib,1,N);     % MatRexpo_ib(1,N),MatRqqtilde(1,N)
       MatRS_ib         = exp(-MatRtempw -cox*(MatRexpo_ib.^2));   % MatRS_ib(1,N),MatRtempw(1,N),MatRexpo_ib(1,N)
       num_ib           = sum(MatRS_ib,2);                         % MatRS_ib(1,N)
       Rpdfqq_ww0(ib,1) = (coefsx/std_qq)*num_ib/den;              % Rpdfqq_ww0(nbpoint,1)
   end
   return 
end