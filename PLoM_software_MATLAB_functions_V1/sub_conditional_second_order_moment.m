function  [REqq_ww0,REqq2_ww0] = sub_conditional_second_order_moment(mw,mq,N,MatRqq,MatRww,Rww0)      
   
   %------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 31 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_conditional_second_order_moment
   %  Subject      : Let QQ  = (QQ_1,...,QQ_mq)   be the vector-valued random quantity of interest
   %                 Let WW  = (WW_1,...,WW_mw)   be the vector-valued control random variable
   %                 Let ww0 = (ww0_1,...,ww0_mw) be a value of the vector-valued control parameter
   %                 We consider the family of conditional real-valued random variables: QQ_k | WW = ww0,  for k = 1, ..., mq
   %                 For all k = 1,...,mq, this function computes, for k = 1, ..., mq,
   %                       - the conditional mean value           Eqq_ww0_k  =  E{QQ_k   | WW = ww0}
   %                       - the conditional second-order moement Eqq2_ww0_k =  E{QQ_k^2 | WW = ww0}
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
   %
   %--- OUTPUTS
   %          REqq_ww0(mq,1)         : For k=1,...,mq, REqq_ww0(k,1) is the conditional mean value E{QQ_k | WW = ww0}
   %          REqq2_ww0(mq,1)        : For k=1,...,mq, REqq2_ww0(k,1) is the conditional second-order moement E{QQ_k^2 | WW = ww0}
  
   
   %---  Pre-computation 
   nx  = mq + mw;
   sx  = (4/(N*(2+nx)))^(1/(nx+4));          % Silverman bandwidth for XX = (QQ,WW)
   cox = 1/(2*sx*sx);

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

   % Realizations of the normalized random variable QQtilde(mq,1), WWtilde(mw,1), and corresponding Rwwtilde_0(mw,1)
   MatRqqtilde   = (MatRqq - Rmean_qq)./Rstd_qq;        % MatRqqtilde(mq,N),MatRqq(mq,N),Rmean_qq(mq,1),Rstd_qq(mq,1)
   clear MatRqq
   MatRwwtilde   = (MatRww - Rmean_ww)./Rstd_ww;        % MatRwwtilde(mw,N),MatRww(mw,N),Rmean_ww(mw,1),Rstd_ww(mw,1)
   Rwwtilde0     = (Rww0 - Rmean_ww)./Rstd_ww;
  
   %--- Conditional mean Eqq_ww0_k = E{QQ_k | WW = ww0} and second-order moment Eqq2_ww0_k = E{QQ_k^2 | WW = ww0}, for k = 1,...,mq  
   MatRexpo  = MatRwwtilde - Rwwtilde0;                 % MatRexpo(mw,N),MatRwwtilde(mw,N),Rwwtilde0(mw,1)
   MatRS     = exp(-cox*(sum(MatRexpo.^2,1)));          % MatRS(1,N)
   den       = sum(MatRS,2); 
   Rnum      = sum(MatRqqtilde.*MatRS,2);               % MatRqqtilde(mq,N),MatRS(1,N)  
   Rnum2     = sum((MatRqqtilde.^2).*MatRS,2);          % MatRqqtilde(mq,N),MatRS(1,N)  
   Rtemp     = Rstd_qq.*Rnum/den;
   REqq_ww0  = Rmean_qq + Rtemp;                                                     % REqq_ww0(mq,1)
   REqq2_ww0 = Rmean_qq.^2 + 2*Rmean_qq.*Rtemp + (Rstd_qq.^2).*(sx^2 + Rnum2/den);   % REqq2_ww0(mq,1)  
   return 
end
