function [entropy] = sub_solverDirect_Entropy(MatRx,ind_parallel)

   %--------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 07 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverDirect_Entropy
   %  Subject      : Computing the entropy =  - E_X{log(p_X(X))}  in which p_X is the pdf of X
   %                               
   %--- INPUT    
   %         MatRx(n,N)      : N realizations of random vector X of dimension n
   %         ind_parallel    : 0 no parallel computation
   %                           1    parallel computation
   %
   %--- OUTPUT 
   %         entropy         : entropy of probability density function p_X of X

   n = size(MatRx,1);        % n : dimension of random vector X 
   N = size(MatRx,2);        % N : number of realizations of random vector X   

   %--- Check data
   if n <= 0 || N <= 0
      error('STOP in sub_solverDirect_Entropy:  n <= 0 or N <= 0');
   end

   %--- Silver bandwidth       
   sx     = ((4/((n+2)*N))^(1/(n+4)));                                
   modifx = 1;                                 % in the usual theory, modifx = 1;          
   sx     = modifx*sx;                         % Silver bandwidth modified 
   cox    = 1/(2*sx*sx);

   %--- std of X 
   Rstd_x  = std(MatRx,0,2);                   % Rstd_x(n,1),MatRx(n,N)  
       
   %--- Computation of J0
   Rtemp = log(Rstd_x);
   J0    = sum(Rtemp,1) + n*log(sqrt(2*pi)*sx);
   clear Rtemp

   %--- Computation of 1/std(X) 
   Rstdm1_x     = 1./Rstd_x;                                       % Rstd_x(n,1)
  
   %--- Computing entropy   
   MatRtempx = zeros(1,N);                                         % MatRtempx(1,N) 

   %--- Vectorized sequence
   if ind_parallel == 0
      for j=1:N
          Rx_j           = MatRx(:,j);                             % Rx_j(n,1),MatRx(n,N)
          MatRxx_j       = (MatRx - Rx_j).*Rstdm1_x;               % MatRxx_j(n,N),MatRx(n,N),Rx_j(n,1),Rstdm1_x(n,1)
          MatRtempx(1,j) = mean(exp(-cox*(sum(MatRxx_j.^2,1))),2); % MatRtempx(1,N),MatRxx_j(n,N) 
      end
   end

   %--- Parallel sequence
   if ind_parallel == 1
      parfor j=1:N
          Rx_j           = MatRx(:,j);                             % Rx_j(n,1),MatRx(n,N)
          MatRxx_j       = (MatRx - Rx_j).*Rstdm1_x;               % MatRxx_j(n,N),MatRx(n,N),Rx_j(n,1),Rstdm1_x(n,1)
          MatRtempx(1,j) = mean(exp(-cox*(sum(MatRxx_j.^2,1))),2); % MatRtempx(1,N),MatRxx_j(n,N) 
      end
   end  
   MatRlog = log(MatRtempx);                                       % MatRlog(1,N)
   entropy = J0 - mean(MatRlog,2); 
   return 
end

%----------- validation Gaussian n=1 : entropie = 0.5 + log(sqrt(2*pi)) =  1.4189
% clear all
% rng('default');
% ind_parallel = 0;
% n     = 1;
% N     = 10000;
% MatRx = randn(n,N);
% [entropy] = sub_solverDirect_Entropy(MatRx,ind_parallel);
