function [iX] = sub_solverDirect_Mutual_Information(MatRx,ind_parallel)

   %--------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 07 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverDirect_Mutual_Information
   %  Subject      : Computing the Mutual Information iX = E_X{log(p_X(X)/(p_X1(X1) x ... x pXn(Xn))} 
   %                 in which X = (X1,...,Xn), p_X is the pdf of X and p_Xj is the pdf of Xj
   %                 The algorithm used is those of Kullback 
   %                               
   %--- INPUT    
   %         MatRx(n,N)      : N realizations of random vector X of dimension n
   %         ind_parallel    : 0 no parallel computation
   %                           1    parallel computation
   %--- OUTPUT 
   %         iX

   n = size(MatRx,1);      % n : dimension of random vector X 
   N = size(MatRx,2);      % N : number of realizations of random vector X   

   %--- Check data
   if n <= 0 || N <= 0
      error('STOP in sub_solverDirect_Mutual_Information:  n <= 0 or N <= 0');
   end

   %--- Silver bandwidth       
   sx     = ((4/((n+2)*N))^(1/(n+4)));                                
   modifx = 1;                                                       % in the usual theory, modifx=1;          
   sx     = modifx*sx;                                               % Silver bandwidth modified 
   cox    = 1/(2*sx*sx);
   
   %--- std of X 
   Rstd_x = std(MatRx,0,2);                                          % Rstd_x(n,1),MatRx(n,N)  

   %--- Computation of 1/std(X) 
   Rstdm1_x = 1./Rstd_x;                                             % Rstdm1_x(n,1)
   
   %--- Computing iX (the mutual information)
   MatRtempx = zeros(1,N);                                           % MatRtempx(1,N) 
   MatRy = zeros(n,N);                                               % MatRy(n,N)

   %--- Vectorial sequence
   if ind_parallel == 0
      for j =1:N
          Rx_j           = MatRx(:,j);                               % Rx_j(n,1),MatRx(n,N)
          MatRxx_j       = (MatRx - Rx_j).*Rstdm1_x;                 % MatRxx_j(n,N),MatRx(n,N),Rx_j(n,1),Rstdm1_x(n,1)
          MatRtempx(1,j) = mean(exp(-cox*(sum(MatRxx_j.^2,1))),2);   % MatRtempx(1,N),MatRxx_j(n,N) 
          MatRy(:,j)     = mean(exp(-cox*(MatRxx_j.^2)),2);          % MatRy(n,N),MatRxx_j(n,N)              
      end          
   end

   %--- Parallel sequence
   if ind_parallel == 1
      parfor j =1:N
          Rx_j           = MatRx(:,j);                               % Rx_j(n,1),MatRx(n,N)
          MatRxx_j       = (MatRx - Rx_j).*Rstdm1_x;                 % MatRxx_j(n,N),MatRx(n,N),Rx_j(n,1),Rstdm1_x(n,1)
          MatRtempx(1,j) = mean(exp(-cox*(sum(MatRxx_j.^2,1))),2);   % MatRtempx(1,N),MatRxx_j(n,N) 
          MatRy(:,j)     = mean(exp(-cox*(MatRxx_j.^2)),2);          % MatRy(n,N),MatRxx_j(n,N)              
      end            
   end  
   MatRtempy = prod(MatRy);                                           % MatRtempy(1,N)
   MatRlog   = log(MatRtempx./MatRtempy);                             % MatRlog(1,N)
   iX        = mean(MatRlog,2); 
   return 
end

%----------- Validation for the case of independent random components.
%            we then have to obtain iX = 0 
% clear all
% rng('default')
% ind_parallel = 1;
% n            = 4;
% N            = 100000;
% MatRxtemp    = randn(n,N);
% 
% % Rotation matrix to introduce a dependence without changing the normalization
% MatRa        = randn(n,n);
% MatRb        = MatRa*MatRa';
% [MatRphi,~]  = eig(MatRb);
% MatRx        = MatRphi*MatRxtemp;
% [iX] = sub_solverDirect_Mutual_Information(MatRx,ind_parallel)
