function [divKL] = sub_solverDirect_Kullback(MatRx,MatRy,ind_parallel)
    
    %------------------------------------------------------------------------------------------------------------------------------------------
    %
    %  Copyright: Christian Soize, Universite Gustave Eiffel, 07 June 2024
    %
    %  Software     : Probabilistic Learning on Manifolds (PLoM) 
    %  Function name: sub_solverDirect_Kullback
    %  Subject      : Computing the Kullback-Leibler divergence: divKL = E_X{log(p_X(X)/p_Y(X))}  in which
    %                 X and Y are random vectors of dimension n with pdf p_X and p_Y    
    %                               
    %--- INPUT    
    %         MatRx(n,Nx)     : Nx realizations of random vector X
    %         MatRy(n,Ny)     : Ny realizations of random vector Y
    %         ind_parallel    : 0 no parallel computation
    %                           1    parallel computation
    %
    %--- OUTPUT 
    %         divKL

    n      = size(MatRx,1);     % n  : dimension of random vectors X and Y
    nyTemp = size(MatRy,1);
    Nx     = size(MatRx,2);     % Nx : number of realizations of random vector X
    Ny     = size(MatRy,2);     % Ny : number of realizations of random vector Y  

    if n ~= nyTemp        
       error('STOP1 in sub_solverDirect_Kullback: dimension of X and Y must be the same')
    end
    if n <= 0 || Nx <= 0 || Ny <= 0
       error('STOP2 in sub_solverDirect_Kullback:  n <= 0 or Nx <= 0 or Nx <= 0');
    end
   
    %--- Silver bandwidth       
    sx     = ((4/((n+2)*Nx))^(1/(n+4)));                                
    modifx = 1;                                  % in the usual theory, modifx = 1;          
    sx     = modifx*sx;                          % Silver bandwidth modified 
    cox    = 1/(2*sx*sx);

    sy     = ((4/((n+2)*Ny))^(1/(n+4)));                                
    modify = 1;                                  % in the usual theory, modify = 1;          
    sy     = modify*sy;                          % Silver bandwidth modified 
    coy    = 1/(2*sy*sy);

    %--- std of X and Y
    Rstd_x = std(MatRx,0,2);                     % Rstd_x(n,1),MatRx(n,Nx)  
    Rstd_y = std(MatRy,0,2);                     % Rstd_y(n,1),MatRy(n,Ny)                                                  
        
    %--- Computation of J0
    Rtemp = log(Rstd_y./Rstd_x);
    J0 = sum(Rtemp,1) + log(Ny/Nx) + n*log(sy/sx);
    clear Rtemp

    %--- Computation of 1/std(X) and 1/std(Y)
    Rstdm1_x     = 1./Rstd_x;                                      % Rstd_x(n,1)
    Rstdm1_y     = 1./Rstd_y;                                      % Rstd_y(n,1)
    
    %--- computation of the Kullback
    MatRtempx = zeros(1,Nx);                                       % MatRtempx(1,Nx) 
    MatRtempy = zeros(1,Nx);                                       % MatRtempy(1,Nx): it is Nx AND NOT Ny

    % Vectorized sequence
    if ind_parallel == 0
       for j=1:Nx
           Rx_j           = MatRx(:,j);                            % Rx_j(n,1),MatRx(n,Nx)
           MatRxx_j       = (MatRx - Rx_j).*Rstdm1_x;              % MatRxx_j(n,Nx),MatRx(n,Nx),Rx_j(n,1),Rstdm1_x(n,1)
           MatRtempx(1,j) = sum(exp(-cox*(sum(MatRxx_j.^2,1))),2); % MatRtempx(1,Nx),MatRxx_j(n,Nx) 
           MatRyy_j       = (MatRy-Rx_j).*Rstdm1_y;                % MatRyy_j(n,Ny),MatRy(n,Ny),Rx_j(n,1),Rstdm1_y(n,1)
           MatRtempy(1,j) = sum(exp(-coy*(sum(MatRyy_j.^2,1))),2); % MatRtempy(1,Nx),MatRyy_j(n,Nx) 
       end
    end

   % Parallel sequence 
   if ind_parallel == 1
      parfor j=1:Nx
          Rx_j           = MatRx(:,j);                             % Rx_j(n,1),MatRx(n,Nx)
          MatRxx_j       = (MatRx - Rx_j).*Rstdm1_x;               % MatRxx_j(n,Nx),MatRx(n,Nx),Rx_j(n,1),Rstdm1_x(n,1)
          MatRtempx(1,j) = sum(exp(-cox*(sum(MatRxx_j.^2,1))),2);  % MatRtempx(1,Nx),MatRxx_j(n,Nx) 
          MatRyy_j       = (MatRy-Rx_j).*Rstdm1_y;                 % MatRyy_j(n,Ny),MatRy(n,Ny),Rx_j(n,1),Rstdm1_y(n,1)
          MatRtempy(1,j) = sum(exp(-coy*(sum(MatRyy_j.^2,1))),2);  % MatRtempy(1,Nx),MatRyy_j(n,Nx) 
      end
   end  

   MatRlog = log(MatRtempx./MatRtempy);                           % MatRlog(1,Nx)
   divKL   = J0 + mean(MatRlog,2); 
    
    return 
end

% --- validation test
% clear all
% rng('default')
% ind_parallel = 1;
% n      = 4;
% Nx     = 1000;
% Ny     = 100
% mX     = 1.8;
% mY     = 0.7;
% sigmaX = 0.5;
% sigmaY = 1.2;
% MatRx  = mX + sigmaX*randn(n,Nx);
% MatRy  = mY + sigmaY*randn(n,Ny);
% [divKL]   = sub_solverDirect_Kullback(MatRx,MatRy,ind_parallel)


