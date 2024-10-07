function [Rb_targ1,coNr,coNr2,MatReta_targ,Rb_targ2,Rb_targ3] = sub_projection_target(n_x,n_d,MatRx_d,ind_exec_solver,ind_scaling, ...
                   ind_type_targ,Indx_targ_real,Indx_targ_pos,nx_targ,Indx_targ,N_r,MatRxx_targ_real,MatRxx_targ_pos,Rmeanxx_targ, ...
                   MatRcovxx_targ,nu,RmuPCA,MatRVectPCA,ind_display_screen,ind_print,ind_parallel,Rbeta_scale_real,Ralpham1_scale_real, ...
                   Rbeta_scale_log,Ralpham1_scale_log) 

   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_projection_target
   %  Subject      : for ind_exec_solver = 2 (Inverse analysis imposing targets for the leaning), computing and loading information
   %                 that defined the constraints as a function of ind_type_targ that is 1, 2, 3, or 4 
   %
   %  Publications: [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
   %                       Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).
   %                [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
   %                       American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
   %                [3] C. Soize, R. Ghanem, Physics-constrained non-Gaussian probabilistic learning on manifolds, 
   %                       International Journal for Numerical Methods in Engineering, doi: 10.1002/nme.6202, 121 (1), 110-145 (2020).
   %                [4] C. Soize, R. Ghanem, Probabilistic learning on manifolds constrained by nonlinear partial differential equations 
   %                       from small datasets, Computer Methods in Applied Mechanics and Engineering, doi:10.1016/j.cma.2021.113777, 
   %                       380, 113777 (2021).
   %                [5] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
   %                       Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).
   %                [6] C. Soize, Probabilistic learning inference of boundary value problem with uncertainties based on Kullback-Leibler 
   %                       divergence under implicit constraints, Computer Methods in Applied Mechanics and Engineering,
   %                       doi:10.1016/j.cma.2022.115078, 395, 115078 (2022). 
   %                [7] C. Soize, Probabilistic learning constrained by realizations using a weak formulation of Fourier transform of 
   %                       probability measures, Computational Statistics, doi:10.1007/s00180-022-01300-w, 38(4),1879–1925 (2023).
   %                [8] C. Soize, R. Ghanem, Probabilistic-learning-based stochastic surrogate model from small incomplete datasets 
   %                       for nonlinear dynamical systems,Computer Methods in Applied Mechanics and Engineering, 
   %                       doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024).
   %                [9] C. Soize, R. Ghanem, Transient anisotropic kernel for probabilistic learning on manifolds, 
   %                       Computer Methods in Applied Mechanics and Engineering, pp.1-44 (2024).
   %
   %--- INPUTS
   %
   %    n_x                               : dimension of random vector XX_d (unscaled) and X_d (scaled)
   %    n_d                               : number of points in the training set for XX_d and X_d
   %    MatRx_d(n_x,n_d)                  : n_d realizations of X_d (scaled)
   %
   %    ind_exec_solver                   : = 1 Direct analysis : giving a training dataset, generating a learned dataset
   %                                      : = 2 Inverse analysis: giving a training dataset and a target dataset, generating a learned dataset
   %    ind_scaling                       : = 0 no scaling
   %                                      : = 1 scaling
   %    ind_type_targ                     : = 1, targets defined by giving N_r realizations
   %                                      : = 2, targets defined by giving target mean-values 
   %                                      : = 3, targets defined by giving target mean-values and target covariance matrix
   %    Indx_targ_real(nbreal_targ,1)     : contains the nbreal_targ component numbers of XX, which are real  
   %                                        with 0 <= nbreal_targ <= n_x and for which a "standard scaling" will be used
   %    Indx_targ_pos(nbpos_targ,1)       : contains the nbpos_targ component numbers of XX, which are strictly positive (specific scaling)
   %                                        with 0 <= nbpos_targ <= n_x  and for which the scaling is {log + "standard scaling"}   
   %    nx_targ                           : number of components of XX with targets such that nx_targ = nbreal_targ + nbpos_targ <= n_x
   %    Indx_targ(nx_targ,1)              : nx_targ component numbers of XX with targets 
   %
   %                                      --- ind_type_targ = 1: targets defined by giving N_r realizations
   %    N_r                               : number of realizations of the targets               
   %    MatRxx_targ_real(nbreal_targ,N_r) : N_r realizations (unscaled) of the nbreal_targ targets of XX that are real
   %    MatRxx_targ_pos(nbpos_targ,N_r)   : N_r realizations (unscaled) of the nbpos_targ targets of XX that are positive
   %
   %                                      --- ind_type_targ = 2 or 3: targets defined by giving the mean value of unscaled XX_targ 
   %    Rmeanxx_targ(nx_targ,1)           : nx_targ components of mean value E{XX_targ} of vector-valued random target XX_targ
   %
   %                                      --- ind_type_targ = 3: targets defined by giving the covariance matrix of unscaled XX_targ 
   %    MatRcovxx_targ(nx_targ,nx_targ)   : covariance matrix of XX_targ 
   % 
   %    nu                                : dimension of H
   %    RmuPCA(nu,1)                      : vector of eigenvalues in descending order
   %    MatRVectPCA(n_x,nu)               : matrix of the eigenvectors associated to the eigenvalues loaded in RmuPCA
   %   
   %    ind_display_screen                : = 0 no display,                = 1 display
   %    ind_print                         : = 0 no print,                  = 1 print
   %    ind_parallel                      : = 0 no parallel computation,   = 1 parallele computation
   %
   %                                      --- ind_exec_solver = 2 and with ind_scaling = 1
   %    Rbeta_scale_real(nbreal,1)        : loaded if nbreal >= 1 or = [] if nbreal  = 0  
   %    Ralpham1_scale_real(nbreal,1)     : loaded if nbreal >= 1 or = [] if nbreal  = 0   
   %    Rbeta_scale_log(nbpos,1)          : loaded if nbpos >= 1  or = [] if nbpos   = 0   
   %    Ralpham1_scale_log(nbpos,1)       : loaded if nbpos >= 1  or = [] if nbpos   = 0  
   %
   %--- OUPUTS
   %                                      --- ind_type_targ = 1: targets defined by giving N_r realizations of XX_targ 
   %    Rb_targ1(N_r,1)                  : E{h_targ1(H^c)} = b_targ1  with h_targ1 = (h_{targ1,1}, ... , h_{targ1,N_r})
   %    coNr                             : parameter used for evaluating  E{h^c_targ(H^c)}               
   %    coNr2                            : parameter used for evaluating  E{h^c_targ(H^c)} 
   %    MatReta_targ(nu,N_r)             : N_r realizations of the projection of XX_targ on the model
   %
   %                                     --- ind_type_targ = 2 or 3: targets defined by giving mean value of XX_targ
   %    Rb_targ2(nu,1)                                               yielding the constraint E{H^c} = b_targ2 
   %
   %                                     --- ind_type_targ = 3: targets defined by giving target covariance matrix of XX_targ
   %    Rb_targ3(nu,1)                                          yielding the constraint diag(E{H_c H_c'}) = b_targ3   
   %
   %--- COMMENTS
   %             (1) for ind_type_targ = 2 or 3, we have nbpos_targ = 0 and Indx_targ_pos(nbpos_targ,1) = []
   %             (2) Note that the constraints on H^c is not E{H_c H_c'}) = [b_targ3]  but  diag(E{H_c H_c'}) = b_targ3 

   if ind_display_screen ~= 0 && ind_display_screen ~= 1       
      error('STOP1 in sub_projection_target: ind_display_screen must be equal to 0 or equal to 1')
   end
   if ind_print ~= 0 && ind_print ~= 1       
      error('STOP2 in sub_projection_target: ind_print must be equal to 0 or equal to 1')
   end
   if ind_parallel ~= 0 && ind_parallel ~= 1       
      error('STOP3 in sub_projection_target: ind_parallel must be equal to 0 or equal to 1')
   end

   if ind_display_screen == 1                              
      disp('--- beginning Task8_ProjectionTarget')
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task8_ProjectionTarget \n ');
      fprintf(fidlisting,'      \n ');  
      fclose(fidlisting);  
   end

   TimeStart = tic; 
     
   %--- Loading and checking input data and parameters
   if n_d < 1 || n_x < n_d
      error('STOP4 in sub_projection_target: n_d < 1 or n_x < n_d');
   end
   [n1temp,n2temp] = size(MatRx_d);    % MatRx_d(n_x,n_d) (scaled)
   if n1temp ~= n_x || n2temp ~= n_d
      error('STOP5 in sub_projection_target: the dimensions of MatRx_d(n_x,n_d) are not coherent');
   end
   if ind_exec_solver ~= 2
      error('STOP6 in sub_projection_target: ind_exec_solver should be equal to 2');
   end
   if ind_scaling ~= 0 && ind_scaling ~= 1
      error('STOP7 in sub_projection_target: ind_scaling must be equal to 0 or 1');
   end
   if ind_type_targ ~= 1 &&  ind_type_targ ~= 2 &&  ind_type_targ ~= 3 
      error('STOP8 in sub_projection_target: ind_type_targ must be equal to 1, 2 or 3');    
   end

   % Checking parameters and data related to Indx_targ_real(nbreal_targ,1) and Indx_targ_pos(nbreal_pos,1)
   nbreal_targ = size(Indx_targ_real,1);
   if nbreal_targ >= 1
      n2temp = size(Indx_targ_real,2);        % Indx_targ_real(nbreal_targ,1)
      if n2temp ~= 1
         error('STOP9 in sub_projection_target: error in the second dimension of matrix Indx_targ_real(nbreal_targ,1)');
      end
   end
   nbpos_targ = size(Indx_targ_pos,1);
   if nbpos_targ >= 1
      if ind_type_targ == 2 || ind_type_targ == 3 
         error('STOP10 in sub_projection_target: for an inverse problem, when ind_type_targ = 2 or 3, one must have nbpos_targ = 0')
      end
      n2temp = size(Indx_targ_pos,2);        % Indx_targ_pos(nbreal_pos,1)
      if n2temp ~= 1
         error('STOP11 in sub_projection_target: error in the second dimension of matrix Indx_targ_pos(nbreal_pos,1)');
      end
   end

   % Checking parameters that control target parameteres and data
   if nx_targ - (nbreal_targ + nbpos_targ) ~= 0
      error('STOP12 in sub_projection_target: nx_targ is not equal to (nbreal_targ + nbpos_targ)'); 
   end
   if nx_targ < 1 || nx_targ > n_x
      error('STOP13 in sub_projection_target: nx_targ is less than 1 or greater than n_x'); 
   end
   n2temp = size(Indx_targ,2);               % Indx_targ(nx_targ,1)
   if n2temp ~= 1
      error('STOP14 in sub_projection_target: error in the second dimension of matrix Indx_targ(nx_targ,1)');
   end  

   % For ind_type_targ = 1, checking parameters and data related to MatRxx_targ_real and MatRxx_targ_pos
   if ind_type_targ == 1 
      if  N_r <= 0
          error('STOP15 in sub_projection_target: for ind_type_targ = 1, one must have N_r greater than or equal to 1');
      end
      if nbreal_targ >= 1
         [n1temp,n2temp] = size(MatRxx_targ_real);                              % MatRxx_targ_real(nbreal_targ,N_r)     
         if n1temp ~= nbreal_targ || n2temp ~= N_r
            error('STOP16 in sub_projection_target: for ind_type_targ = 1, the dimensions of MatRxx_targ_real(nbreal_targ,N_r) are not coherent');
         end
      end
      if nbpos_targ >= 1
         [n1temp,n2temp] = size(MatRxx_targ_pos);                              % MatRxx_targ_pos(nbpos_targ,N_r)     
         if n1temp ~= nbpos_targ || n2temp ~= N_r
            error('STOP17 in sub_projection_target: for ind_type_targ = 1, the dimensions of MatRxx_targ_pos(nbpos_targ,N_r) are not coherent');
         end
      end
   end

   % For ind_type_targ = 2 or 3, checking parameters and data related to Rmeanxx_targ(nx_targ,1)
   if ind_type_targ == 2 || ind_type_targ == 3 
      [n1temp,n2temp] = size(Rmeanxx_targ);                              % Rmeanxx_targ(nx_targ,1) 
      if n1temp ~= nx_targ || n2temp ~= 1
         error('STOP18 in sub_projection_target: for ind_type_targ >= 2, the dimensions of Rmeanxx_targ(nx_targ,1) are not coherent');
      end
   end

   % For ind_type_targ = 3, checking that matrix MatRcovxx_targ is symmetric and positive definite
   if ind_type_targ == 3
      [n1temp,n2temp] = size(MatRcovxx_targ);                            % MatRcovxx_targ(nx_targ,nx_targ)
      if n1temp ~= nx_targ || n2temp ~= nx_targ
         error('STOP19 in sub_projection_target: for ind_type_targ = 3, the dimensions of MatRcovxx_targ(nx_targ,nx_targ) are not coherent');
      end
      if ~isequal(MatRcovxx_targ, MatRcovxx_targ')
        error('STOP20 in sub_projection_target: matrix MatRcovxx_targ is not symmetric');
      end
      [~,MatReigen] = eig(MatRcovxx_targ);
      Reigen = diag(MatReigen);
      if ~all(Reigen > 0)          
         error('STOP21 in sub_projection_target: matrix MatRcovxx_targ is not positive definite.');
      end
   end 

   % Checking parameters and data related to PCA
   if nu < 1 || nu > n_d
      error('STOP22 in sub_projection_target: nu < 1 or nu > n_d');
   end
   [n1temp,n2temp] = size(RmuPCA);          % RmuPCA(nu,1)  
   if n1temp ~= nu || n2temp ~= 1
      error('STOP23 in sub_projection_target: dimensions of RmuPCA(nu,1) are not coherent');
   end
   if nu > n_x || n_x < 1
      error('STOP24 in sub_projection_target: nu > n_x or n_x < 1');
   end
   [n1temp,n2temp] = size(MatRVectPCA);    % MatRVectPCA(n_x,nu)  
   if n1temp ~= n_x || n2temp ~= nu
      error('STOP25 in sub_projection_target: dimensions of MatRVectPCA(n_x,nu) are not coherent');
   end
   
   % If ind_scaling = 1, 
   % (1) checking the data coherence of Rbeta_scale_real(nbreal,1), Ralpham1_scale_real(nbreal,1), Rbeta_scale_log(nbpos,1), 
   %     and Ralpham1_scale_log(nbpos,1)      
   % (2) loading Rbeta_scale(n_x,1) and Ralpham1_scale(n_x,1)
   % (3) Extracting Rbeta_targ(nx_targ,1) and Ralpham1_targ(nx_targ,1) from Rbeta_scale(n_x,1) and Ralpham1_scale(n_x,1)

   if ind_scaling == 1
      if ind_type_targ == 1
         [nbreal,n2temp] = size(Rbeta_scale_real);           % Rbeta_scale_real(nbreal,1) 
         if nbreal >= 1
            if n2temp ~= 1
               error('STOP26 in sub_projection_target: the second dimension of Rbeta_scale_real must be 1');
            end
         end
         [nbpos,n2temp] = size(Rbeta_scale_log);              % Rbeta_scale_log(nbpos,1)  
         if nbpos >= 1
            if n2temp ~= 1
               error('STOP27 in sub_projection_target: the second dimension of Rbeta_scale_log must be 1');
            end
         end
         [n1temp,n2temp] = size(Ralpham1_scale_real);         % Ralpham1_scale_real(nbreal,1)
         if n1temp >= 1
            if nbreal ~= n1temp || n2temp ~= 1
               error('STOP28 in sub_projection_target: the second dimension of Ralpham1_scale_real must be 1');
            end
         end
         [n1temp,n2temp] = size(Ralpham1_scale_log);         % Ralpham1_scale_log(nbpos,1) 
         if n1temp >= 1
            if nbpos ~= n1temp || n2temp ~= 1
               error('STOP29 in sub_projection_target: the second dimension of Ralpham1_scale_log must be 1');
            end
         end

         % Loading Rbeta_scale(n_x,1) and Ralpham1_scale(n_x,1)
         Rbeta_scale = [Rbeta_scale_real         % Rbeta_scale_real(nbreal,1) 
                        Rbeta_scale_log];        % Rbeta_scale_log(nbpos,1) 
         Ralpham1_scale =[Ralpham1_scale_real    % Ralpham1_scale_real(nbreal,1)
                          Ralpham1_scale_log];   % Ralpham1_scale_log(nbpos,1)
      end                

      if ind_type_targ == 2 || ind_type_targ == 3 
         [nbreal,n2temp] = size(Rbeta_scale_real);          % Rbeta_scale_real(nbreal,1)  
         if nbreal ~= n_x 
            error('STOP30 in sub_projection_target: for ind_type_targ = 2 or 3, nbreal must be equal to n_x');
         end
         if n2temp ~= 1
            error('STOP31 in sub_projection_target: the second dimension of Rbeta_scale_real must be 1');
         end
         [n1temp,n2temp] = size(Ralpham1_scale_real);        % Ralpham1_scale_real(nbreal,1)  
         if nbreal ~= n1temp 
            error('STOP32 in sub_projection_target: for ind_type_targ = 2 or 3, nbreal must be equal to n_x');
         end
         if n2temp ~= 1
            error('STOP33 in sub_projection_target: the second dimension of Ralpham1_scale_real must be 1');
         end

         % Loading Rbeta_scale(n_x,1) and Ralpham1_scale(n_x,1)
         Rbeta_scale    = Rbeta_scale_real;         % Rbeta_scale_real(nbreal,1) 
         Ralpham1_scale = Ralpham1_scale_real;      % Ralpham1_scale_real(nbreal,1)
      end

      % Extraction Rbeta_targ(nx_targ,1) and Ralpham1_targ(nx_targ,1) from Rbeta_scale(n_x,1) and Ralpham1_scale(n_x,1)
         Rbeta_targ       = Rbeta_scale(Indx_targ,1);           % Rbeta_targ(nx_targ,1),Rbeta_scale(n_x,1),Indx_targ(nx_targ,1)
         Ralpham1_targ    = Ralpham1_scale(Indx_targ,1);        % Ralpham1_targ(nx_targ,1),Ralpham1_scale(n_x,1),Indx_targ(nx_targ,1)
         MatRalpham1_targ = diag(Ralpham1_targ);                % MatRalpham1_targ(nx_targ,nx_targ)
   end
   
   %--- Computing Rmeanx_d_targ(nx_targ,1) and MatRVectPCA_targ(nx_targ,nu) from Rmeanx_d(nx,1) and MatRVectPCA(nx,nu)
   Rmeanx_d         = mean(MatRx_d,2);                    % Rmeanx_d(n_x,1), MatRx_d(n_x,n_d) (scaled)    
   Rmeanx_d_targ    = Rmeanx_d(Indx_targ,1);              % Rmeanx_d_targ(nx_targ,1), Indx_targ(nx_targ,1)
   MatRVectPCA_targ = MatRVectPCA(Indx_targ,:);           % MatRVectPCA_targ(nx_targ,nu),MatRVectPCA(n_x,nu)
   Rcoef            = sqrt(RmuPCA);                       % RmuPCA(nu,1)
   MatRdiagRmu1s2   = diag(Rcoef);                        % MatRdiagRmu1s2(nu,nu) 
   MatRV            = MatRVectPCA_targ*MatRdiagRmu1s2;    % MatRV(nx_targ,nu),MatRVectPCA_targ(nx_targ,nu),MatRdiagRmu1s2(nu,nu)
   MatRVt           = pinv(MatRV);                        % MatRVt(nu,nx_targ),MatRV(nx_targ,nu) 
   clear Rmeanx_d Rcoef MatRdiagRmu1s2 MatRVectPCA_targ

   %--- initialization 
   if ind_type_targ == 1
      Rb_targ2    = [];
      Rb_targ3    = [];
   end
   if ind_type_targ == 2
      coNr         = 0;
      coNr2        = 0;
      Rb_targ1     = [];
      MatReta_targ = [];
      Rb_targ3     = [];
   end
   if ind_type_targ == 3
      coNr         = 0;
      coNr2        = 0;
      MatReta_targ = [];
      Rb_targ1     = [];
   end

   %--------------------------------------------------------------------------------------------------------------------------
   %   Case ind_type_targ = 1: targets defined by giving N_r realizations
   %                           computing coNr,coNr2,Rb_targ1(N_r,1)  
   %--------------------------------------------------------------------------------------------------------------------------

   if ind_type_targ == 1

      MatRxx_targ_log = [];
      if nbpos_targ >= 1
         MatRxx_targ_log = log(MatRxx_targ_pos);   % MatRxx_targ_log(nbpos_targ,N_r), MatRxx_targ_pos(nbpos_targ,N_r) 
      end            

      %--- Computing MatRx_targ(nx_targ,N_r)  (scaled) from  MatRxx_targ(nx_targ,N_r)  (unscaled)
      MatRxx_targ = [MatRxx_targ_real     % MatRxx_targ(nx_targ,N_r)
                     MatRxx_targ_log];
      % No scaling 
      if ind_scaling == 0                                                          
         MatRx_targ = MatRxx_targ;        % MatRx_targ(nx_targ,N_r),MatRxx_targ(nx_targ,N_r)
      end
      % Scaling
      if ind_scaling == 1
         MatRx_targ  = Ralpham1_targ.*(MatRxx_targ - repmat(Rbeta_targ,1,N_r));  % MatRx_targ(nx_targ,N_r),Ralpham1_targ(nx_targ,1)
      end

      %--- Computing MatReta_targ(nu,N_r)      
      MatRx_targ_tilde = MatRx_targ - Rmeanx_d_targ;         % MatRx_targ_tilde(nx_targ,N_r),MatRx_targ(nx_targ,N_r)
      MatReta_targ     = MatRVt*MatRx_targ_tilde;            % MatReta_targ(nu,N_r),MatRVt(nu,nx_targ),MatRx_targ_tilde(nx_targ,N_r)
     
      %--- Computing error_target as the projection error E{||X_targ - (meanx_targ + V*eta_targ)||^2} / E{||X_targ||^2}
      RtempNum = zeros(N_r,1);
      RtempDen = zeros(N_r,1);
      for r = 1:N_r
          RtempNum(r) = (norm(MatRx_targ_tilde(:,r) - MatRV*MatReta_targ(:,r)))^2;  % MatRV(nx_targ,nu),MatReta_targ(nu,N_r)
          RtempDen(r) = (norm(MatRx_targ(:,r)))^2;  
      end
      error_target = sum(RtempNum,1)/sum(RtempDen,1);
      clear RtempNum RtempDen MatRx_targ_tilde MatRx_targ
      
      %---display screen
      if ind_display_screen == 1
         disp([' Relative projection error of the target dataset onto the model = ',num2str(error_target)]);
      end

      %--- print 
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+'); 
         fprintf(fidlisting,'      \n ');
         fprintf(fidlisting,'--- Relative projection error of the target dataset onto the model \n '); 
         fprintf(fidlisting,'      \n ');
         fprintf(fidlisting,'    error_projection_target_dataset      = %14.7e \n ',error_target); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n '); 
         fclose(fidlisting);         
      end

      %--- Computing  Rbc(N_r,1) of the experimental constraints with the PCA representation 
      sNr   = ((4/((nu+2)*N_r))^(1/(nu+4)));   % Silver bandwidth 
      s2Nr  = sNr*sNr;
      coNr  = 1/(nu*s2Nr);
      coNr2 = 2*coNr;   
                            %--- scalar sequence for computing Rb_targ1(N_r,1) (given for readability of the used algebraic formula)
                            % Rb_targ1 = zeros(N_r,1);
                            % for r = 1:N_r
                            %     b_r = 0;
                            %     for rp = 1:N_r
                            %         Reta_rp_r = MatReta_targ(:,rp) - MatReta_targ(:,r);
                            %         expo = exp(-coNr*sum(Reta_rp_r.^2,1));
                            %         b_r = b_r + expo;
                            %     end
                            %     Rb_targ1(r) = b_r/N_r;
                            % end

                            %--- vectorized sequence for computing Rb_targ1(N_r,1)
      Rsumexpo = zeros(1,N_r);                                              % Rsumexpo(1,N_r);
      for rp = 1:N_r
          Reta_targ_rp = MatReta_targ(:,rp);                                % Reta_targ_rp(nu,1),MatReta_targ(nu,N_r)
          MatRtarg_rp  = Reta_targ_rp - MatReta_targ;                       % MatRtarg_rp(nu,N_r),MatReta_targ(nu,N_r)
          Rtarg_rp     = exp(-coNr*(sum(MatRtarg_rp.^2,1)));                % Rtarg_rp(1,N_r),MatRtarg_rp(nu,N_r), 
          Rsumexpo     = Rsumexpo +  Rtarg_rp;                              % Rsumexpo(1,N_r);
      end 
      Rb_targ1     = Rsumexpo'/N_r;                                         % Rb_targ1(N_r,1) 
      normRb_targ1 = norm(Rb_targ1);
      clear Rsumexpo Reta_targ_rp MatRtarg_rp Rtarg_rp

      %---display screen
      if ind_display_screen == 1
         disp(' ');
         disp([' Sylverman bandwidth sNr  = ',num2str(sNr)]);   
         disp([' nu                       = ',num2str(nu)]);
         disp([' coNr  = 1/(nu*sNr*sNr)   = ',num2str(coNr)]);
         disp([' coNr2 = 2*coNr           = ',num2str(coNr2)]);
         disp(' ');
         disp([' norm of Rb_targ1         = ',num2str(normRb_targ1)]); 
         disp(' ');
      end

      %--- print 
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+'); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' Sylverman bandwidth sNr  = %14.7e \n ',sNr); 
         fprintf(fidlisting,' nu                       = %7i \n ',nu); 
         fprintf(fidlisting,' coNr  = 1/(nu*sNr*sNr)   = %14.7e \n ',coNr); 
         fprintf(fidlisting,' coNr2 = 2*coNr           = %14.7e \n ',coNr2); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' norm of Rb_targ1         = %14.7e \n ',normRb_targ1);
         fprintf(fidlisting,'      \n '); 
         fclose(fidlisting); 
      end
   end

   %--------------------------------------------------------------------------------------------------------------------------------------
   %    Case ind_type_targ = 2 or 3: computing Rb_targ2(nu,1) giving the mean value Rmeanxx_targ(nx_targ,1) of XX_targ (unscaled)
   %    ind_type_targ : = 2, targets defined by giving the mean value Rmeanxx_targ(nx_targ,1) of XX_targ 
   %                  : = 3, targets defined by giving the mean value and the covariance matrix MatRcovxx_targ(nx_targ,nx_targ) of XX_targ
   %--------------------------------------------------------------------------------------------------------------------------------------

   if ind_type_targ == 2 || ind_type_targ == 3 
      
      %--- Computing Rmeanxx_targ(nx_targ,1)  (scaled) from  Rmeanxx_targ(nx_targ,1)  (unscaled)
      % No scaling 
      if ind_scaling == 0                                                          
         Rmeanx_targ = Rmeanxx_targ;                                  % Rmeanx_targ(nx_targ,1),Rmeanxx_targ(nx_targ,1)
      end
      % Scaling
      if ind_scaling == 1                                             % Rmeanxx_targ(nx_targ,1),Rbeta_targ(nx_targ,1)
         Rmeanx_targ = MatRalpham1_targ*(Rmeanxx_targ - Rbeta_targ);  % Rmeanx_targ(nx_targ,1),MatRalpham1_targ(nx_targ,nx_targ)
      end 
      Rmeanx_targ_tilde = Rmeanx_targ - Rmeanx_d_targ;           % Rmeanx_targ_tilde(nx_targ,1),Rmeanx_targ(nx_targ,1),Rmeanx_d_targ(nx_targ,1)
      Rb_targ2          = MatRVt*Rmeanx_targ_tilde;              % Rb_targ2(nu,1),MatRVt(nu,nx_targ),Rmeanx_targ_tilde(nx_targ,1)
   end

   %--------------------------------------------------------------------------------------------------------------------------------------
   %    Case ind_type_targ = 3: computing Rb_targ3(nu,1) giving the mean value Rmeanxx_targ(nx_targ,1) and the covariance 
   %                            matrix MatRcovxx_targ(nx_targ,nx_targ) of XX_targ  (unscaled)
   %--------------------------------------------------------------------------------------------------------------------------------------

   if ind_type_targ == 3
      MatRcovx_targ = MatRalpham1_targ*MatRcovxx_targ*MatRalpham1_targ;      % MatRcovxx_targ(nx_targ,nx_targ),MatRalpham1_targ(nx_targ,nx_targ) 
      MatRtemp      = MatRcovx_targ + Rmeanx_targ_tilde*Rmeanx_targ_tilde';  % MatRtemp(nx_targ,nx_targ)Rmeanx_targ_tilde(nx_targ,1)
      Rb_targ3      = diag(MatRVt*MatRtemp*MatRVt');                         % Rb_targ3(nu,1),MatRVt(nu,nx_targ) 
   end

   ElapsedTime = toc(TimeStart);   

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ----- Elapsed time for Task8_ProjectionTarget \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' Elapsed Time   =  %10.2f\n',ElapsedTime);   
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task8_ProjectionTarget')
   end    
   
   return
end
      

