function [] = sub_conditional_statistics(ind_mean,ind_mean_som,ind_pdf,ind_confregion, ...
                                         n_x,n_q,nbParam,n_w,n_d,n_ar,nbMC,nu,MatRx_d,MatRxx_d,MatReta_ar,RmuPCA, ...
                                         MatRVectPCA,Indx_real,Indx_pos,Indq_obs,Indw_obs,nx_obs,Indx_obs,ind_scaling, ...
                                         Rbeta_scale_real,Ralpha_scale_real,Rbeta_scale_log,Ralpha_scale_log, ...
                                         nbw0_obs,MatRww0_obs,Ind_Qcomp,nbpoint_pdf,pc_confregion,ind_display_screen,ind_print)

   %------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 26 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_conditional_statistics
   %  Subject      : computation of conditional statistics:
   %                - Estimation of the conditional mean
   %                - Estimation of the conditional mean and second-order moment
   %                - Estimation of the conditional pdf of component jcomponent <= nq_obs
   %                - Estimation of the conditional confidence region
   %
   %  Publications [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
   %                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
   %               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
   %                         American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020).   
   %               [3] C. Soize, R. Ghanem, Probabilistic-learning-based stochastic surrogate model from small incomplete datasets 
   %                         for nonlinear dynamical systems, Computer Methods in Applied Mechanics and Engineering, 
   %                         doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024). 
   %               [ ] For the conditional statistics formula, see the Appendix of paper [3]
   %
   %--- INPUTS    
   % 
   %     ind_mean               : = 0  No estimation of the conditional mean
   %                              = 1     estimation of the conditional mean
   %     ind_mean_som           : = 0 No estimation of the conditional mean and second-order moment
   %                              = 1    estimation of the conditional mean and second-order moment
   %     ind_pdf                : = 0 No estimation of the conditional pdf of component jcomponent <= nq_obs
   %                              = 1    estimation of the conditional pdf of component jcomponent <= nq_obs
   %     ind_confregion         : = 0 No estimation of the conditional confidence region
   %                              = 1    estimation of the conditional confidence region
   %
   %     n_x                    : dimension of random vectors XX_ar  (unscaled) and X_ar (scaled)  
   %     n_q                    : dimension of random vector QQ (unscaled quantitity of interest)  1 <= n_q    
   %     nbParam                : number of sampling point of the physical parameters 1 <= nbParam <= n_q
   %                     WARNING: For the analysis of the conditional statistics of Step4, the organization of the components of the 
   %                              QQ vector of the quantity of interest QoI is as follows (this organization must be planned from the 
   %                              creation of the data in this function "mainWorkflow_Data_generation1.m" and  also in
   %                              "mainWorkflow_Data_generation2.m" .
   %
   %                              If the QoI depends on the sampling in nbParam points of a physical system parameter
   %                              (such as time or frequency), if QoI_1, QoI_2, ... are the scalar quantities of interest, and if 
   %                              f_1,...,f_nbParam are the nbParam sampling points, the components of the QQ vector must be organized 
   %                              as follows: 
   %                              [(QoI_1,f_1) , (QoI_1,f_2), ... ,(QoI_1,f_nbParam), (QoI_2,f_1), (QoI_2,f_2), ... ,(QoI_2,f_nbParam), ... ]'.
   %
   %                     WARNING: If nbParam > 1, this means that nq_obs is equal to nqobsPhys*nbParam, in which nqobsPhys is the number
   %                              of the components of the state variables that are observed. Consequently, nq_obs/nbParam must be 
   %                              an integer, if not there is an error in the given value of nbParam of in the Data generation in 
   %                              "mainWorkflow_Data_generation1.m" and "mainWorkflow_Data_generation2.m" 
   %
   %                     WARNING: NOTE THAT if such a parameter does not exist, it must be considered that nbParam = 1, but the 
   %                              information structure must be consistent with the case nbParam > 1.  
   %
   %     n_w                         : dimension of random vector WW (unscaled control variable) with 1 <= n_w  
   %     n_d                         : number of points in the training set for XX_d and X_d  
   %     n_ar                        : number of points in the learning set for H_ar, X_obs, and XX_obs
   %     nbMC                        : number of learned realizations of (nu,n_d)-valued random matrix [H_ar]    
   %     nu                          : order of the PCA reduction, which is the dimension of H_ar 
   %     MatRx_d(n_x,n_d)            : n_d realizations of X_d (scaled)
   %     MatRxx_d(n_x,n_d)           : n_d realizations of XX_d (unscaled)
   %     MatReta_ar(nu,n_ar)         : n_ar realizations of H_ar 
   %     RmuPCA(nu,1)                : vector of PCA eigenvalues in descending order
   %     MatRVectPCA(n_x,nu)         : matrix of the PCA eigenvectors associated to the eigenvalues loaded in RmuPCA   
   %     Indx_real(nbreal,1)         : nbreal component numbers of XX_ar that are real (positive, negative, or zero) 
   %     Indx_pos(nbpos,1)           : nbpos component numbers of XX_ar that are strictly positive 
   %     Indq_obs(nq_obs,1)          : nq_obs component numbers of QQ that are observed , 1 <= nq_obs <= n_q
   %     Indw_obs(nw_obs,1)          : nw_obs component numbers of WW that are observed,  1 <= nw_obs <= n_w
   %     nx_obs                      : dimension of random vectors XX_obs (unscaled) and X_obs (scaled) (extracted from X_ar)  
   %     Indx_obs(nx_obs,1)          : nx_obs component numbers of X_ar and XX_ar that are observed with nx_obs <= n_x  
   %     ind_scaling                 : = 0 no scaling
   %                                 : = 1    scaling
   %     Rbeta_scale_real(nbreal,1)  : loaded if nbreal >= 1 or = [] if nbreal  = 0               
   %     Ralpha_scale_real(nbreal,1) : loaded if nbreal >= 1 or = [] if nbreal  = 0    
   %     Rbeta_scale_log(nbpos,1)    : loaded if nbpos >= 1  or = [] if nbpos = 0                 
   %     Ralpha_scale_log(nbpos,1)   : loaded if nbpos >= 1  or = [] if nbpos = 0  
   %
   %     nbw0_obs                     : number of vectors Rww0_obs de WW_obs  used for conditional statistics  Q_obs | WW_obs = Rww0_obs
   %     MatRww0_obs(nw_obs,nbw0_obs) : MatRww0_obs(:,kw0) = Rww0_obs_kw0(nw_obs,1)
   %                                  --- for ind_pdf = 1
   %     Ind_Qcomp(nbQcomp,1)         :   pdf of Q_obs(k) | WW_obs = Rww0_obs for k = Ind_Qcomp(kcomp,1), kcomp = 1,...,nbQcomp
   %                                      where 1 <= k < = nq_obs is such that MatRqq_obs(k,:) are the n_ar realizations of Q_obs(k)
   %                                      note that for ind_pdf = 0, Ind_Qcomp = [] 
   %     nbpoint_pdf                  :   number of points in which the pdf is computed
   % 
   %     pc_confregion                : only used if ind_confregion (example pc_confregion =  0.98)
   %
   %     ind_display_screen  : = 0 no display,              = 1 display
   %     ind_print           : = 0 no print,                = 1 print
   %
   %--- OUPUTS  
   %     []

   if ind_display_screen == 1                              
      disp('--- beginning Task13_ConditionalStatistics')
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task13_ConditionalStatistics \n ');
      fprintf(fidlisting,'      \n ');   
      fclose(fidlisting);  
   end

   TimeStartCondStats = tic; 
   numfig = 0;

   %-------------------------------------------------------------------------------------------------------------------------------------   
   %             Checking parameters and data, and loading MatRqq_obs(nq_obs,n_ar) and MatRww_obs(nw_obs,n_ar) 
   %-------------------------------------------------------------------------------------------------------------------------------------
 
   %--- Checking parameters and data
   if n_x <= 0 
      error('STOP1 in sub_conditional_statistics: n_x <= 0');
   end  
   if n_q <= 0 || n_w <= 0
      error('STOP2 in sub_conditional_statistics: n_q <= 0 or n_w <= 0');
   end
   nxtemp = n_q + n_w;                                                 % dimension of random vector XX = (QQ,WW)
   if nxtemp ~= n_x 
      error('STOP3 in sub_conditional_statistics: n_x not equal to n_q + n_w');
   end
   if nbParam <= 0 || nbParam > n_q
      error('STOP4 in sub_conditional_statistics: integer nbParam must be in interval [1,n_q]');
   end   
   if n_d <= 0 
      error('STOP5 in sub_conditional_statistics: n_d <= 0');
   end
   if n_ar <= 0 
      error('STOP6 in sub_conditional_statistics: n_ar <= 0');
   end
   if nbMC <= 0 
      error('STOP7 in sub_conditional_statistics: nbMC <= 0');
   end   

   if nu <= 0 || nu >= n_d
      error('STOP8 in sub_conditional_statistics: nu <= 0 or nu >= n_d');
   end  

   [n1temp,n2temp] = size(MatRx_d);                  %  MatRx_d(n_x,n_d) 
   if n1temp ~= n_x || n2temp ~= n_d
      error('STOP9 in sub_conditional_statistics: dimension error in matrix MatRx_d(n_x,n_d)');
   end
   [n1temp,n2temp] = size(MatRxx_d);                 %  MatRxx_d(n_x,n_d) 
   if n1temp ~= n_x || n2temp ~= n_d
      error('STOP10 in sub_conditional_statistics: dimension error in matrix MatRxx_d(n_x,n_d)');
   end
   [n1temp,n2temp] = size(MatReta_ar);               %  MatReta_ar(nu,n_ar) 
   if n1temp ~= nu || n2temp ~= n_ar
      error('STOP11 in sub_conditional_statistics: dimension error in matrix MatReta_ar(nu,n_ar)');
   end
   [n1temp,n2temp] = size(RmuPCA);                   %  RmuPCA(nu,1) 
   if n1temp ~= nu || n2temp ~= 1
      error('STOP12 in sub_conditional_statistics: dimension error in matrix RmuPCA(nu,1)');
   end
   [n1temp,n2temp] = size(MatRVectPCA);                   %  MatRVectPCA(n_x,nu)
   if n1temp ~= n_x || n2temp ~= nu
      error('STOP13 in sub_conditional_statistics: dimension error in matrix MatRVectPCA(n_x,nu)');
   end

   nbreal = size(Indx_real,1);                           % Indx_real(nbreal,1) 
   if nbreal >= 1
      [n1temp,n2temp] = size(Indx_real);                  
      if n1temp ~= nbreal || n2temp ~= 1
         error('STOP14 in sub_conditional_statistics: dimension error in matrix Indx_real(nbreal,1)');
      end
   end

   nbpos = size(Indx_pos,1);                             % Indx_pos(nbpos,1)
   if nbpos >= 1
      [n1temp,n2temp] = size(Indx_pos);                  
      if n1temp ~= nbpos || n2temp ~= 1
         error('STOP15 in sub_conditional_statistics: dimension error in matrix Indx_pos(nbpos,1)');
      end
   end

   nxtemp = nbreal + nbpos;
   if nxtemp ~= n_x 
       error('STOP16 in sub_conditional_statistics: n_x not equal to nreal + nbpos');
   end

   % Loading dimension nq_obs of Indq_obs(nq_obs,1)
   nq_obs = size(Indq_obs,1);     %  Indq_obs(nq_obs,1)
   
   % Checking input data and parameters of Indq_obs(nq_obs,1) 
   if nq_obs < 1 || nq_obs > n_q
      error('STOP17 in sub_conditional_statistics: nq_obs < 1 or nq_obs > n_q');
   end
   [n1temp,n2temp] = size(Indq_obs);                      % Indq_obs(nq_obs,1)
   if n1temp ~= nq_obs || n2temp ~= 1
      error('STOP18 in sub_conditional_statistics: dimension error in matrix Indq_obs(nq_obs,1)');
   end   
   if length(Indq_obs) ~= length(unique(Indq_obs))
      error('STOP19 in sub_conditional_statistics: there are repetitions in Indq_obs');  
   end
   if any(Indq_obs < 1) || any(Indq_obs > n_q)
      error('STOP20 in sub_conditional_statistics: at least one integer in Indq_obs is not within the valid range');
   end

   % Loading dimension nw_obs of Indw_obs(nw_obs,1)
   nw_obs = size(Indw_obs,1);     %  Indw_obs(nw_obs,1)  

   % Checking input data and parameters of Indw_obs(nw_obs,1)
   if nw_obs < 1 || nw_obs > n_w
      error('STOP21 in sub_conditional_statistics: nw_obs < 1 or nw_obs > n_w');
   end
   [n1temp,n2temp] = size(Indw_obs);                      % Indw_obs(nw_obs,1)
   if n1temp ~= nw_obs || n2temp ~= 1
      error('STOP22 in sub_conditional_statistics: dimension error in matrix Indw_obs(nw_obs,1)')
   end   
   if length(Indw_obs) ~= length(unique(Indw_obs))
      error('STOP23 in sub_conditional_statistics: there are repetitions in Indw_obs');  
   end
   if any(Indw_obs < 1) || any(Indw_obs > n_w)
      error('STOP24 in sub_conditional_statistics: at least one integer in Indw_obs is not within the valid range');
   end

   if nx_obs <= 0 
       error('STOP25 in sub_conditional_statistics: nx_obs <= 0');
   end
   [n1temp,n2temp] = size(Indx_obs);                      % Indx_obs(nx_obs,1)                
   if n1temp ~= nx_obs || n2temp ~= 1
      error('STOP26 in sub_conditional_statistics: dimension error in matrix Indx_obs(nx_obs,1)');
   end
   if ind_scaling ~= 0 && ind_scaling ~= 1
      error('STOP27 in sub_conditional_statistics: ind_scaling must be equal to 0 or to 1');
   end
   if nbreal >= 1 
      [n1temp,n2temp] = size(Rbeta_scale_real);                   % Rbeta_scale_real(nbreal,1)              
      if n1temp ~= nbreal || n2temp ~= 1
         error('STOP28 in sub_conditional_statistics: dimension error in matrix Rbeta_scale_real(nbreal,1) ');
      end
      [n1temp,n2temp] = size(Ralpha_scale_real);                   % Ralpha_scale_real(nbreal,1)              
      if n1temp ~= nbreal || n2temp ~= 1
         error('STOP29 in sub_conditional_statistics: dimension error in matrix Ralpha_scale_real(nbreal,1) ');
      end                    
   end
   if nbpos >= 1 
      [n1temp,n2temp] = size(Rbeta_scale_log);                     % Rbeta_scale_log(nbpos,1)              
      if n1temp ~= nbpos || n2temp ~= 1
         error('STOP30 in sub_conditional_statistics: dimension error in matrix Rbeta_scale_log(nbpos,1) ');
      end
      [n1temp,n2temp] = size(Ralpha_scale_log);                    % Ralpha_scale_log(nbpos,1)              
      if n1temp ~= nbpos || n2temp ~= 1
         error('STOP31 in sub_conditional_statistics: dimension error in matrix Ralpha_scale_log(nbpos,1) ');
      end                    
   end
   if nbw0_obs <= 0
      error('STOP32 in sub_conditional_statistics: nbw0_obs must be greater than or equal to 1');
   end
   [n1temp,n2temp] = size(MatRww0_obs);                    % MatRww0_obs(nw_obs,nbw0_obs)            
   if n1temp ~= nw_obs || n2temp ~= nbw0_obs
      error('STOP33 in sub_conditional_statistics: dimension error in matrix MatRww0_obs(nw_obs,nbw0_obs) ');
   end  
   if ind_pdf == 0  % Check if matrix Ind_Qcomp is empty 
      if isempty(Ind_Qcomp) == 0
         error('STOP34 in sub_conditional_statistics: for ind_pdf = 0, matrix Ind_Qcomp must be empty ');
      end
      if nbpoint_pdf ~= 0
         error('STOP35 in sub_conditional_statistics: for ind_pdf = 0, nbpoint_pdf must be equal to 0 ');
      end
   end
   if ind_pdf == 1  
  
      [nbQcomp,n2temp] = size(Ind_Qcomp);                    % Ind_Qcomp(nbQcomp,1)       
      if n2temp ~= 1
         error('STOP36 in sub_conditional_statistics: dimension error in matrix Ind_Qcomp(nbQcomp,1)');
      end  

      for kcomp = 1:nbQcomp 
          kk = Ind_Qcomp(kcomp,1);        % Ind_Qcomp(nbQcomp,1)
          ind_error = 0;
          for iqobs = 1:nq_obs
              if kk == Indq_obs(iqobs,1)  % Indq_obs(nq_obs,1)
                 ind_error = 1;
                 break
              end
          end
          if ind_error == 0
              error('STOP37 in sub_conditional_statistics: some values in Ind_Qcomp(nbQcomp,1) are not in Indq_obs(nq_obs,1)');
          end 
      end

      if nbpoint_pdf < 1
         error('STOP38 in sub_conditional_statistics: for ind_pdf = 1, nbpoint_pdf must be greater than of equal to 1 ');
      end
   end
   if ind_confregion == 1
      if pc_confregion <= 0 || pc_confregion >= 1
         error('STOP39 in sub_conditional_statistics: for ind_confregion = 1, pc_confregion must be in ]0,1[ ');
      end
   end

   % Case nbParam = 1
   if nbParam == 1      
      nqobsPhys = nq_obs;
   end
   
   % Checking that, if nbParam > 1, nq_obs is equal to nqobsPhys*nbParam, in which nqobsPhys is the number of the components of 
   % the state variables that are observed. Consequently, nq_obs/nbParam must be an integer, if not there is an error in the given 
   % value of nbParam of in the Data generation in "mainWorkflow_Data_generation1.m" and "mainWorkflow_Data_generation2.m" 

   if (nbParam > 1) && (ind_mean == 1 || ind_mean_som  == 1 || ind_confregion == 1)
      % calculate nqobsPhys and check if it is an integer
      nqobsPhys = nq_obs/nbParam;
      % test if nqobsPhys is an integer
      if mod(nq_obs, nbParam) ~= 0
         error(['STOP40 in sub_conditional_statistics: nq_obs divided by nbParam is not an integer. ', ...
           'There is either an error in the given value of nbParam, or in the data generation ', ...
           'in function mainWorkflow_Data_generation1.m and function mainWorkflow_Data_generation2.m']);
      end
   end
   
   %--- PCA back: MatRx_obs(nx_obs,n_ar)
   [MatRx_obs] = sub_conditional_PCAback(n_x,n_d,nu,n_ar,nx_obs,MatRx_d,MatReta_ar,Indx_obs,RmuPCA,MatRVectPCA, ...
                             ind_display_screen,ind_print);

   %--- Scaling back: MatRxx_obs(nx_obs,n_ar)
   [MatRxx_obs] = sub_conditional_scalingBack(nx_obs,n_x,n_ar,MatRx_obs,Indx_real,Indx_pos,Indx_obs,Rbeta_scale_real,Ralpha_scale_real, ...
                                  Rbeta_scale_log,Ralpha_scale_log,ind_display_screen,ind_print,ind_scaling); 
   clear MatRx_obs

   %--- Loading MatRqq_obs(nq_obs,n_ar) and MatRww_obs(nw_obs,n_ar) from MatRxx_obs(nx_obs,n_ar) 
   MatRqq_obs = MatRxx_obs(1:nq_obs,1:n_ar);                  % MatRqq_obs(nq_obs,n_ar),MatRxx_obs(nx_obs,n_ar) 
   MatRww_obs = MatRxx_obs(nq_obs+1:nq_obs+nw_obs,1:n_ar);    % MatRww_obs(nw_obs,n_ar),MatRxx_obs(nx_obs,n_ar) 
   clear MatRxx_obs 

   %--- Construction of MatRqq_d_obs(nq_obs,n_d) and MatRww_d_obs(nw_obs,n_d) from MatRxx_d(n_x,n_d) 
   MatRxx_d_obs = MatRxx_d(Indx_obs,:);                          % MatRxx_d_obs(nx_obs,n_d),MatRxx_d(n_x,n_d),Indx_obs(nx_obs,1)  
   MatRqq_d_obs = MatRxx_d_obs(1:nq_obs,1:n_d);                  % MatRqq_d_obs(nq_obs,n_d),MatRxx_d_obs(nx_obs,n_d) 
   MatRww_d_obs = MatRxx_d_obs(nq_obs+1:nq_obs+nw_obs,1:n_d);    % MatRww_d_obs(nw_obs,n_d),MatRxx_d_obs(nx_obs,n_d) 
   clear MatRxx_d_obs

   %--- print  
   if ind_print == 1
      fidlisting=fopen('listing.txt','a+'); 
      fprintf(fidlisting,'                                \n '); 
      fprintf(fidlisting,' ind_mean      = %9i \n ',ind_mean); 
      fprintf(fidlisting,' ind_mean_som  = %9i \n ',ind_mean_som); 
      fprintf(fidlisting,' ind_pdf       = %9i \n ',ind_pdf); 
      fprintf(fidlisting,' ind_confregion= %9i \n ',ind_confregion); 
      fprintf(fidlisting,'                                \n '); 
      fprintf(fidlisting,' n_q           = %9i \n ',n_q);   
      fprintf(fidlisting,' nbParam       = %9i \n ',nbParam);   
      fprintf(fidlisting,' n_w           = %9i \n ',n_w);
      fprintf(fidlisting,' n_x           = %9i \n ',n_x);   
      fprintf(fidlisting,' nbreal        = %9i \n ',nbreal);
      fprintf(fidlisting,' nbpos         = %9i \n ',nbpos);
      fprintf(fidlisting,'                                \n '); 
      fprintf(fidlisting,' nq_obs        = %9i \n ',nq_obs);
      fprintf(fidlisting,' nbParam       = %9i \n ',nbParam);
      fprintf(fidlisting,' nqobsPhys     = %9i \n ',nqobsPhys);
      fprintf(fidlisting,' nw_obs        = %9i \n ',nw_obs); nqobsPhys = nq_obs/nbParam;
      fprintf(fidlisting,'                                \n ');
      fprintf(fidlisting,' ind_scaling   = %9i \n ',ind_scaling);    
      fprintf(fidlisting,'                                \n '); 
      fprintf(fidlisting,' n_d           = %9i \n ',n_d);
      fprintf(fidlisting,' nbMC          = %9i \n ',nbMC);
      fprintf(fidlisting,' n_ar          = %9i \n ',n_ar);    
      fprintf(fidlisting,' nu            = %9i \n ',nu);
      fprintf(fidlisting,'                                \n ');       
      fprintf(fidlisting,' nbw0_obs      = %9i \n ',nbw0_obs);
      fprintf(fidlisting,'                                \n '); 
      if ind_pdf == 1
         fprintf(fidlisting,' nbQcomp       = %9i \n ',nbQcomp);
         fprintf(fidlisting,' nbpoint_pdf   = %9i \n ', nbpoint_pdf);
      end
      if ind_confregion == 1
         fprintf(fidlisting,' pc_confregion = %9i \n ',pc_confregion );  
      end
      fprintf(fidlisting,'                                \n '); 
      fprintf(fidlisting,' ind_display_screen = %1i \n ',ind_display_screen); 
      fprintf(fidlisting,' ind_print          = %1i \n ',ind_print); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting); 
   end

   %-------------------------------------------------------------------------------------------------------------------------------------   
   %                             Computing and plot of conditional statistics
   %-------------------------------------------------------------------------------------------------------------------------------------
   % nqobsPhys = nq_obs/nbParam;
   if nbParam == 1                   
      mobsPhys             = 1;
      mq                   = nq_obs;
      ArrayMatRqq(:,:,1)   = MatRqq_obs;   % ArrayMatRqq(mq,n_ar,1),MatRqq_obs(nq_obs,n_ar) 
      ArrayMatRqq_d(:,:,1) = MatRqq_d_obs; % ArrayMatRqq_d(mq,n_d,1),MatRqq_d_obs(nq_obs,n_d) 
   end
   if nbParam > 1
      mobsPhys      = nqobsPhys;
      mq            = nbParam;
      ArrayMatRqq   = zeros(mq,n_ar,nqobsPhys);
      ArrayMatRqq_d = zeros(mq,n_d,nqobsPhys);
      istart   = 1;
      for iobsPhys = 1:nqobsPhys
          iend = istart + nbParam - 1;
          ArrayMatRqq(:,:,iobsPhys)   = MatRqq_obs(istart:iend,:);
          ArrayMatRqq_d(:,:,iobsPhys) = MatRqq_d_obs(istart:iend,:);
          istart = iend + 1;
      end
   end

   for kw0 = 1:nbw0_obs               % nbw0_obs: number of Rww0_obs for WW_obs used for conditional statistics Q_obs | WW_obs = Rww0_obs
                                      %           and Q_d_obs | WW_obs = Rww0_obs
       Rww0_obs = MatRww0_obs(:,kw0); % Rww0_obs(nw_obs,1),MatRww0_obs(nw_obs,nw0_obs) 
      
      %--- Estimation of the conditional mean
      if ind_mean == 1  
         for iobsPhys = 1:mobsPhys
            MatRqq   = ArrayMatRqq(:,:,iobsPhys);
            MatRqq_d = ArrayMatRqq_d(:,:,iobsPhys);
            [REqq_ww0]   = sub_conditional_mean(nw_obs,mq,n_ar,MatRqq,MatRww_obs,Rww0_obs);  
            [REqq_d_ww0] = sub_conditional_mean(nw_obs,mq,n_d,MatRqq_d,MatRww_d_obs,Rww0_obs);  
            % Plot
            Rk = (1:1:mq)';
            h=figure; 
            plot(Rk,REqq_ww0,'LineStyle','-','LineWidth',1,'Color','b');  
            hold on
            plot(Rk,REqq_d_ww0,'k-');
            xlabel(['$k$'],'FontSize',16,'Interpreter','latex'); 
            ylabel(['$E\{ Q_k \vert W = w_0(:,',num2str(kw0),')\}$'],'FontSize',16,'Interpreter','latex'); 
            title(['$E\{ Q_k$ (blue thick) and ${Q_d}_k$ (black thin) $\vert W = w_0(:,',num2str(kw0),')\}$'], ...
                    'FontSize',16,'Interpreter','latex','FontWeight','normal');
            numfig = numfig + 1;
            saveas(h,['figure_CondStats',num2str(numfig),'_mean_w0_',num2str(kw0),'_obsPhys_',num2str(iobsPhys),'.fig']); 
            hold off
            close(h);
         end
      end
   
      %--- Estimation of the conditional mean and second-order moment
      if ind_mean_som == 1
         for iobsPhys = 1:mobsPhys
            MatRqq   = ArrayMatRqq(:,:,iobsPhys);
            MatRqq_d = ArrayMatRqq_d(:,:,iobsPhys);
            [REqq_ww0,REqq2_ww0]     = sub_conditional_second_order_moment(nw_obs,mq,n_ar,MatRqq,MatRww_obs,Rww0_obs);
            [REqq_d_ww0,REqq2_d_ww0] = sub_conditional_second_order_moment(nw_obs,mq,n_d,MatRqq_d,MatRww_d_obs,Rww0_obs);

            % Plot mean value
            Rk = (1:1:mq)';
            h=figure; 
            plot(Rk,REqq_ww0,'LineWidth',1,'color','b');
            hold on
            plot(Rk,REqq_d_ww0,'k-');
            xlabel(['$k$'],'FontSize',16,'Interpreter','latex'); 
            ylabel(['$E\{ Q_k \vert W = w_0(:,',num2str(kw0),')\}$'],'FontSize',16,'Interpreter','latex'); 
            title(['$E\{ Q_k$ (blue thick) and ${Q_d}_k$ (black thin) $\vert W = w_0(:,',num2str(kw0),')\}$'], ...
                    'FontSize',16,'Interpreter','latex','FontWeight','normal');
            numfig = numfig + 1;
            saveas(h,['figure_CondStats',num2str(numfig),'_mean_w0_',num2str(kw0),'_obsPhys_',num2str(iobsPhys),'.fig']); 
            hold off
            close(h);
            
            % Plot second-order moment
            Rk = (1:1:mq)';
            h=figure; 
            plot(Rk,REqq2_ww0,'LineWidth',1,'color','b');
            hold on
            plot(Rk,REqq2_d_ww0,'k-');
            xlabel(['$k$'],'FontSize',16,'Interpreter','latex'); 
            ylabel(['$E\{ Q_k^2 \vert W = w_0(:,',num2str(kw0),')\}$'],'FontSize',16,'Interpreter','latex'); 
            title(['$E\{ Q_k^2$ (blue thick) and ${Q_d^2}_k$ (black thin) $\vert W = w_0(:,',num2str(kw0),')\}$'], ...
                    'FontSize',16,'Interpreter','latex','FontWeight','normal');
            numfig = numfig + 1;
            saveas(h,['figure_CondStats',num2str(numfig),'_mean_som_w0_',num2str(kw0),'_obsPhys_',num2str(iobsPhys),'.fig']); 
            hold off
            close(h);
         end
      end
 
      %--- Estimation of the conditional pdf of component k <= nq_obs
      if ind_pdf == 1
         for kcomp = 1:nbQcomp 
             kk = Ind_Qcomp(kcomp,1);        % Ind_Qcomp(nbQcomp,1)
             for iqobs = 1:nq_obs
                 if kk == Indq_obs(iqobs,1)
                    k = iqobs;
                    break
                 end
             end
             MatRqq_obs_k   = MatRqq_obs(k,:);             % MatRqq_obs_k(1,n_ar)
             MatRqq_d_obs_k = MatRqq_d_obs(k,:);           % MatRqq_d_obs_k(1,n_d)
             [~,Rq,Rpdfqq_ww0]     = sub_conditional_pdf(nw_obs,n_ar,MatRqq_obs_k,MatRww_obs,Rww0_obs,nbpoint_pdf); 
             [~,Rq_d,Rpdfqq_d_ww0] = sub_conditional_pdf(nw_obs,n_d,MatRqq_d_obs_k,MatRww_d_obs,Rww0_obs,nbpoint_pdf); 

             % Plot  
             h=figure; 
             plot(Rq,Rpdfqq_ww0,'LineWidth',1,'color','b');
             hold on
             plot(Rq_d,Rpdfqq_d_ww0,'k-');
             xlabel(['$q_{',num2str(kk),'}$'],'FontSize',16,'Interpreter','latex'); 
             ylabel(['$p_{Q_{',num2str(kk),'}\vert W}(q_{',num2str(kk),'})$'],'FontSize',16,'Interpreter','latex'); 
             title(['pdf of $Q_{',num2str(kk),'}$ (blue thick) and of ${Q_d}_{',num2str(kk),'}$ (black thin) $\vert W = w_0(:,',num2str(kw0),')$'],  ...
                      'FontSize',16,'Interpreter','latex','FontWeight','normal');
             numfig = numfig + 1;
             saveas(h,['figure_CondStats',num2str(numfig),'_pdfQ',num2str(kk),'_w0_',num2str(kw0),'.fig']); 
             hold off
             close(h); 
         end
      end
   
      %--- Estimation of the conditional confidence region
      if ind_confregion == 1   
         for iobsPhys = 1:mobsPhys
            MatRqq = ArrayMatRqq(:,:,iobsPhys);

            % Confidence region computation
            [RqqLower_ww0,RqqUpper_ww0] = sub_conditional_confidence_interval(nw_obs,mq,n_ar,MatRqq,MatRww_obs,Rww0_obs,pc_confregion); 
            RDplus     = RqqUpper_ww0';             
            RDmoins    = RqqLower_ww0';
            Rkp        = (1:1:mq)';
            Rkm        = (mq:-1:1)'; 
            RDmoinsinv = RDmoins(Rkm);
            Rpm        = [Rkp
                          Rkm];
            RDrc       = [RDplus'
                          RDmoinsinv'];

            % Mean value computation of the conditional mean with the lean dataset and the training dataset
            MatRqq_d = ArrayMatRqq_d(:,:,iobsPhys);
            [REqq_ww0]   = sub_conditional_mean(nw_obs,mq,n_ar,MatRqq,MatRww_obs,Rww0_obs);  
            [REqq_d_ww0] = sub_conditional_mean(nw_obs,mq,n_d,MatRqq_d,MatRww_d_obs,Rww0_obs);  

            % Plot
            Rk = (1:1:mq)';
            h=figure; 
            hold on
            fill(Rpm,RDrc,'y','LineWidth',1,'FaceColor',[1 1 0],'EdgeColor',[0.850980392156863 0.325490196078431 0.0980392156862745]); 
            plot(Rk,REqq_ww0,'LineStyle','-','LineWidth',1,'Color','b');  
            plot(Rk,REqq_d_ww0,'k-'); 
            xlabel(['$k$'],'FontSize',16,'Interpreter','latex'); 
            ylabel(['$\{ Q_k \vert W = w_0(:,',num2str(kw0),')\}$'],'FontSize',16,'Interpreter','latex'); 
            title(['Confidence region of $\{ Q_k \vert W = w_0(:,',num2str(kw0),')\}$ and conditional ';'meanvalue of $Q_k$ (blue thick) and ${Q_d}_k$ (black thin)         '],...
                    'FontSize',16,'Interpreter','latex');
            numfig = numfig + 1;
            saveas(h,['figure_CondStats',num2str(numfig),'_confregion_w0_',num2str(kw0),'_obsPhys_',num2str(iobsPhys),'.fig']); 
            hold off
            close(h);
         end
      end
   end

   ElapsedTimeCondStats = toc(TimeStartCondStats);   

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ----- Elapsed time for Task13_ConditionalStatistics \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' Elapsed Time   =  %10.2f\n',ElapsedTimeCondStats);   
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task13_ConditionalStatistics')
   end    
   
   return
end





