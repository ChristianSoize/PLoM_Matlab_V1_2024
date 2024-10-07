
function [n_x,Indx_real,Indx_pos,nx_obs,Indx_obs,Indx_targ_real,Indx_targ_pos,nx_targ,Indx_targ] =  ...
             sub_data_structure_and_check(n_q,Indq_real,Indq_pos,n_w,Indw_real,Indw_pos,n_d,MatRxx_d,Indq_obs,Indw_obs, ...
                                          ind_display_screen,ind_print,ind_exec_solver,Indq_targ_real,Indq_targ_pos, ...
                                          Indw_targ_real,Indw_targ_pos,ind_type_targ,N_r,MatRxx_targ_real,MatRxx_targ_pos, ...
                                          Rmeanxx_targ,MatRcovxx_targ)
        
   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 01 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_data_structure_and_check
   %  Subject      : for XX = (QQ,WW) and the associated observation XX_obs = (QQ_obs,WW_obs), construction of the information structure 
   %                 that is used for 
   %                 - the training dataset with n_d realizations 
   %                 - the target datasets with given second-order moments or N_r realizations
   %                 - the learned dataset with n_ar realizations  (with or without targets)
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
   %          n_q                    : dimension of random vector QQ (unscaled quantitity of interest)  1 <= n_q     
   %          Indq_real(nqreal,1)    : contains the nqreal component numbers of QQ, which are real (positive, negative, or zero) 
   %                                   with 0 <= nqreal <=  n_q and for which a "standard scaling" will be used
   %          Indq_pos(nqpos,1)      : contains the nqpos component numbers of QQ, which are strictly positive a "specific scaling"
   %                                   with  0 <= nqpos <=  n_q  and for which the scaling is {log + "standard scaling"}
   %                                   --- we must have n_q = nqreal + nqpos 
   %
   %          n_w                    : dimension of random vector WW (unscaled control variable) with 0 <= n_w.
   %                                   n_w = 0: unsupervised case, n_w >= 1: supervised case
   %          Indw_real(nwreal,1)    : contains the nwreal component numbers of WW, which are real (positive, negative, or zero) 
   %                                   with 0 <= nwreal <=  n_w and for which a "standard scaling" will be used
   %          Indw_pos(nwpos,1)      : contains the nwpos component numbers of WW, which are strictly positive a "specific scaling"
   %                                   with  0 <= nwpos <=  n_w  and for which the scaling is {log + "standard scaling"}
   %                                   --- we must have n_w = nwreal + nwpos
   %
   %          n_d                    : number of points in the training set for XX_d and X_d
   %          MatRxx_d(n_x,n_d)      : n_d realizations of random vector XX_d (unscale) with dimension n_x = n_q + n_w
   %
   %          Indq_obs(nq_obs,1)     : nq_obs component numbers of QQ that are observed , 1 <= nq_obs <= n_q
   %          Indw_obs(nw_obs,1)     : nw_obs component numbers of WW that are observed,  1 <= nw_obs <= n_w
   %                                 --- we must have nx_obs = nq_obs + nw_obs <= n_x
   %
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
   %                              an integer, if not there is an errod in the Data generation in "mainWorkflow_Data_generation1.m" and 
   %                              "mainWorkflow_Data_generation2.m" 
   %
   %                     WARNING: NOTE THAT if such a parameter does not exist, it must be considered that nbParam = 1, but the 
   %                              information structure must be consistent with the case nbParam > 1.  
   %  
   %
   %          ind_display_screen    : = 0 no display,            = 1 display
   %          ind_print             : = 0 no print,              = 1 print
   %
   %          ind_exec_solver       : = 1 Direct analysis : giving a training dataset, generating a learned dataset
   %                                : = 2 Inverse analysis: giving a training dataset and a target dataset, generating a learned dataset
   %
   %                                --- data structure for the taget datasets used if ind_exec_solver = 2:  QQ_targ = (QQ_targ,WW_targ)
   %          ind_type_targ         : = 1, targets defined by giving N_r realizations
   %                                : = 2, targets defined by giving target mean-values 
   %                                : = 3, targets defined by giving target mean-values and target variance-values
   %
   %          Indq_targ_real(nqreal_targ,1): nqreal_targ component numbers of QQ for which a target is real, 0 <= nqreal_targ <= n_q
   %          Indq_targ_pos(nqpos_targ,1)  : nqpos_targ component numbers of QQ for which a target is positive, 0 <= nqpos_targ <= n_q                                            
   %
   %          Indw_targ_real(nwreal_targ,1): nwreal_targ component numbers of WW for which a target is real, 0 <= nwreal_targ <= n_w
   %          Indw_targ_pos(nwpos_targ,1)  : nwpos_targ  component numbers of WW for which a target is positive, 0 <= nwpos_targ <= n_w
   %
   %                                         Indx_targ_real = [Indq_targ_real           % Indx_targ_real(nbreal_targ,1)
   %                                                           n_q + Indw_targ_real];   % nbreal_targ component numbers of XX, which are real
   %                                         Indx_targ_pos  = [Indq_targ_pos            % Indx_targ_pos(nbpos_targ,1)
   %                                                           n_q + Indw_targ_pos];    % nbpos_targ component numbers of XX, 
   %                                                                                    % which are strictly positive
   %                                         nx_targ        = nbreal_targ + nbpos_targ; % dimension of random vector XX_targ = (QQ_targ,WW_targ)
   %                                         Indx_targ      = [Indx_targ_real           % nx_targ component numbers of XX_targ 
   %                                                           Indx_targ_pos];          % for which a target is given 
   %
   %                                     WARNING: if ind_exec_solver = 2 and ind_type_targ = 2 or 3, all the components of XX and XX_targ
   %                                              must be considered as real even if some components are positive. When thus must have:
   %                                              nqreal     = n_q and nwreal     = n_w
   %                                              nqpos      = 0   and nwpos      = 0
   %                                              nqpos_targ = 0   and nwpos_targ = 0
   %
   %          N_r                               : number of target realizations 
   %
   %                                            --- ind_type_targ = 1: targets defined by giving N_r realizations                 
   %          MatRxx_targ_real(nbreal_targ,N_r) : N_r realizations (unscaled) of the nbreal_targ targets of XX that are real
   %          MatRxx_targ_pos(nbpos_targ,N_r)   : N_r realizations (unscaled) of the nbpos_targ targets of XX that are positive
   %  
   %                                            --- ind_type_targ = 2 or 3: targets defined by giving the mean value of unscaled XX_targ 
   %          Rmeanxx_targ(nx_targ,1)           : nx_targ components of mean value E{XX_targ} of vector-valued random target XX_targ
   %
   %                                            --- ind_type_targ = 3: targets defined by giving the covariance matrix of unscaled XX_targ 
   %          MatRcovxx_targ(nx_targ,nx_targ)   : covariance matrix of XX_targ 
   %
   %--- OUPUTS  
   %          n_x                      : dimension of random vector XX = (QQ,WW) (unscaled), n_x = n_q + n_w     
   %                                     We must have n_x = nbreal + nbpos
   %          Indx_real(nbreal,1)      : contains the nbreal component numbers of XX, which are real (positive, negative, or zero) 
   %                                     with 0 <= nbreal <=  n_x and for which a "standard scaling" will be used
   %          Indx_pos(nbpos,1)        : contains the nbpos component numbers of XX, which are strictly positive a "specific scaling"
   %                                     with  0 <= nbpos <=  n_x  and for which the scaling is {log + "standard scaling"}
   %
   %          nx_obs                   : number of components of XX that are observed with nx_obs = nq_obs + nw_obs <= n_x
   %          Indx_obs(nx_obs,1)       : nx_obs component numbers of XX that are observed 
   %  
   %                                          --- if ind_exec_solver = 2:  XX_targ = (XX_targ_real,XX_targ_pos)
   %                                              we must have nx_targ = nbreal_targ + nbpos_targ
   %          Indx_targ_real(nbreal_targ,1)   : contains the nbreal_targ component numbers of XX, which are real  
   %                                            with 0 <= nbreal_targ <= n_x and for which a "standard scaling" will be used
   %          Indx_targ_pos(nbpos_targ,1)     : contains the nbpos_targ component numbers of XX, which are strictly positive (specific scaling)
   %                                            with 0 <= nbpos_targ <= n_x  and for which the scaling is {log + "standard scaling"}   %
   %          nx_targ                         : number of components of XX with targets such that nx_targ = nbreal_targ + nbpos_targ <= n_x
   %          Indx_targ(nx_targ,1)            : nx_targ component numbers of XX with targets 
   %
   %--- COMMENTS
   %          If ind_exec_solver = 2 and ind_type_targ = 2 or 3, 
   %          all the components of XX and XX_targ are considered as real even if some components are positive.    
   %          When thus have:  nx_targ = nbreal_targ, nbpos_targ = 0, and Indx_targ(nx_targ,1) = Indx_targ_real(nbreal_targ,1)   

   if ind_display_screen == 1                              
      disp('--- beginning Task1_DataStructureCheck');
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task1_DataStructureCheck \n ');
      fprintf(fidlisting,'      \n ');  
      fclose(fidlisting);  
   end

   TimeStartDataCheck = tic; 

   % Checking ind_type_targ 
   if ind_exec_solver == 2 && (ind_type_targ ~= 1 && ind_type_targ ~= 2  && ind_type_targ ~= 3)
      error('STOP1 in sub_data_structure_and_check: for an inverse problem, ind_type_targ  must be equal to 1, 2 or 3');
   end

   %--- Checking n_q and n_w
   if n_q <= 0 || n_w < 0
       error('STOP2 in sub_data_structure_and_check: n_q <= 0 or n_w < 0');
   end

   %--- Loading dimensions nq_obs, nqreal, and nqpos of Indq_obs(nq_obs,1), Indq_real(nqreal,1), and Indq_pos(nqpos,1)
   nq_obs = size(Indq_obs,1);     %  Indq_obs(nq_obs,1)
   nqreal = size(Indq_real,1);    %  Indq_real(nqreal,1) 
   nqpos  = size(Indq_pos,1);     %  Indq_pos(nqpos,1)     
   
   %--- Checking input data and parameters of Indq_obs(nq_obs,1), Indq_real(nqreal,1) and Indq_pos(nqpos,1)   
   if nq_obs < 1 || nq_obs > n_q
      error('STOP3 in sub_data_structure_and_check: nq_obs < 1 or nq_obs > n_q');
   end
   [n1temp,n2temp] = size(Indq_obs);                      % Indq_obs(nq_obs,1)
   if n1temp ~= nq_obs || n2temp ~= 1
      error('STOP4 in sub_data_structure_and_check: dimension error in matrix Indq_obs(nq_obs,1)');
   end   
   if length(Indq_obs) ~= length(unique(Indq_obs))
      error('STOP5 in sub_data_structure_and_check: there are repetitions in Indq_obs');  
   end
   if any(Indq_obs < 1) || any(Indq_obs > n_q)
      error('STOP6 in sub_data_structure_and_check: at least one integer in Indq_obs is not within the valid range');
   end
   if nqreal + nqpos ~= n_q
      error('STOP7 in sub_data_structure_and_check: nqreal + nqpos ~= n_q');
   end 
   if nqreal >= 1   
      if length(Indq_real) ~= length(unique(Indq_real))
         error('STOP8 in sub_data_structure_and_check: there are repetitions in Indq_real');               
      end
      if any(Indq_real < 1) || any(Indq_real > n_q)
         error('STOP9 in sub_data_structure_and_check: at least one  integer in Indq_real is not within the valid range');                
      end
   end
   if nqpos >= 1  
      if ind_exec_solver == 2 && (ind_type_targ == 2 || ind_type_targ == 3)
         error('STOP10 in sub_data_structure_and_check: for ind_exec_solver = 2 and ind_type_targ = 2 or 3, we must have nqpos = 0');  
      end
      if length(Indq_pos) ~= length(unique(Indq_pos))
         error('STOP11 in sub_data_structure_and_check: there are repetitions in Indq_pos');  
      end
       if any(Indq_pos < 1) || any(Indq_pos > n_q)
         error('STOP12 in sub_data_structure_and_check: at least one  integer in Indq_pos is not within the valid range');    
      end
   end
   if nqreal >= 1 && nqpos >= 1                            % Check that all integers in Indq_real are different from those in Indq_pos
      if isempty(intersect(Indq_real, Indq_pos))
         combined_list = [Indq_real; Indq_pos];         % Check that the union of both lists is exactly 1:n_q without missing or repetition
         if length(combined_list) ~= n_q || ~all(sort(combined_list) == (1:n_q)')
            error('STOP13 in sub_data_structure_and_check: the union of Indq_real with Indq_pos is not equal to the set (1:n_q)');
         end
      else
         error('STOP14 in sub_data_structure_and_check: there are common integers in Indq_real and Indq_pos');
      end
   end

   %--- Loading dimensions nw_obs, nwreal, and nwpos of Indw_obs(nw_obs,1), Indw_real(nwreal,1), and Indw_pos(nwpos,1)
   nw_obs = size(Indw_obs,1);     %  Indw_obs(nw_obs,1)
   nwreal = size(Indw_real,1);    %  Indw_real(nwreal,1)
   nwpos  = size(Indw_pos,1);     %  Indw_pos(nwpos,1)  
   
   %--- Checking input data and parameters of Indw_obs(nw_obs,1), Indw_real(nwreal,1) and Indw_pos(nwpos,1)   
   if n_w >= 1        % Supervised case
      if nw_obs < 1 || nw_obs > n_w
         error('STOP15 in sub_data_structure_and_check: nw_obs < 1 or nw_obs > n_w');
      end
      [n1temp,n2temp] = size(Indw_obs);                      % Indw_obs(nw_obs,1)
      if n1temp ~= nw_obs || n2temp ~= 1
         error('STOP16 in sub_data_structure_and_check: dimension error in matrix Indw_obs(nw_obs,1)');
      end   
      if length(Indw_obs) ~= length(unique(Indw_obs))
         error('STOP17 in sub_data_structure_and_check: there are repetitions in Indw_obs');  
      end
      if any(Indw_obs < 1) || any(Indw_obs > n_w)
         error('STOP18 in sub_data_structure_and_check: at least one integer in Indw_obs is not within the valid range');
      end
      if nwreal + nwpos ~= n_w
         error('STOP19 in sub_data_structure_and_check: nwreal + nwpos ~= n_w');
      end 
      if nwreal >= 1   
         if length(Indw_real) ~= length(unique(Indw_real))
            error('STOP20 in sub_data_structure_and_check: there are repetitions in Indw_real');               
         end
         if any(Indw_real < 1) || any(Indw_real > n_w)
            error('STOP21 in sub_data_structure_and_check: at least one  integer in Indw_real is not within the valid range');                
         end
      end
      if nwpos >= 1   
         if ind_exec_solver == 2 && (ind_type_targ == 2 || ind_type_targ == 3)
            error('STOP22 in sub_data_structure_and_check: for ind_exec_solver = 2 and ind_type_targ = 2 or 3, we must have nwpos = 0');  
         end
         if length(Indw_pos) ~= length(unique(Indw_pos))
            error('STOP23 in sub_data_structure_and_check: there are repetitions in Indw_pos');  
         end
          if any(Indw_pos < 1) || any(Indw_pos > n_w)
            error('STOP24 in sub_data_structure_and_check: at least one  integer in Indw_pos is not within the valid range');    
         end
      end
      if nwreal >= 1 && nwpos >= 1                         % Check that all integers in Indw_real are different from those in Indw_pos
         if isempty(intersect(Indw_real, Indw_pos))
            combined_list = [Indw_real; Indw_pos];         % Check that the union of both lists is exactly 1:n_w without missing or repetition
            if length(combined_list) ~= n_w || ~all(sort(combined_list) == (1:n_w)')
               error('STOP25 in sub_data_structure_and_check: the union of Indw_real with Indw_pos is not equal to the set (1:n_w)');
            end
         else
            error('STOP26 in sub_data_structure_and_check: there are common integers in Indw_real and Indw_pos');
         end
      end
   end

   if ind_exec_solver ~= 1 && ind_exec_solver ~= 2
      error('STOP27 in sub_data_structure_and_check: ind_exec_solver must be equal to 1 or equal to 2');
   end

   %--- Loading dimensions  nqreal_targ, nqpos_targ, nwreal_targ, nwpos_targ 
   %    if ind_exec_solver = 1 (no targets) matrices Indq_targ_real,Indq_targ_pos,Indw_targ_real,Indw_targ_pos must be equal to []
   %    if ind_exec_solver = 2 and ind_type_targ = 2 or 3, Indq_targ_pos and Indw_targ_pos must be equal to [] 
   nqreal_targ = size(Indq_targ_real,1);    % Indq_targ_real(nqreal_targ,1)
   nqpos_targ  = size(Indq_targ_pos,1);     % Indq_targ_pos(nqpos_targ,1)
   nwreal_targ = size(Indw_targ_real,1);    % Indw_targ_real(nwreal_targ,1)
   nwpos_targ  = size(Indw_targ_pos,1);     % Indw_targ_pos(nwpos_targ,1)

   %--- Checking data for ind_exec_solver = 2 and ind_type_targ = 2 or = 3: inverse problem with given targets defined by moments.
   %    For such a scale all the components must be considered as real even if there are positive-valued components.
   if ind_exec_solver == 2 && (ind_type_targ == 2 || ind_type_targ == 3)
      if nqpos_targ ~= 0 && nwpos_targ ~= 0
         error(['STOP28 in sub_data_structure_and_check: for an inverse problem and if ind_type_targ = 2 or 3, ' ...
                'then we must have nqpos_targ = nwpos_targ = 0']);
      end
   end

   %--- Checking data for ind_exec_solver = 2 (inverse problem with given targets)
   if ind_exec_solver == 2
      nbreal_targ = nqreal_targ + nwreal_targ;
      nbpos_targ  = nqpos_targ  + nwpos_targ;
      ntemp       = nbreal_targ + nbpos_targ;
      if ntemp == 0
         error('STOP29 in sub_data_structure_and_check: for an inverse problem, at least 1 component of XX must have a target');
      end

      %--- Checking input data and parameters of  Indq_targ_real(nqreal_targ,1) and Indq_targ_pos(nqpos_targ,1)  
      if nqreal_targ >= 1   
         if length(Indq_targ_real) ~= length(unique(Indq_targ_real))
            error('STOP30 in sub_data_structure_and_check: there are repetitions in Indq_targ_real');               
         end
         if any(Indq_targ_real < 1) || any(Indq_targ_real > n_q)
            error('STOP31 in sub_data_structure_and_check: at least one  integer in Indq_targ_real is not within the valid range');                
         end
         is_subset = all(ismember(Indq_targ_real,Indq_real));  
         is_equal  = isequal(Indq_targ_real,Indq_real);  
         if ~is_equal && ~is_subset
             error('STOP32 in sub_data_structure_and_check: Indq_targ_real is neither a subset of Indq_real nor equal to Indq_real');
         end
      end
      if nqpos_targ >= 1  
         if ind_exec_solver == 2 && (ind_type_targ == 2 || ind_type_targ == 3)
            error('STOP33 in sub_data_structure_and_check: for ind_exec_solver = 2 and ind_type_targ = 2 or 3, we must have nqpos_targ = 0');  
         end
         if length(Indq_targ_pos) ~= length(unique(Indq_targ_pos))
            error('STOP34 in sub_data_structure_and_check: there are repetitions in Indq_targ_pos');  
         end
         if any(Indq_targ_pos < 1) || any(Indq_targ_pos > n_q)
            error('STOP35 in sub_data_structure_and_check: at least one  integer in Indq_targ_pos is not within the valid range');    
         end
         is_subset = all(ismember(Indq_targ_pos,Indq_pos));  
         is_equal  = isequal(Indq_targ_pos,Indq_pos);  
         if ~is_equal && ~is_subset
             error('STOP36 in sub_data_structure_and_check: Indq_targ_pos is neither a subset of Indq_pos nor equal to Indq_pos');
         end
      end
      if nqreal_targ >= 1 && nqpos_targ >= 1              % Check the coherence
         if isempty(intersect(Indq_targ_real,Indq_targ_pos))
            combined_list = [Indq_targ_real; Indq_targ_pos]; 
            if length(combined_list) <= n_q 
               ind_error = 0;                            % All integers in Indq_targ_real are different from those in Indq_targ_pos, and the 
                                                         % length of union is <= n_q
            else
               ind_error = 1;                            % The length of union of Indq_targ_real and Indq_targ_pos  is not <= n_q 
            end
         else
               ind_error = 2;                            % There are common integers in Indq_targ_real and Indq_targ_pos
         end
         if ind_error == 1
            error('STOP37 in sub_data_structure_and_check: at least one integer in Indq_targ_real is equal to an integer in Indq_targ_pos'); 
         end
         if ind_error == 2
            error('STOP38 in sub_data_structure_and_check: there are common integers in Indq_targ_real and Indq_targ_pos'); 
         end
      end
         
      %--- Checking input data and parameters of  Indw_targ_real(nwreal_real,1) and Indw_targ_pos(nwpos_targ,1)  
      if nwreal_targ >= 1   
         if length(Indw_targ_real) ~= length(unique(Indw_targ_real))
            error('STOP39 in sub_data_structure_and_check: there are repetitions in Indw_targ_real');               
         end
         if any(Indw_targ_real < 1) || any(Indw_targ_real > n_w)
            error('STOP40 in sub_data_structure_and_check: at least one integer in Indw_targ_real is not within the valid range');                
         end
         is_subset = all(ismember(Indw_targ_real,Indw_real));  
         is_equal  = isequal(Indw_targ_real,Indw_real);  
         if ~is_equal && ~is_subset
             error('STOP41 in sub_data_structure_and_check: Indw_targ_real is neither a subset of Indw_real nor equal to Indw_real');
         end
      end
      if nwpos_targ >= 1  
         if ind_exec_solver == 2 && (ind_type_targ == 2 || ind_type_targ == 3)
            error('STOP42 in sub_data_structure_and_check: for ind_exec_solver = 2 and ind_type_targ = 2 or 3, we must have nwpos_targ = 0');  
         end
         if length(Indw_targ_pos) ~= length(unique(Indw_targ_pos))
            error('STOP43 in sub_data_structure_and_check: there are repetitions in Indw_targ_pos');  
         end
         if any(Indw_targ_pos < 1) || any(Indw_targ_pos > n_w)
            error('STOP44 in sub_data_structure_and_check: at least one  integer in Indw_targ_pos is not within the valid range');    
         end
         is_subset = all(ismember(Indw_targ_pos,Indw_pos));  
         is_equal  = isequal(Indw_targ_pos,Indw_pos);  
         if ~is_equal && ~is_subset
             error('STOP45 in sub_data_structure_and_check: Indw_targ_pos is neither a subset of Indw_pos nor equal to Indw_pos');
         end
      end
      if nwreal_targ >= 1 && nwpos_targ >= 1             % Check the coherence
         if isempty(intersect(Indw_targ_real,Indw_targ_pos))
            combined_list = [Indw_targ_real; Indw_targ_pos]; 
            if length(combined_list) <= n_w 
               ind_error = 0;                            % All integers in Indw_targ_real are different from those in Indw_targ_pos, and the 
                                                         % length of union is <= n_w
            else
               ind_error = 1;                            % The length of union of Indw_targ_real and Indw_targ_pos  is not <= n_w 
            end
         else
               ind_error = 2;                            % There are common integers in Indw_targ_real and Indw_targ_pos
         end
         if ind_error == 1
            error('STOP46 in sub_data_structure_and_check: at least one integer in Indw_targ_real is equal to an integer in Indw_targ_pos'); 
         end
         if ind_error == 2
            error('STOP47 in sub_data_structure_and_check: there are common integers in Indw_targ_real and Indw_targ_pos'); 
         end
      end   
   end   

   %--- Construction of Indx_obs(nx_obs,1) : nx_obs component numbers of XX = (QQ,WW) that are observed 
   n_x      = n_q + n_w;                     % dimension of random vector XX = (QQ,WW)
   nx_obs   = nq_obs + nw_obs;               % number of components of XX that are observed
   Indx_obs = [Indq_obs                      % Indq_obs(nq_obs,1) : nq_obs component numbers of QQ that are observed 
               n_q + Indw_obs];              % Indw_obs(nw_obs,1) : nw_obs component numbers of WW that are observed  


   %--- Checking n_d and MatRxx_d(n_x,n_d)
   if n_d <= 0
      error('STOP48 in sub_data_structure_and_check: n_d must be greater than of equal to 1');
   end
   [n1temp,n2temp] = size(MatRxx_d);         % MatRxx_d(n_x,n_d)
   if n1temp ~= n_x || n2temp ~= n_d 
      error('STOP49 in sub_data_structure_and_check: dimension error in matrix MatRxx_d(n_x,n_d)');
   end 
   MatRqq_d = MatRxx_d(1:n_q,:);             % MatRqq_d(n_q,n_d)
   MatRww_d = MatRxx_d(n_q+1:n_q+n_w,:);     % MatRww_d(n_w,n_d)
   if nqpos >= 1
      MatRqq_pos = MatRqq_d(Indq_pos,:);     % MatRqq_pos(nqpos,:), Indq_pos(nqpos,1)
      if any(MatRqq_pos(:) <= 0)
         error('STOP50 in sub_data_structure_and_check: all values in MatRqq_pos(nqpos,n_d) must be strictly positive');
      end
   end
   if nwpos >= 1
      MatRww_pos = MatRww_d(Indw_pos,:);     % MatRww_pos(nwpos,:), Indw_pos(nwpos,1)
      if any(MatRww_pos(:) <= 0)
         error('STOP51 in sub_data_structure_and_check: all values in MatRww_pos(nwpos,n_d) must be strictly positive');
      end
   end

   %--- construction of :
   %                     Indx_real(nbreal,1) that contains the nbreal component numbers of XX, which are real (positive, negative, or zero) 
   %                                         with 0 <= nbreal <=  n_x and for which a "standard scaling" will be used
   %                     Indx_pos(nbpos,1)   that contains the nbpos component numbers of XX, which are strictly positive a "specific scaling"
   %                                         with  0 <= nbpos <=  n_x  and for which the scaling is {log + "standard scaling"}
   nbreal    = nqreal + nwreal;
   nbpos     = nqpos  + nwpos;
   Indx_real = [Indq_real
                n_q + Indw_real]; % nbreal component numbers of XX, which are real
   Indx_pos  = [Indq_pos
                n_q + Indw_pos];  % nbpos component numbers of XX, which are strictly positive   

   %--- construction of :
   %      Indx_targ_real(nbreal_targ,1) that contains the nbreal_targ component numbers of XX, which are real 
   %                                    with 0 <= nbreal_targ <=  n_x and for which a "standard scaling" will be used
   %      Indx_targ_pos(nbpos_targ,1)   that contains the nbpos_targ component numbers of XX, which are strictly positive a "specific scaling"
   %                                    with  0 <= nbpos_targ <=  n_x  and for which the scaling is {log + "standard scaling"}
   %                                    nx_targ = nbreal_targ + nbpos_targ <= n_x
   if ind_exec_solver == 1
      Indx_targ_real = [];      
      Indx_targ_pos  = [];
      nx_targ        = 0;
      Indx_targ      = [];
   end
   if ind_exec_solver == 2
      if ind_type_targ == 1
         Indx_targ_real = [Indq_targ_real           % Indx_targ_real(nbreal_targ,1)
                           n_q + Indw_targ_real];   % nbreal_targ component numbers of XX, which are real
         Indx_targ_pos  = [Indq_targ_pos            % Indx_targ_pos(nbpos_targ,1)
                           n_q + Indw_targ_pos];    % nbpos_targ component numbers of XX, which are strictly positive
         nx_targ        = nbreal_targ + nbpos_targ; % dimension of random vector XX_targ = (QQ_targ,WW_targ)
         Indx_targ      = [Indx_targ_real           % nx_targ component numbers of XX_targ for which a target is given 
                           Indx_targ_pos]; 
      end
      if ind_type_targ == 2 || ind_type_targ == 3
         Indx_targ_real = [Indq_targ_real           % Indx_targ_real(nbreal_targ,1)
                           n_q + Indw_targ_real];   % nbreal_targ component numbers of XX, which are real
         Indx_targ_pos  = [];                       % Indx_targ_pos(nbpos_targ,1)
         nx_targ        = nbreal_targ;              % dimension of random vector XX_targ = (QQ_targ,WW_targ)
         Indx_targ      = Indx_targ_real;           % nx_targ component numbers of XX_targ for which a target is given 
      end
   end
   
   %--- Checking data related to target dataset  
   if ind_exec_solver == 2
      if ind_type_targ == 1 
         if N_r <= 0
            error('STOP52 in sub_data_structure_and_check: for an inverse problem, if ind_type_targ = 1, then N_r must be greater than or equal to 1');
         end
         if nbreal_targ >= 1  
            [n1temp,n2temp] = size(MatRxx_targ_real);         % MatRxx_targ_real(nbreal_targ,N_r)
            if n1temp ~= nbreal_targ || n2temp ~= N_r 
               error('STOP53 in sub_data_structure_and_check: dimension error in matrix MatRxx_targ_real(nbreal_targ,N_r)');
            end
         end 
         if nbpos_targ >= 1  
            [n1temp,n2temp] = size(MatRxx_targ_pos);          % MatRxx_targ_pos(nbpos_targ,N_r)
            if n1temp ~= nbpos_targ || n2temp ~= N_r 
               error('STOP54 in sub_data_structure_and_check: dimension error in matrix MatRxx_targ_pos(nbpos_targ,N_r)');
            end
            if any(MatRxx_targ_pos(:) <= 0)
               error('STOP55 in sub_data_structure_and_check: all values in MatRxx_targ_pos(nbpos_targ,N_r) must be strictly positive');
            end
         end 
      end
      if ind_type_targ == 2 ||  ind_type_targ == 3
         [n1temp,n2temp] = size(Rmeanxx_targ);                % Rmeanxx_targ(nx_targ,1)
            if n1temp ~= nx_targ || n2temp ~= 1 
               error('STOP56 in sub_data_structure_and_check: dimension error in matrix Rmeanxx_targ(nx_targ,1)');
            end
      end
      if ind_type_targ == 3 
         [n1temp,n2temp] = size(MatRcovxx_targ);              % MatRcovxx_targ(nx_targ,nx_targ)
         if n1temp ~= nx_targ || n2temp ~= nx_targ
            error('STOP57 in sub_data_structure_and_check: dimension error in matrix MatRcovxx_targ(nx_targ,nx_targ)');
         end  
   
         % Check if the matrix is symmetric and positive
         if ~issymmetric(MatRcovxx_targ)
            error('STOP58 in sub_data_structure_and_check: matrix MatRcovxx_targ(nx_targ,nx_targ) must be symmetric');
         end
         Reigenvalues   = eig(MatRcovxx_targ);
         max_eigenvalue = max(Reigenvalues);
         if any(Reigenvalues <= 1e-12*max_eigenvalue)
            error('STOP59 in sub_data_structure_and_check: matrix MatRcovxx_targ(nx_targ,nx_targ) must be positive definite'); 
         end
      end
   end
  
   %--- Display 
   if ind_display_screen == 1
      if ind_exec_solver == 1
         disp(['ind_exec_solver = ',num2str(ind_exec_solver),', Direct Solver used.']);     
      end
      if ind_exec_solver == 2
         disp(['ind_exec_solver = ',num2str(ind_exec_solver),', Inverse Solver used.']);     
      end
      disp(' ');
      disp(['n_q    = ',num2str(n_q)]);     
      disp(['n_w    = ',num2str(n_w)]); 
      disp(['n_x    = ',num2str(n_x)]); 
      disp(' ');
      disp(['nqreal = ',num2str(nqreal)]); 
      disp(['nwreal = ',num2str(nwreal)]);  
      disp(['nbreal = ',num2str(nbreal)]); 
      disp(' ');
      disp(['nqpos  = ',num2str(nqpos)]);  
      disp(['nwpos  = ',num2str(nwpos)]); 
      disp(['nbpos  = ',num2str(nbpos)]); 
      disp(' ');
      disp(['nq_obs    = ',num2str(nq_obs)]);     
      disp(['nw_obs    = ',num2str(nw_obs)]); 
      disp(['nx_obs    = ',num2str(nx_obs)]); 
      disp(' ');        
      if ind_exec_solver  == 2
         disp(' ');
         disp(['ind_type_targ     = ',num2str(ind_type_targ)]); 
         disp(' ');
         disp(['     nqreal_targ  = ',num2str(nqreal_targ)]); 
         disp(['     nwreal_targ  = ',num2str(nwreal_targ)]);  
         disp(['     nbreal_targ  = ',num2str(nbreal_targ)]);  
         disp(' ');
         disp(['     nqpos_targ   = ',num2str(nqpos_targ)]);  
         disp(['     nwpos_targ   = ',num2str(nwpos_targ)]);  
         disp(['     nbpos_targ   = ',num2str(nbpos_targ)]); 
         disp(' ');
         disp(['     nx_targ      = ',num2str(nx_targ)]);  
         if ind_type_targ == 1
            disp(' ');
            disp(['     N_r          = ',num2str(N_r)]);  
         end
      end
   end

   %--- Print          
   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      if ind_exec_solver == 1
         fprintf(fidlisting,'ind_exec_solver = %1i, Direct Solver used \n ',ind_exec_solver);    
      end
      if ind_exec_solver == 2
         fprintf(fidlisting,'ind_exec_solver = %1i, Inverse Solver used \n ',ind_exec_solver);    
      end        
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'n_q     = %9i \n ',n_q); 
      fprintf(fidlisting,'n_w     = %9i \n ',n_w);  
      fprintf(fidlisting,'n_x     = %9i \n ',n_x); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'nqreal  = %9i \n ',nqreal); 
      fprintf(fidlisting,'nwreal  = %9i \n ',nwreal); 
      fprintf(fidlisting,'nbreal  = %9i \n ',nbreal);  
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'nqpos   = %9i \n ',nqpos);  
      fprintf(fidlisting,'nwpos   = %9i \n ',nwpos); 
      fprintf(fidlisting,'nbpos   = %9i \n ',nbpos);  
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'nq_obs  = %9i \n ',nq_obs);  
      fprintf(fidlisting,'nw_obs  = %9i \n ',nw_obs);  
      fprintf(fidlisting,'nx_obs  = %9i \n ',nx_obs);  
      fprintf(fidlisting,'      \n ');      
      if ind_exec_solver  == 2
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'ind_type_targ     = %9i \n ',ind_type_targ);  
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'     nqreal_targ      = %9i \n ',nqreal_targ);    
         fprintf(fidlisting,'     nwreal_targ      = %9i \n ',nwreal_targ);  
         fprintf(fidlisting,'     nbreal_targ      = %9i \n ',nbreal_targ);  
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'     nqpos_targ       = %9i \n ',nqpos_targ);    
         fprintf(fidlisting,'     nwpos_targ       = %9i \n ',nwpos_targ);  
         fprintf(fidlisting,'     nbpos_targ       = %9i \n ',nbpos_targ);  
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'     nx_targ          = %9i \n ',nx_targ);  
         if ind_type_targ == 1
            fprintf(fidlisting,'      \n '); 
            fprintf(fidlisting,'     N_r              = %9i \n ',N_r);  
         end   
         fprintf(fidlisting,'      \n '); 
      end
      fclose(fidlisting); 
   end

   ElapsedTimeDataCheck = toc(TimeStartDataCheck);   

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');                                                         
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ----- Elapsed time for Task1_DataStructureCheck \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' Elapsed Time   =  %10.2f\n',ElapsedTimeDataCheck);   
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task1_DataStructureCheck')
   end    
   
   return
end
      