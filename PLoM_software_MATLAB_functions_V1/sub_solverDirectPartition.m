function  [n_ar,MatReta_ar,SAVERANDendDirectPartition,d2mopt_ar,divKL,iHd,iHar,entropy_Hd,entropy_Har] = ...
             sub_solverDirectPartition(nu,n_d,nbMC,MatReta_d,ind_generator,icorrectif,f0_ref,ind_f0,coeffDeltar,M0transient, ...
                              epsilonDIFFmin,step_epsilonDIFF,iterlimit_epsilonDIFF,comp_ref, ... 
                              ind_constraints,ind_coupling,iter_limit,epsc,minVarH,maxVarH,alpha_relax1,iter_relax2, ...
                              alpha_relax2,ngroup,Igroup,MatIgroup,SAVERANDstartDirectPartition,ind_display_screen,ind_print, ...
                              ind_plot,ind_parallel,MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D, ...
                              ind_Kullback,ind_Entropy,ind_MutualInfo)
                               

   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 01 July 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverDirectPartitionPartition
   %  Subject      : solver PLoM for direct predictions with the partition, and with or without constraints of normalization for H_ar
   %                 computation of n_ar learned realizations MatReta_ar(nu,n_ar) of H_ar using partition
   %                 H    = (H_1,...,H_r,...,H_nu)    in ngroup subsets (groups) H^1,...,H^j,...,H^ngroup
   %                 H    = (Y^1,...,Y^j,...,Y^ngroup) 
   %                 Y^j  = = (Y^j_1,...,Y^j_mj) = (H_rj1,...,Hrjmj)   with j = 1,...,ngroup and with n1 + ... + nngroup = nu
   %
   %  Publications: [1] C. Soize, Optimal partition in terms of independent random vectors of any non-Gaussian vector defined by a set of
   %                       realizations,SIAM-ASA Journal on Uncertainty Quantification,doi: 10.1137/16M1062223, 5(1), 176-211 (2017).
   %                [2] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
   %                       Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).
   %                [3] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
   %                       American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
   %                [4] C. Soize, R. Ghanem, Physics-constrained non-Gaussian probabilistic learning on manifolds, 
   %                       International Journal for Numerical Methods in Engineering, doi: 10.1002/nme.6202, 121 (1), 110-145 (2020).   
   %                [5] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
   %                       Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).  
   %                       doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024).
   %                [6] C. Soize, R. Ghanem, Transient anisotropic kernel for probabilistic learning on manifolds, 
   %                       Computer Methods in Applied Mechanics and Engineering, pp.1-44 (2024).
   %
   %--- INPUTS 
   %
   %    nu                  : dimension of random vector H = (H_1, ... H_nu)
   %    n_d                 : number of points in the training set for H
   %    nbMC                : number of realizations of (nu,n_d)-valued random matrix [H_ar]    
   %    MatReta_d(nu,n_d)   : n_d realizations of H   
   %    ind_generator:      : 0 generator without using ISDE-projection basis = standard MCMC generator based on Hamiltonian dissipative
   %                        : 1 generator using the ISDE-projection basis 
   %    icorrectif          = 0: usual Silveman-bandwidth formulation for which the normalization conditions are not exactly satisfied
   %                        = 1: modified Silverman bandwidth implying that, for any value of nu, the normalization conditions are verified  
   %    f0_ref              : reference value (recommended value f0_ref = 4)
   %    ind_f0              : indicator for generating f0 (recommended value ind_f0 = 0): 
   %                          if ind_f0 = 0, then f0 = f0_ref, and if ind_f0 = 1, then f0 = f0_ref/sh    
   %    coeffDeltar         : coefficient > 0 (usual value is 20) for calculating Deltar
   %    M0transient         : the end-integration value, M0transient (for instance, 30), at which the stationary response of the ISDE is 
   %                          reached, is given by the user. The corresponding final time at which the realization is extrated from 
   %                          solverDirect_Verlet is M0transient*Deltar 
   %
   %--- parameters for computing epsolonDIFF for each group j
   %    epsilonDIFFmin        : epsilonDIFF is searched in interval [epsilonDIFFmin , +infty[                                    
   %    step_epsilonDIFF      : step for searching the optimal value epsilonDIFF starting from epsilonDIFFmin
   %    iterlimit_epsilonDIFF : maximum number of the iteration algorithm for computing epsilonDIFF                              
   %    comp_ref              : value in  [ 0.1 , 0.5 [  used for stopping the iteration algorithm.
   %                            if comp =  Rlambda(nbmDMAP+1)/Rlambda(nbmDMAP) <= comp_ref, then algorithm is stopped
   %                            The standard value for comp_ref is 0.2 
   %
   %--- parameters for the constraints (ind_constraints >= 1) related to the convergence of the Lagrange-multipliers iteration algorithm 
   %    these constraints are applied to each group of the partition  Y^j  = (Y^j_1,...Y^j_mj) with dimension mj
   %    ind_constraints     = 0 : no constraints concerning E{Y^j] = 0 and E{Y^j (Y^j)'} = [I_mj]
   %                        = 1 : constraints E{(Y^j_k)^2} = 1 for k =1,...,mj   
   %                        = 2 : constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
   %                        = 3 : constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj]  
   %    ind_coupling        = 0 : for ind_constraints = 2 or 3, no coupling in  matrix MatRGammaS_iter (HIGHLY RECOMMENDED)
   %                        = 1 : for ind_constraints = 2 or 3, coupling all the extra-diagonal blocs in matrix MatRGammaS_iter are kept
   %    iter_limit          : maximum number of iterations used by the iterative algorithm to compute the Lagrange multipliers. 
   %    epsc                =   : relative tolerance (for instance 1e-3) for the iteration-algorithm convergence 
   %    minVarH                 : minimum imposed on E{(Y^j_k)^2} with respect to 1 (for instance 0.999) 
   %    maxVarH                 : maximum imposed on E{(Y^j_k)^2} with respect to 1 (for instance 1.001) 
   %                          NOTE 5: on the convergence criteria for the iteration algorithm computing the Lagrange multipliers:
   %                               Criterion 1: if iter > 10 and if Rerr(iter)-Rerr(iter-1) > 0, there is a local minimum, obtained 
   %                                            for iter - 1. Convergence is then assumed to be reached and then, exit from the loop on iter  
   %                               Criterion 2: if {minVarH_iter >= minVarH and maxVarH_iter <= maxVarH} or Rerr(iter) <= epsc, the 
   %                                            variance of each component is greater than or equal to minVarH and less than or equal 
   %                                            to maxVarH, or the relative error of the constraint satisfaction is less than or equal 
   %                                            to the tolerance. The convergence is reached, and then exit from the loop on iter.
   %                               Criterion 3: if iter > min(20,iter_limit) and Rtol(iter) < epsc,  the error is stationary and thus
   %                                            the convergence is assumed to be reached and then, exit from the loop on iter  
   %                          --- Relaxation function  iter --> alpha_relax(iter) controlling convergence of the iterative algorithm 
   %                              is described by 3 parameters: alpha_relax1, iter_relax2, and alpha_relax2
   %    alpha_relax1        : value of alpha_relax for iter = 1  (for instance 0.001)
   %    iter_relax2         : value of iter (for instance, 20) such that  alpha_relax2 = alpha_relax(iter_relax2) 
   %                          if iter_relax2 = 1, then alpha_relax (iter) = alpha_relax2 for all iter >=1   
   %    alpha_relax2        : value of alpha_relax (for instance, 0.05) such that alpha_relax(iter >= iter_relax2) = apha_relax2
   %                          NOTE 1: If iter_relax2 = 1 , then Ralpha_relax(iter) = alpha_relax2 for all iter >=1
   %                          NOTE 2: If iter_relax2 >= 2, then  
   %                                  for iter >= 1 and for iter < iter_relax2, we have:
   %                                      alpha_relax(iter) = alpha_relax1 + (alpha_relax2 - alpha_relax1)*(iter-1)/(iter_relax2-1)
   %                                  for iter >= iter_relax2, we have:
   %                                      alpha_relax(iter) = alpha_relax2
   %                          NOTE 3: for decreasing the error err(iter), increase the value of iter_relax2
   %                          NOTE 4: if iteration algorithm dos not converge, decrease alpha_relax2 and/or increase iter_relax2  
   %
   %--- information concerning the partition 
   %    ngroup                 : number of constructed independent groups  
   %    Igroup(ngroup,1)       : vector Igroup(ngroup,1), mj = Igroup(j),  mj is the dimension of Y^j = (Y^j_1,...,Y^j_mj) = (H_jr1,... ,H_jrmj)  
   %    MatIgroup(ngroup,mmax) : MatIgroup1(j,r) = rj, in which rj is the component of H in group j such that Y^j_r = H_jrj 
   %                             with mmax = max_j mj for j = 1, ... , ngroup
   %
   %--- parameters and variables controling execution
   %    SAVERANDstartDirectPartition : state of the random generator at the end of the PCA step
   %    ind_display_screen           : = 0 no display,              = 1 display
   %    ind_print                    : = 0 no print,                = 1 print
   %    ind_plot                     : = 0 no plot,                 = 1 plot
   %    ind_parallel                 : = 0 no parallel computation, = 1 parallel computation
   %
   %--- data for the plots
   %    MatRplotHsamples(1,nbplotHsamples) : contains the components numbers of H_ar for which the plot of the realizations are made
   %                                         example 1: MatRplotHsamples = [3 7 8]; plot for the nbplotHsamples = 3 components 3,7, and 8
   %                                         example 2: MatRplotHsamples = [];      no plot, nbplotHsamples = 0
   %    MatRplotHClouds(nbplotHClouds,3)   : contains the 3 components numbers of H_ar for which the plot of the clouds are made
   %                                         example 1: MatRplotHClouds = [2 4 6   plot for the 3 components 2,4, and 6 
   %                                                                       3 4 8]; plot for the 3 components 3,4, and 8
   %                                         example 2: MatRplotHClouds = [];      no plot, nbplotHClouds = 0
   %    MatRplotHpdf(1,nbplotHpdf)         : contains the components numbers of H_d and H_ar for which the plot of the pdfs are made
   %                                         example 1: MatRplotHpdf = [3 5 7 9]; plot for the nbplotHpdf = 4 components 3,5,7, and 9
   %                                         example 2: MatRplotHpdf = [];        no plot, nbplotHpdf = 0
   %
   %    MatRplotHpdf2D(nbplotHpdf2D,2)     : contains the 2 components numbers of H_d and H_ar for which the plot of the joint pdfs are made
   %                                         example 1: MatRplotHpdf2D = [2 4   plot for the 3 components 2 and 4 
   %                                                                      3 4]; plot for the 3 components 3 and 4
   %                                         example 2: MatRplotHpdf2D = [];    no plot, nbplotHpdf2D = 0
   %    ind_Kullback                       = 0 : no computation of the Kullback-Leibler divergence of H_ar with respect to H_d
   %                                       = 1 :    computation of the Kullback-Leibler divergence of H_ar with respect to H_d
   %    ind_Entropy                        = 0 : no computation of the Entropy of Hd and Har 
   %                                       = 1 :    computation of the Entropy of Hd and Har 
   %    ind_MutualInfo                     = 0 : no computation of the Mutual Information iHd and iHar for Hd and Har
   %                                       = 1 :    computation of the Mutual Information iHd and iHar for Hd and Har
   %  
   %--- OUPUTS  
   %
   %      n_ar                        : number of realizations of H_ar such that n_ar  = nbMC x n_d
   %      MatReta_ar(nu,n_ar)         : n_ar realizations of H_ar 
   %      SAVERANDendDirectPartition  : state of the random generator at the end of sub_solverDirectPartition
   %      d2mopt_ar                   : concentration of the probability measure of H_ar with respect to H_d in the means-square sense
   %      divKL                       : Kullback-Leibler divergence of H_ar with respect to H_d
   %      iHd                         : Mutual Information iHd for Hd 
   %      iHar                        : Mutual Information iHar for Har
   %      entropy_Hd                  : Entropy of Hd
   %      entropy_Har                 : Entropy of Har
   %
   %--- INTERNAL PARAMETERS
   %      For each group j:
   %      sj       : usual Silver bandwidth for the GKDE estimate (with the n_d points of the training dataset) 
   %                 of the pdf p_{Y^j} of Y^j, having to satisfy the normalization condition E{Y^j] = 0 and E{Y^j (Y^j)'} = [I_mj] 
   %      shj      : modified Silver bandwidth for wich the normalization conditions are satisfied for any value of mj >= 1 
   %      shssj    : = 1 if icorrectif  = 0, and = shj/sj if icorrectif = 1
   %      f0j      : damping parameter in the ISDE, which controls the speed to reach the stationary response of the ISDE
   %      Deltarj  : Stormer-Verlet integration-step of the ISDE        
   %      M0estimj : estimate of M0transient provided as a reference to the user
   %
   %      For all groups (independent of j)
   %      Ralpha_relax(iter_limit,1): relaxation function for the iteration algorithm that computes the LaGrange multipliers
   %      n_ar    : number of learned realizations equal to nbMC*n_d:   %
   %      nbplotHsamples : number >= 0 of the components numbers of H_ar for which the plot of the realizations are made   
   %      nbplotHClouds  : number >= 0 of the 3 components numbers of H_ar for which the plot of the clouds are made
   %      nbplotHpdf     : number >= 0 of the components numbers of H_d and H_ar for which the plot of the pdfs are made   
   %      nbplotHpdf2D   : number >= 0 of the 2 components numbers of H_d and H_ar for which the plot of the joint pdfs are made
   
   if ind_display_screen == 1                              
      disp('--- beginning Task7_SolverDirectPartition')
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task7_SolverDirectPartition \n ');
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end

   TimeStartSolverDirectPartition = tic; 
   n_ar                  = nbMC*n_d;

   nbplotHsamples = size(MatRplotHsamples,2);   % MatRplotHsamples(1,nbplotHsamples)
   nbplotHClouds  = size(MatRplotHClouds,1);    % MatRplotHClouds(nbplotHClouds,3)
   nbplotHpdf     = size(MatRplotHpdf,2);       % MatRplotHpdf(1,nbplotHpdf)
   nbplotHpdf2D   = size(MatRplotHpdf2D,1);     % MatRplotHpdf2D(nbplotHpdf2D,2)

   %--- initializing the random generator at the value of the end of the PCA step 
   rng(SAVERANDstartDirectPartition);   

   %----------------------------------------------------------------------------------------------------------------------------------
   %                                    Check data, parameters, and initialization
   %---------------------------------------------------------------------------------------------------------------------------------- 
   
   if nu > n_d || nu < 1 || n_d < 1
      error('STOP1 in sub_solverDirectPartition: nu > n_d or nu < 1 or n_d < 1')
   end   
   [nutemp,ndtemp] = size(MatReta_d);              % MatReta_d(nu,n_d) 
   if nutemp ~= nu || ndtemp ~= n_d
      error('STOP2 in sub_solverDirectPartition: the dimensions of MatReta_d are not consistent with nu and n_d')
   end  
   if ind_generator ~= 0 && ind_generator ~= 1  
       error('STOP3 in sub_solverDirectPartition: ind_generator must be equal to 0 or equal to 1')
   end  
   if icorrectif ~= 0 && icorrectif ~= 1  
       error('STOP4 in sub_solverDirectPartition: icorrectif must be equal to 0 or equal to 1')
   end 

   %--- Checking parameters controlling the time integration scheme of the ISDE
   if f0_ref <= 0
      error('STOP5 in sub_solverDirectPartition: f0_ref must be strictly positif') 
   end
   if ind_f0 ~= 0 && ind_f0 ~= 1       
      error('STOP6 in sub_solverDirectPartition: ind_f0 must be equal to 0 or equal to 1')
   end
   if coeffDeltar < 1
      error('STOP7 in sub_solverDirectPartition: coeffDeltar must be greater than or equal to 1')
   end
   if M0transient < 1
      error('STOP8 in sub_solverInverse: M0transient must be greater than or equal to 1')
   end
   
   %--- Parameters controlling the computation of epsilonDIFFj for each group j
   if ind_generator == 0
      comp_ref               = 0;
      epsilonDIFFmin         = 0;
      step_epsilonDIFF       = 0;
      iterlimit_epsilonDIFF  = 0;
   end
   if ind_generator == 1
      if 0.1 > comp_ref || comp_ref > 0.5  %  comp_ref given by the user in [ 0.1 , 0.5 [
         error('STOP9 in sub_solverDirectPartition: for ind_basis_type = 2, comp_ref must be given by the user between 0.1 and 0.5')
      end
      if epsilonDIFFmin  <= 0
         error('STOP10 in sub_solverDirectPartition: for ind_basis_type = 2, epsilonDIFFmin must be given by the user as a strictly posive real number')
      end
      if step_epsilonDIFF  <= 0
         error('STOP11 in sub_solverDirectPartition: for ind_basis_type = 2, step_epsilonDIFF must be given by the user as a strictly posive real number')
      end
      if iterlimit_epsilonDIFF < 1
         error('STOP12 in sub_solverDirectPartition: for ind_basis_type = 2, iterlimit_epsilonDIFF must be given by the user as an integer larger than or equal to 1')
      end
   end

   %--- Checking data controlling constraints
   if ind_constraints ~= 0 && ind_constraints ~= 1 && ind_constraints ~= 2  && ind_constraints ~= 3 
      error('STOP13 in sub_solverDirectPartition: ind_constraints must be equal to 0, 1, 2, or 3');
   end
   if ind_constraints == 0
      iter_limit   = 0;
      iter_relax2  = 0;
      alpha_relax1 = 0;
      alpha_relax2 = 0;
   end
   if ind_constraints >= 1
      if ind_coupling ~= 0 && ind_coupling ~= 1       
         error('STOP14 in sub_solverDirectPartition: ind_coupling must be equal to 0 or equal to 1');
      end
      if iter_limit < 1       
         error('STOP15 in sub_solverDirectPartition: iter_limit must be greater than or equal to 1');
      end
      if epsc < 0 || epsc >= 1       
         error('STOP16 in sub_solverDirectPartition: epsc < 0 or epsc >= 1 ');
      end
      if minVarH <= 0 || minVarH >= 1       
         error('STOP17 in sub_solverDirectPartition: minVarH <= 0 or minVarH >= 1 ');
      end
      if maxVarH <= minVarH || maxVarH <= 1       
         error('STOP18 in sub_solverDirectPartition: maxVarH <= minVarH or maxVarH <= 1 ');
      end
      if alpha_relax1 < 0 || alpha_relax1 > 1
         error('STOP19 in sub_solverDirectPartition: value of alpha_relax1 out the range [0,1]');
      end
      if alpha_relax2 < 0 || alpha_relax2 > 1
         error('STOP20 in sub_solverDirectPartition: value of alpha_relax2 out the range [0,1]');
      end
      if iter_relax2 >= 2 && iter_relax2 <= iter_limit
         if alpha_relax1 >  alpha_relax2
             error('STOP21 in sub_solverDirectPartition: alpha_relax1 must be less than or equal to alpha_relax2');
         end
      end
      if iter_relax2 > iter_limit
          error('STOP22 in sub_solverDirectPartition: iter_relax2 must be less than or equal to iter_limit');
      end  
   end
   %--- Checking data related to the partition
   if ngroup <= 0
       error('STOP23 in sub_solverDirectPartition: ngroup must be greater than or equal to 1');
   end
   [n1temp,n2temp] = size(Igroup);         % Igroup(ngroup,1) 
   if n1temp ~= ngroup || n2temp ~= 1
       error('STOP24 in sub_solverDirectPartition: the dimensions of Igroup(ngroup,1) are not correct');
   end
   mmax = size(MatIgroup,2);              % MatIgroup(ngroup,mmax)  
   if mmax <= 0 || mmax > nu
      error('STOP25 in sub_solverDirectPartition: mmax must be in the range of integers [1,nu]');
   end
   n1temp = size(MatIgroup,1);              % MatIgroup(ngroup,mmax) 
   if n1temp ~= ngroup
      error('STOP26 in sub_solverDirectPartition: the first dimension of MatIgroup must be equal to ngroup');
   end
   nutemp = 0;
   for j = 1:ngroup
       mj = Igroup(j);
       nutemp = nutemp + mj;
   end
   if nutemp ~= nu
      error('STOP27 in sub_solverDirectPartition: data in Igroup is not consistent with dimension nu');
   end

   %--- Checking parameters that control options
   if ind_display_screen ~= 0 && ind_display_screen ~= 1       
         error('STOP28 in sub_solverDirectPartition: ind_display_screen must be equal to 0 or equal to 1');
   end
   if ind_print ~= 0 && ind_print ~= 1       
         error('STOP29 in sub_solverDirectPartition: ind_print must be equal to 0 or equal to 1');
   end
   if ind_plot ~= 0 && ind_plot ~= 1       
         error('STOP30 in sub_solverDirectPartition: ind_plot must be equal to 0 or equal to 1');
   end
   if ind_parallel ~= 0 && ind_parallel ~= 1       
         error('STOP31 in sub_solverDirectPartition: ind_parallel must be equal to 0 or equal to 1')
   end

   %--- Checking parameters and data controling the plots
   if nbplotHsamples >= 1                          % MatRplotHsamples(1,nbplotHsamples)
      n1temp = size(MatRplotHsamples,1);
      if n1temp ~= 1 
         error('STOP32 in sub_solverDirectPartition: the first dimension of MatRplotHsamples must be equal to 1') 
      end
      if any(MatRplotHsamples(1,:) < 1) || any(MatRplotHsamples(1,:) > nu)   % at least one integer is not within the valid range
         error('STOP33 in sub_solverDirectPartition: at least one integer is not within the valid range for MatRplotHsamples') 
      end
   end
   if nbplotHClouds >= 1                           % MatRplotHClouds(nbplotHClouds,3)
      n2temp = size(MatRplotHClouds,2);
      if n2temp ~= 3
         error('STOP34 in sub_solverDirectPartition: the second dimension of MatRplotHClouds must be equal to 3') 
      end
      if any(MatRplotHClouds(:) < 1) || any(MatRplotHClouds(:) > nu)   % At least one integer is not within the valid range
         error('STOP35 in sub_solverDirectPartition: at least one integer is not within the valid range for MatRplotHClouds')         
      end
   end
   if nbplotHpdf >= 1                               % MatRplotHpdf(1,nbplotHpdf)
      n1temp = size(MatRplotHpdf,1);
      if n1temp ~= 1 
          error('STOP36 in sub_solverDirectPartition: the first dimension of MatRplotHpdf must be equal to 1') 
      end
      if any(MatRplotHpdf(1,:) < 1) || any(MatRplotHpdf(1,:) > nu) % at least one integer  is not within the valid range
         error('STOP37 in sub_solverDirectPartition: at least one integer is not within the valid range for MatRplotHpdf')            
      end
   end
   if nbplotHpdf2D >= 1                             % MatRplotHpdf2D(nbplotHpdf2D,2)
      n2temp = size(MatRplotHpdf2D,2);
      if n2temp ~= 2
         error('STOP38 in sub_solverDirectPartition: the second dimension of MatRplotHpdf2D must be equal to 2') 
      end
      if any(MatRplotHpdf2D(:) < 1) || any(MatRplotHpdf2D(:) > nu)  % at least one integer is not within the valid range
         error('STOP39 in sub_solverDirectPartition: at least one integer is not within the valid range for MatRplotHpdf2D')       
      end
   end
      
   %--- Computing and loading Ralpha
   [Ralpha_relax] = sub_solverDirectPartition_parameters_Ralpha(ind_constraints,iter_limit,alpha_relax1,iter_relax2,alpha_relax2); 
                         
   %--- Print the data inputs that are used for all the  groups  
   if ind_print == 1
      fidlisting=fopen('listing.txt','a+'); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n ');                     
      fprintf(fidlisting,' ---  Parameters for the learning \n ');   
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n ');  
      fprintf(fidlisting,' nu            = %7i \n ',nu);
      fprintf(fidlisting,' n_d           = %7i \n ',n_d);
      fprintf(fidlisting,' nbMC          = %7i \n ',nbMC); 
      fprintf(fidlisting,' n_ar          = %7i \n ',n_ar); 
      fprintf(fidlisting,'      \n ');  
      fprintf(fidlisting,' ind_generator = %1i \n ',ind_generator); 
      fprintf(fidlisting,' icorrectif    = %1i \n ',icorrectif); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' f0_ref        = %8.4f \n ',f0_ref); 
      fprintf(fidlisting,' ind_f0        = %7i   \n ',ind_f0);
      fprintf(fidlisting,' coeffDeltar   = %4i   \n ',coeffDeltar); 
      fprintf(fidlisting,' M0transient   = %7i   \n ',M0transient); 
      fprintf(fidlisting,'      \n ');        
      fprintf(fidlisting,'      \n ');     
      fprintf(fidlisting,' epsilonDIFFmin        = %14.7e \n ',epsilonDIFFmin);  
      fprintf(fidlisting,' step_epsilonDIFF      = %14.7e \n ',step_epsilonDIFF);  
      fprintf(fidlisting,' iterlimit_epsilonDIFF = %7i    \n ',iterlimit_epsilonDIFF);                            
      fprintf(fidlisting,' comp_ref              = %8.4f  \n ',comp_ref); 
      fprintf(fidlisting,'      \n ');        
      fprintf(fidlisting,'      \n ');  
      fprintf(fidlisting,' ind_constraints  = %1i \n ',ind_constraints);
      if ind_constraints >= 1
         fprintf(fidlisting,'    ind_coupling  = %1i    \n ',ind_coupling);
         fprintf(fidlisting,'    iter_limit    = %7i    \n ',iter_limit); 
         fprintf(fidlisting,'    epsc          = %14.7e \n ',epsc); 
         fprintf(fidlisting,'    minVarH       = %8.6f  \n ',minVarH); 
         fprintf(fidlisting,'    maxVarH       = %8.6f  \n ',maxVarH); 
         fprintf(fidlisting,'    alpha_relax1  = %8.4f  \n ',alpha_relax1); 
         fprintf(fidlisting,'    iter_relax2   = %7i    \n ',iter_relax2); 
         fprintf(fidlisting,'    alpha_relax2  = %8.4f  \n ',alpha_relax2); 
      end
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ind_display_screen = %1i \n ',ind_display_screen); 
      fprintf(fidlisting,' ind_print          = %1i \n ',ind_print); 
      fprintf(fidlisting,' ind_plot           = %1i \n ',ind_plot); 
      fprintf(fidlisting,' ind_parallel       = %1i \n ',ind_parallel); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ind_Kullback       = %1i \n ',ind_Kullback); 
      fprintf(fidlisting,' ind_Entropy        = %1i \n ',ind_Entropy); 
      fprintf(fidlisting,' ind_MutualInfo     = %1i \n ',ind_MutualInfo);
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end

   %-----------------------------------------------------------------------------------------------------------------------------------------                           
   %                                Generating NnbMC realizations on the independent groups
   %-----------------------------------------------------------------------------------------------------------------------------------------  
   
   %--- optimal partition in ngroup independent random vectors Y^1,...,Y^ngroup of random vector H of dimension nu
   %    ngroup = number of groups Y^1,...,Y^ngroup
   %    Igroup = vector (ngroup,1) such that Igroup(j): number mj of the components of  Y^j = (H_jr1,... ,H_jrmj)
   %    MatIgroup = matrix(ngroup,mmax) such that MatIgroup1(j,r) = rj : indice rj de H dans le groupe j tel que Y^j_r = H_jrj

   %--- Construction of cellMatReta_d{j} = MatReta_dj(mj,n_d)
   cellMatReta_d        = cell(ngroup,1);                   %    cellMatReta_d{j} =  MatReta_dj(mj,n_d),  cellMatReta_d{ngroup}                          
   for j = 1:ngroup                                         
       mj               = Igroup(j);                        %    length mj of vector   Y^j = (H_jr1,... ,H_jrmj) of group j
       MatReta_dj       = MatReta_d(MatIgroup(j,1:mj),:);   %    MatReta_dj(mj,n_d): realizations of Y^j of length mj = Igroup(j)
       cellMatReta_d{j} = MatReta_dj;          
   end  
   clear mj MatReta_dj
   
   %--- Generation of random Wienner germs for the Ito equation for each group
   cellGauss               = cell(ngroup,1);
   cellWiennerM0transientG = cell(ngroup,1); 
   
   for j = 1:ngroup                                         
       mj                         = Igroup(j);                        %  length mj of vector   Y^j = (H_jr1,... ,H_jrmj) of group j  
       cellGauss{j}               = randn(mj,n_d,nbMC);               %  ArrayGaussj(mj,n_d,nbMC)
       cellWiennerM0transientG{j} = randn(mj,n_d,M0transient,nbMC);   %  ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)
   end      
 
   %--- Display screen
   if ind_display_screen == 1
      disp('--- Beginning the construction of the parameters for each group')
   end
  
   %--- Initializing parameters values for the groups   
   Rsg            = zeros(ngroup,1);
   Rshg           = zeros(ngroup,1);
   Rshssg         = zeros(ngroup,1);
   Rf0            = zeros(ngroup,1);
   RDeltarg       = zeros(ngroup,1);
   RM0estimg      = zeros(ngroup,1);
   RepsilonDIFFg  = zeros(ngroup,1);
   RmDPg          = zeros(ngroup,1);
   RnbmDMAPg      = zeros(ngroup,1);     
   cellMatRg      = cell(ngroup,1);       % cellMatRg{j,1} =  MatRgj,  cellMatRg{ngroup,1}     
   cellMatRa      = cell(ngroup,1);       % cellMatRa{j,1} =  MatRaj,  cellMatRa{ngroup,1}     
   numfig         = 0;

   %--- Loop on the groups for computing MatReta_argj(mj,NnbMC)
   for j = 1:ngroup                                                 % DO NOT USE parfor loop                                          
      mj = Igroup(j);                                               % length mj of vector   Y^j = (H_jr1,... ,H_jrmj) of group j
      MatReta_dj = cellMatReta_d{j};                                % MatReta_dj(mj,n_d): realizations of Y^j of length mj = Igroup(j)
      
      %--- computing the parameters for group j  
      [sj,shj,shssj,f0j,Deltarj,M0estimj] = sub_solverDirectPartition_parameters_groupj(mj,n_d,icorrectif,f0_ref,ind_f0,coeffDeltar); 

      %--- Computing the DMAPS basis for group j using the isotropic kernel
      [epsilonDIFFj,mDPj,nbmDMAPj,MatRgj,MatRaj] = sub_solverDirectPartition_isotropic_kernel_groupj(ind_generator,j,mj,n_d,MatReta_dj, ...
                                                    epsilonDIFFmin,step_epsilonDIFF,iterlimit_epsilonDIFF,comp_ref, ...
                                                    ind_display_screen,ind_plot,numfig);

      %--- Loading the calculated parameters for group j  
      Rsg(j)           = sj;
      Rshg(j)          = shj;
      Rshssg(j)        = shssj;
      Rf0(j)           = f0j;
      RDeltarg(j)      = Deltarj;
      RM0estimg(j)     = M0estimj;
      RepsilonDIFFg(j) = epsilonDIFFj;
      RmDPg(j)         = mDPj;
      RnbmDMAPg(j)     = nbmDMAPj;    
      cellMatRg{j,1}   =  MatRgj;     % MatRgj(n_d,nbmDMAPj)
      cellMatRa{j,1}   =  MatRaj;     % MatRaj(n_d,nbmDMAPj)

      %--- Print the data inputs that are used for all the  groups  
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+'); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n ');                     
         fprintf(fidlisting,' --- Parameters for group %7i \n ',j); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n ');  
         fprintf(fidlisting,' sj           = %14.7e \n ',Rsg(j)); 
         fprintf(fidlisting,' shj          = %14.7e \n ',Rshg(j)); 
         fprintf(fidlisting,' shssj        = %14.7e \n ',Rshssg(j));          
         fprintf(fidlisting,'                                 \n '); 
         fprintf(fidlisting,' f0j          = %14.7e \n ',Rf0(j)); 
         fprintf(fidlisting,' Deltarj      = %14.7e \n ',RDeltarg(j)); 
         fprintf(fidlisting,' M0estimj     = %7i    \n ',RM0estimg(j));       
         fprintf(fidlisting,'                                 \n '); 
         fprintf(fidlisting,'                                 \n '); 
         fprintf(fidlisting,' mj           = %7i    \n ',mj); 
         fprintf(fidlisting,' mDPj         = %7i    \n ',RmDPg(j)); 
         fprintf(fidlisting,' nbmDMAPj     = %7i    \n ',RnbmDMAPg(j)); 
         fprintf(fidlisting,' epsilonDIFFj = %14.7e \n ',RepsilonDIFFg(j)); 
         fprintf(fidlisting,'                                 \n '); 
         fprintf(fidlisting,'                                 \n '); 
         fclose(fidlisting);  
      end
   end
   
   %--- Display screen
   if ind_display_screen == 1
      disp('--- End of the construction of the parameters for each group')
      disp(' ')
      disp('--- Beginning the learning for each group')
   end
  
   %--- Initialization cellMatReta_ar
   cellMatReta_ar = cell(ngroup,1);     % cellMatReta_ar{j} = MatReta_arj, with  MatReta_arj(mj,n_ar)

   %--- Loop on the groups (DO NOT PARALLELIZE THIS LOOP)
   for j = 1:ngroup                      
       mj           = Igroup(j);  
       MatReta_dj   = cellMatReta_d{j};  
       MatRgj       = cellMatRg{j};     % MatRgj(n_d,nbmDMAPj)
       MatRaj       = cellMatRa{j};     % MatRaj(n_d,nbmDMAPj)

       nbmDMAPj     = RnbmDMAPg(j);                          
       shssj        = Rshssg(j);
       shj          = Rshg(j);
       Deltarj      = RDeltarg(j);
       f0j          = Rf0(j);
       
       ArrayGaussj              = cellGauss{j};               %  ArrayGaussj(mj,n_d,nbMC)
       ArrayWiennerM0transientj = cellWiennerM0transientG{j}; %  ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)

       %--- Display screen
       if ind_display_screen == 1
          disp(' ');
          disp(['--- Learning group number ',num2str(j),' with mj = ',num2str(mj)']);
          disp(' ');
       end

       %--- Print
       if ind_print == 1
         fidlisting=fopen('listing.txt','a+'); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n ');                     
         fprintf(fidlisting,' --- Learning group number j = %5i with mj = %5i \n ',j,mj);
         fprintf(fidlisting,'                                 \n '); 
         fclose(fidlisting);  
       end

       %--- No constraints on H
       if ind_constraints == 0       % MatReta_arj(mj,n_ar)
          [MatReta_arj] = sub_solverDirectPartition_constraint0(mj,n_d,nbMC,n_ar,nbmDMAPj,M0transient,Deltarj,f0j,shssj,shj, ...
                                                     MatReta_dj,MatRgj,MatRaj,ArrayWiennerM0transientj,ArrayGaussj,ind_parallel,ind_print);   
       end
    
       %--- constraints are applied on H
       if ind_constraints >= 1       % MatReta_arj(mj,n_ar)
          if mj == 1
             [MatReta_arj] = sub_solverDirectPartition_constraint0mj1(mj,n_d,nbMC,n_ar,nbmDMAPj,M0transient,Deltarj,f0j,shssj,shj, ...
                                                     MatReta_dj,MatRgj,MatRaj,ArrayWiennerM0transientj,ArrayGaussj,ind_parallel,ind_print);
          end
          if mj >= 2
             [MatReta_arj] = sub_solverDirectPartition_constraint123(j,mj,n_d,nbMC,n_ar,nbmDMAPj,M0transient,Deltarj,f0j,shssj,shj, ...
                                           MatReta_dj,MatRgj,MatRaj,ArrayWiennerM0transientj,ArrayGaussj,ind_constraints,ind_coupling, ...
                                           epsc,iter_limit,Ralpha_relax,minVarH,maxVarH,ind_display_screen,ind_print,ind_parallel,numfig); 
          end
       end
       cellMatReta_ar{j} =  MatReta_arj; 
   end
   clear  MatReta_arj MatReta_dj ArrayGaussj ArrayWiennerM0transientj cellGauss cellWiennerM0transientG cellMatReta_d

   %--- Concatenation of MatReta_arj(mj,n_ar) into MatReta_ar(nu,n_ar) 
   MatReta_ar = zeros(nu,n_ar);                             
   for j = 1:ngroup 
       MatReta_arj = cellMatReta_ar{j};                % MatReta_arj(mj,n_ar)
       mj = Igroup(j);                                 % length mj of vector   Y^j = (H_jr1,... ,H_jrmj) of group j
       MatReta_ar(MatIgroup(j,1:mj),:) = MatReta_arj;  % MatReta_arj(mj,n_ar),MatIgroup(ngroup,nu)
   end  

   clear MatReta_arj cellMatReta_ar
   
   %-----------------------------------------------------------------------------------------------------------------------------------                         
   %                    Estimation of the measure concentration of H_ar with respect to H_d, 
   %                    whose realizations are MatReta_ar(nu,n_ar) and MatReta_ar(nu,n_d)
   %-----------------------------------------------------------------------------------------------------------------------------------

   %--- Estimation of the measure concentration d2mopt_ar = E{|| [H_ar] - [eta_d] ||^2} / || [eta_d] ||^2
   d2mopt_ar = 0; 
   ArrayH_ar = reshape(MatReta_ar,nu,n_d,nbMC);                                % ArrayH_ar(nu,n_d,nbMC),MatReta_ar(nu,n_ar)
   for ell = 1:nbMC
       d2mopt_ar = d2mopt_ar + (norm(ArrayH_ar(:,:,ell) - MatReta_d,'fro'))^2; % MatReta_d(nu,n_d)
   end
   clear ArrayH_ar
   deno2     = (norm(MatReta_d,'fro'))^2;
   d2mopt_ar = d2mopt_ar/(nbMC*deno2);

   if ind_display_screen == 1
      disp(' ');
      disp('------ Concentration of the measure of H_ar with respect to H_d'); 
      disp(['       d^2(m_opt)_ar = ', num2str(d2mopt_ar)]); 
      disp(' ');
   end

   if ind_print == 1
      fidlisting = fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' --- Concentration of the measure of H_ar with respect to H_d \n ');
      fprintf(fidlisting,'                                                 \n '); 
      fprintf(fidlisting,'         d^2(m_opt)_ar =  %14.7e \n ',d2mopt_ar); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end 

   %----------------------------------------------------------------------------------------------------------------------------------                         
   %   Estimation of the Kullback divergence between H_d and H_ar whose realizations are MatReta_d(nu,n_d) and MatReta_ar(nu,n_ar)
   %----------------------------------------------------------------------------------------------------------------------------------
   
   divKL = 0;
   if ind_Kullback == 1
      divKL = sub_solverDirect_Kullback(MatReta_ar,MatReta_d,ind_parallel); 
     
      if ind_display_screen == 1
         disp(' ');
         disp('------ Kullback-Leibler divergence of H_ar with respect to H_d'); 
         disp(['       divKL = ', num2str(divKL)]); 
         disp(' ');
      end
   
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+');
         fprintf(fidlisting,'      \n ');                                                                
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' --- Kullback-Leibler divergence of H_ar with respect to H_d \n ');
         fprintf(fidlisting,'                                                 \n '); 
         fprintf(fidlisting,'        divKL =  %14.7e \n ',divKL); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n '); 
         fclose(fidlisting);  
      end 
   end

   %----------------------------------------------------------------------------------------------------------------------------------                         
   %   Entropy of H_d and H_ar whose realizations are MatReta_d(nu,n_d) and MatReta_ar(nu,n_ar)
   %----------------------------------------------------------------------------------------------------------------------------------

   entropy_Hd  = 0;
   entropy_Har = 0;
   if ind_Entropy == 1
      [entropy_Hd]  = sub_solverDirect_Entropy(MatReta_d,ind_parallel);
      [entropy_Har] = sub_solverDirect_Entropy(MatReta_ar,ind_parallel);
   
      if ind_display_screen == 1
         disp(' ');
         disp('------ Entropy of Hd and Har'); 
         disp(['       entropy_Hd  = ', num2str(entropy_Hd)]);
         disp(['       entropy_Har = ', num2str(entropy_Har)]);
         disp(' ');
      end
   
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+');
         fprintf(fidlisting,'      \n ');                                                                
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' --- Entropy of Hd and Har \n ');
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'        entropy_Hd   =  %14.7e \n',entropy_Hd); 
         fprintf(fidlisting,'         entropy_Har  =  %14.7e \n',entropy_Har);  
         fprintf(fidlisting,'      \n ');                                                          
         fprintf(fidlisting,'      \n '); 
         fclose(fidlisting);  
      end
   end

   %----------------------------------------------------------------------------------------------------------------------------------                         
   %   Statistical dependence of the components of H_d and H_ar whose realizations are MatReta_d(nu,n_d) and MatReta_ar(nu,n_ar)
   %----------------------------------------------------------------------------------------------------------------------------------
   
   iHd  = 0;
   iHar = 0;
   if ind_MutualInfo == 1   % Unnormalized mutual informations     
      [iHd]  = sub_solverDirect_Mutual_Information(MatReta_d,ind_parallel);
      [iHar] = sub_solverDirect_Mutual_Information(MatReta_ar,ind_parallel);
      
      if ind_display_screen == 1
         disp(' ');
         disp('------ Mutual Information iHd and iHar for Hd and Har'); 
         disp(['       iHd  = ', num2str(iHd)]); 
         disp(['       iHar = ', num2str(iHar)]);
         disp(' ');
      end
   
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+');
         fprintf(fidlisting,'      \n ');                                                                
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' --- Mutual Information iHd and iHar for Hd and Har \n ');
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'        iHd   =  %14.7e \n',iHd); 
         fprintf(fidlisting,'         iHar  =  %14.7e \n',iHar);  
         fprintf(fidlisting,'      \n ');                                                          
         fprintf(fidlisting,'      \n '); 
         fclose(fidlisting);  
      end
   end

   %------------------------------------------------------------------------------------------------------------------------------------------                         
   %  plot statistics for H_d and H_ar, from realizations MatReta_d(nu,n_d) (training) and learned realizations MatReta_ar(nu,n_ar) (learning) 
   %------------------------------------------------------------------------------------------------------------------------------------------
   
   sub_solverDirectPartition_plot_Hd_Har(n_d,n_ar,MatReta_d,MatReta_ar,nbplotHsamples,nbplotHClouds,nbplotHpdf,nbplotHpdf2D, ...
                                      MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D,numfig);
   SAVERANDendDirectPartition   = rng;  
   ElapsedSolverDirectPartition = toc(TimeStartSolverDirectPartition );  

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'-------   Elapsed time for Task7_SolverDirectPartition \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'Elapsed Time   =  %10.2f\n',ElapsedSolverDirectPartition);   
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task7_SolverDirectPartition')
   end 
   return
end


          
      