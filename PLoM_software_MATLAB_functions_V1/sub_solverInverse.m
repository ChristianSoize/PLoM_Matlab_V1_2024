function  [n_ar,MatReta_ar,ArrayZ_ar,ArrayWienner,SAVERANDendInverse,d2mopt_ar,divKL,iHd,iHar,entropy_Hd,entropy_Har] = ...
             sub_solverInverse(nu,n_d,nbMC,MatReta_d,ind_generator,icorrectif,f0_ref,ind_f0,coeffDeltar,M0transient, ...
                              nbmDMAP,MatRg,MatRa,ind_type_targ,N_r,Rb_targ1,coNr,coNr2,MatReta_targ,eps_inv,Rb_targ2,Rb_targ3, ...
                              ind_coupling,iter_limit,epsc,alpha_relax1,iter_relax2,alpha_relax2,SAVERANDstartInverse, ...
                              ind_display_screen,ind_print,ind_plot,ind_parallel,MatRplotHsamples,MatRplotHClouds, ...
                              MatRplotHpdf,MatRplotHpdf2D,ind_Kullback,ind_Entropy,ind_MutualInfo) 

   %------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 08 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverInverse
   %  Subject      : solver PLoM for inverse problem for which targets are imposed, the targets being taken into account by constraints.
   %                 Computation of n_ar learned realizations MatReta_ar(nu,n_ar) of H_ar under the constraints defined by the targets
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
   %                          solverInverse_Verlet is M0transient*Deltar 
   %
   %--- parameters related to the ISDE-projection basis
   %
   %    nbmDMAP             : dimension of the ISDE-projection basis
   %    MatRg(n_d,nbmDMAP)  : matrix of the ISDE-projection basis
   %    MatRa(n_d,nbmDMAP)  = MatRg*(MatRg'*MatRg)^{-1} 
   %
   %--- for the inverse problem, paremeters and data used for imposing the targets as constraints
   %
   %    ind_type_targ                     : = 1, targets defined by giving N_r realizations
   %                                      : = 2, targets defined by giving target mean-values 
   %                                      : = 3, targets defined by giving target mean-values and target covariance matrix
   %
   %                                      --- ind_type_targ = 1: targets defined by giving N_r realizations of XX_targ 
   %    N_r                               : number of realizations of the targets     
   %    Rb_targ1(N_r,1)                   : E{h_targ1(H^c)} = b_targ1  with h_targ1 = (h_{targ1,1}, ... , h_{targ1,N_r})
   %    coNr                              : parameter used for evaluating  E{h^c_targ(H^c)}               
   %    coNr2                             : parameter used for evaluating  E{h^c_targ(H^c)} 
   %    MatReta_targ(nu,N_r)              : N_r realizations of the projection of XX_targ on the model
   %    eps_inv                           : tolerance for computing the pseudo-inverse of matrix MatRGammaS_iter with 
   %                                        sub_solverInverse_pseudo_inverse(MatRGammaS_iter,eps_inv) for ind_type_targ = 1,
   %                                        in sub_solverInverse_constrainedByTargets. An adapted value is 0.001. If problems occurs
   %                                        increase the value to 0.01.
   %
   %                                     --- ind_type_targ = 2 or 3: targets defined by giving mean value of XX_targ
   %    Rb_targ2(nu,1)                                               yielding the constraint E{H^c} = b_targ2 
   %
   %                                     --- ind_type_targ = 3: targets defined by giving target covariance matrix of XX_targ
   %    Rb_targ3(nu,1)                                          yielding the constraint diag(E{H_c H_c'}) = b_targ3  
   %
   %--- parameters controling the convergence of the iteration algorithm related to the Lagrange multipliers for imposing the targets
   %
   %    ind_coupling = 0 : for ind_type_targ = 3, no coupling in  matrix MatRGammaS_iter (HIGHLY RECOMMENDED)
   %                 = 1 : for ind_type_targ = 3, coupling all the extra-diagonal blocs in matrix MatRGammaS_iter are kept
   %    iter_limit       : maximum number of iterations used by the iterative algorithm to compute the Lagrange multipliers. 
   %    epsc             : relative tolerance (for instance 1e-3) for the iteration-algorithm convergence 
   %                     NOTE: on the convergence criteria for the iteration algorithm computing the Lagrange multipliers:
   %                           Criterion 1: if iter > 10 and Rerr(iter)-Rerr(iter-1) > 0, there is a local minimum, obtained 
   %                                        for iter - 1. Convergence is then assumed to be reached and then, exit from the loop on iter  
   %                           Criterion 2: if Rerr(iter) <= epsc, the relative error of the constraint satisfaction is less than or equal 
   %                                        to the tolerance. Convergence is reached, and exit from the loop on iter.
   %                           Criterion 3: if iter > min(20,iter_limit) and Rtol(iter) < epsc,  the error is stationary and thus
   %                                        convergence is assumed to be reached and exit from the loop on iter  
   %                      --- Relaxation function  iter --> alpha_relax(iter) controlling convergence of the iterative algorithm 
   %                          is described by 3 parameters: alpha_relax1, iter_relax2, and alpha_relax2
   %    alpha_relax1     : value of alpha_relax for iter = 1  (for instance 0.001)
   %    iter_relax2      : value of iter (for instance, 20) such that  alpha_relax2 = alpha_relax(iter_relax2) 
   %                       if iter_relax2 = 1, then alpha_relax (iter) = alpha_relax2 for all iter >=1   
   %    alpha_relax2     : value of alpha_relax (for instance, 0.05) such that alpha_relax(iter >= iter_relax2) = apha_relax2
   %                       NOTE 2: If iter_relax2 = 1 , then Ralpha_relax(iter) = alpha_relax2 for all iter >=1
   %                       NOTE 3: If iter_relax2 >= 2, then  
   %                               for iter >= 1 and for iter < iter_relax2, we have:
   %                                   alpha_relax(iter) = alpha_relax1 + (alpha_relax2 - alpha_relax1)*(iter-1)/(iter_relax2-1)
   %                               for iter >= iter_relax2, we have:
   %                                   alpha_relax(iter) = alpha_relax2
   %                       NOTE 4: for decreasing the error err(iter), increase the value of iter_relax2
   %                       NOTE 5: if iteration algorithm dos not converge, decrease alpha_relax2 and/or increase iter_relax2  
   %
   %--- parameters and variables controling execution
   %
   %    SAVERANDstartInverse: state of the random generator at the end of the PCA step
   %    ind_display_screen  : = 0 no display,              = 1 display
   %    ind_print           : = 0 no print,                = 1 print
   %    ind_plot            : = 0 no plot,                 = 1 plot
   %    ind_parallel        : = 0 no parallel computation, = 1 parallel computation
   %
   %--- data for the plots
   %    MatRplotHsamples(1,nbplotHsamples) : contains the components numbers of H_ar for which the plot of the ealizations are made
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
   %--- OUPUTS  
   %      n_ar                        : number of realizations of H_ar such that n_ar  = nbMC x n_d
   %      MatReta_ar(nu,n_ar)         : n_ar realizations of H_ar  
   %      ArrayZ_ar(nu,nbmDMAP,nbMC)  : n_ar realizations of Z_ar  
   %      ArrayWienner(nu,n_d,nbMC)   : ArrayWienner(nu,n_d,nbMC) 
   %      SAVERANDEndInverse          : state of the random generator at the end of sub_solverInverse
   %      d2mopt_ar                   : concentration of the probability measure of H_ar with respect to H_d in the means-square sense
   %      divKL                       : Kullback-Leibler divergence of H_ar with respect to H_d
   %      iHd                         : Mutual Information iHd for Hd 
   %      iHar                        : Mutual Information iHar for Har
   %      entropy_Hd                  : Entropy of Hd
   %      entropy_Har                 : Entropy of Har
   %
   %--- INTERNAL PARAMETERS
   %
   %      s       : usual Silver bandwidth for the GKDE estimate (with the n_d points of the training dataset) 
   %                of the pdf p_H of H, having to satisfy the normalization condition E{H} = 0_nu and E{H H'} = [I_nu] 
   %      sh      : modified Silver bandwidth for wich the normalization conditions are satisfied for any value of nu >= 1 
   %      shss    : = 1 if icorrectif  = 0, and = sh/s if icorrectif = 1
   %      f_0     : damping parameter in the ISDE, which controls the speed to reach the stationary response of the ISDE
   %      Deltar  : Stormer-Verlet integration-step of the ISDE        
   %      M0estim : estimate of M0transient provided as a reference to the user
   %      Ralpha_relax(iter_limit,1): relaxation function for the iteration algorithm that computes the LaGrange multipliers
   %      n_ar    : number of learned realizations equal to nbMC*n_d:
   %
   %      nbplotHsamples : number >= 0 of the components numbers of H_ar for which the plot of the realizations are made   
   %      nbplotHClouds  : number >= 0 of the 3 components numbers of H_ar for which the plot of the clouds are made
   %      nbplotHpdf     : number >= 0 of the components numbers of H_d and H_ar for which the plot of the pdfs are made   
   %      nbplotHpdf2D   : number >= 0 of the 2 components numbers of H_d and H_ar for which the plot of the joint pdfs are made
   
   if ind_display_screen == 1                              
      disp('--- beginning Task9_SolverInverse')
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task9_SolverInverse \n ');
      fprintf(fidlisting,'      \n ');
      fclose(fidlisting); 
   end

   TimeStartSolverInverse = tic; 
   numfig                 = 0;
   n_ar                   = nbMC*n_d;

   nbplotHsamples = size(MatRplotHsamples,2);   % MatRplotHsamples(1,nbplotHsamples)
   nbplotHClouds  = size(MatRplotHClouds,1);    % MatRplotHClouds(nbplotHClouds,3)
   nbplotHpdf     = size(MatRplotHpdf,2);       % MatRplotHpdf(1,nbplotHpdf)
   nbplotHpdf2D   = size(MatRplotHpdf2D,1);     % MatRplotHpdf2D(nbplotHpdf2D,2)

   %--- initializing the random generator at the value of the end of the PCA step 
   rng(SAVERANDstartInverse);   

   %----------------------------------------------------------------------------------------------------------------------------------
   %                                    Check data, parameters, and initialization
   %---------------------------------------------------------------------------------------------------------------------------------- 
   
   if nu > n_d || nu < 1 || n_d < 1
      error('STOP1 in sub_solverInverse: nu > n_d or nu < 1 or n_d < 1');
   end   
   [n1temp,n2temp] = size(MatReta_d);              % MatReta_d(nu,n_d) 
   if n1temp ~= nu || n2temp ~= n_d
      error('STOP2 in sub_solverInverse: the dimensions of MatReta_d are not consistent with nu and n_d');
   end  
   if ind_generator ~= 0 && ind_generator ~= 1  
       error('STOP3 in sub_solverInverse: ind_generator must be equal to 0 or equal to 1');
   end  
   if icorrectif ~= 0 && icorrectif ~= 1  
       error('STOP4 in sub_solverInverse: icorrectif must be equal to 0 or equal to 1');
   end  
   if f0_ref <= 0
      error('STOP5 in sub_solverInverse: f0_ref must be strictly positif'); 
   end
   if ind_f0 ~= 0 && ind_f0 ~= 1       
      error('STOP6 in sub_solverInverse: ind_f0 must be equal to 0 or equal to 1');
   end
   if coeffDeltar <= 0
      error('STOP7 in sub_solverInverse: coeffDeltar must be strictly positive');
   end
   if M0transient <= 0
      error('STOP8 in sub_solverInverse: M0transient must be greater than or equal to 1');
   end
   if nbmDMAP < 1 || nbmDMAP > n_d
      error('STOP9 in sub_solverInverse: nbmDMAP < 1 or nbmDMAP > n_d');
   end
   [ndtemp,nbmDMAPtemp] = size(MatRg);              % MatRg(n_d,nbmDMAP) 
   if ndtemp ~= n_d || nbmDMAPtemp ~= nbmDMAP
      error('STOP10 in sub_solverInverse: the dimensions of MatRg are not consistent with n_d and nbmDMAP');
   end  
   [ndtemp,nbmDMAPtemp] = size(MatRa);              % MatRa(n_d,nbmDMAP) 
   if ndtemp ~= n_d || nbmDMAPtemp ~= nbmDMAP
      error('STOP11 in sub_solverInverse: the dimensions of MatRa are not consistent with n_d and nbmDMAP');
   end 
   if ind_type_targ ~= 1 && ind_type_targ ~= 2 && ind_type_targ ~= 3 
      error('STOP12 in sub_solverInverse: ind_type_targ must be equal to 1, 2, or 3');
   end
   if ind_type_targ == 1
      if N_r <= 0
         error('STOP13 in sub_solverInverse: for ind_type_targ = 1, N_r must be greater than or equal to 1');
      end
      if coNr <= 0
         error('STOP14 in sub_solverInverse: for ind_type_targ = 1, coNr must be positive');
      end
      if coNr2 <= 0
         error('STOP15 in sub_solverInverse: for ind_type_targ = 1, coNr2 must be positive');
      end
      [n1temp,n2temp] = size(Rb_targ1);                    % Rb_targ1(N_r,1) 
      if n1temp ~= N_r || n2temp ~= 1
         error('STOP16 in sub_solverInverse: for ind_type_targ = 1, the dimensions of Rb_targ1(N_r,1) are not coherent');
      end
      [n1temp,n2temp] = size(MatReta_targ);  % MatReta_targ(nu,N_r) 
      if n1temp ~= nu || n2temp ~= N_r
         error('STOP17 in sub_solverInverse: for ind_type_targ = 1, the dimensions of MatReta_targ(nu,N_r)  are not coherent');
      end      
      if eps_inv > 0.1 
         error('STOP18 in sub_solverInverse: for ind_type_targ = 1, eps_inv must be less than or equal to 0.1');
      end
   end
   if ind_type_targ == 2 || ind_type_targ == 3
      [n1temp,n2temp] = size(Rb_targ2);                    % Rb_targ2(nu,1) 
      if n1temp ~= nu || n2temp ~= 1
         error('STOP19 in sub_solverInverse: for ind_type_targ = 2 or 3, the dimensions of Rb_targ2(nu,1) are not coherent');
      end
   end
   if ind_type_targ == 3
      [n1temp,n2temp] = size(Rb_targ3);                    % Rb_targ3(nu,1) 
      if n1temp ~= nu || n2temp ~= 1
         error('STOP20 in sub_solverInverse: for ind_type_targ = 3, the dimensions of Rb_targ3(nu,1) are not coherent');
      end
      if ind_coupling ~= 0 && ind_coupling ~= 1
         error('STOP21 in sub_solverInverse: for ind_type_targ = 3, ind_coupling must be 0 or 1');
      end
   end   
   if iter_limit < 1       
      error('STOP22 in sub_solverInverse: iter_limit must be greater than or equal to 1');
   end
   if epsc < 0 || epsc >= 1       
      error('STOP23 in sub_solverInverse: epsc < 0 or epsc >= 1 ');
   end
   if alpha_relax1 < 0 || alpha_relax1 > 1
      error('STOP24 in sub_solverInverse: value of alpha_relax1 out the range [0,1]');
   end
   if alpha_relax2 < 0 || alpha_relax2 > 1
      error('STOP25 in sub_solverInverse: value of alpha_relax2 out the range [0,1]');
   end
   if iter_relax2 >= 2 && iter_relax2 <= iter_limit
      if alpha_relax1 >  alpha_relax2
          error('STOP26 in sub_solverInverse: alpha_relax1 must be less than or equal to alpha_relax2')
      end
   end
   if iter_relax2 > iter_limit
       error('STOP27 in sub_solverInverse: iter_relax2 must be less than or equal to iter_limit')
   end    
   if ind_display_screen ~= 0 && ind_display_screen ~= 1       
         error('STOP28 in sub_solverInverse: ind_display_screen must be equal to 0 or equal to 1')
   end
   if ind_print ~= 0 && ind_print ~= 1       
         error('STOP29 in sub_solverInverse: ind_print must be equal to 0 or equal to 1')
   end
   if ind_plot ~= 0 && ind_plot ~= 1       
         error('STOP30 in sub_solverInverse: ind_plot must be equal to 0 or equal to 1')
   end
   if ind_parallel ~= 0 && ind_parallel ~= 1       
         error('STOP31 in sub_solverInverse: ind_parallel must be equal to 0 or equal to 1')
   end
   if nbplotHsamples >= 1                          % MatRplotHsamples(1,nbplotHsamples)
      n1temp = size(MatRplotHsamples,1);
      if n1temp ~= 1 
         error('STOP32 in sub_solverInverse: the first dimension of MatRplotHsamples must be equal to 1') 
      end
      if any(MatRplotHsamples(1,:) < 1) || any(MatRplotHsamples(1,:) > nu)   % at least one integer is not within the valid range
         error('STOP33 in sub_solverInverse: at least one integer is not within the valid range for MatRplotHsamples') 
      end
   end
   if nbplotHClouds >= 1                           % MatRplotHClouds(nbplotHClouds,3)
      n2temp = size(MatRplotHClouds,2);
      if n2temp ~= 3
         error('STOP34 in sub_solverInverse: the second dimension of MatRplotHClouds must be equal to 3') 
      end
      if any(MatRplotHClouds(:) < 1) || any(MatRplotHClouds(:) > nu)   % At least one integer is not within the valid range
         error('STOP35 in sub_solverInverse: at least one integer is not within the valid range for MatRplotHClouds')         
      end
   end
   if nbplotHpdf >= 1                               % MatRplotHpdf(1,nbplotHpdf)
      n1temp = size(MatRplotHpdf,1);
      if n1temp ~= 1 
          error('STOP36 in sub_solverInverse: the first dimension of MatRplotHpdf must be equal to 1') 
      end
      if any(MatRplotHpdf(1,:) < 1) || any(MatRplotHpdf(1,:) > nu) % at least one integer  is not within the valid range
         error('STOP37 in sub_solverInverse: at least one integer is not within the valid range for MatRplotHpdf')            
      end
   end
   if nbplotHpdf2D >= 1                             % MatRplotHpdf2D(nbplotHpdf2D,2)
      n2temp = size(MatRplotHpdf2D,2);
      if n2temp ~= 2
         error('STOP38 in sub_solverInverse: the second dimension of MatRplotHpdf2D must be equal to 2') 
      end
      if any(MatRplotHpdf2D(:) < 1) || any(MatRplotHpdf2D(:) > nu)  % at least one integer is not within the valid range
         error('STOP39 in sub_solverInverse: at least one integer is not within the valid range for MatRplotHpdf2D')       
      end
   end
   
   %----------------------------------------------------------------------------------------------------------------------------------
   %                                    Computing the parameters used by solverInverse
   %---------------------------------------------------------------------------------------------------------------------------------- 
   
   [s,sh,shss,f0,Deltar,M0estim,Ralpha_relax] = sub_solverInverse_parameters(nu,n_d,icorrectif,f0_ref,ind_f0, ...
                                                       coeffDeltar,iter_limit,alpha_relax1,iter_relax2,alpha_relax2); 
   
   %----------------------------------------------------------------------------------------------------------------------------------                           
   %                                    Print data input for learning with imposed tragets
   %----------------------------------------------------------------------------------------------------------------------------------  

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+'); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n ');                     
      fprintf(fidlisting,' ---  Parameters for the learning imposing targets \n ');   
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
      fprintf(fidlisting,' f0            = %8.4f \n ',f0); 
      fprintf(fidlisting,' coeffDeltar   = %4i   \n ',coeffDeltar);    
      fprintf(fidlisting,' Deltar        = %14.7e\n ',Deltar);      
      fprintf(fidlisting,' M0transient   = %7i   \n ',M0transient); 
      fprintf(fidlisting,' M0estim       = %7i   \n ',M0estim); 
      fprintf(fidlisting,'      \n ');        
      fprintf(fidlisting,' nbmDMAP       = %7i \n ',nbmDMAP); 
      fprintf(fidlisting,'      \n ');       
      fprintf(fidlisting,' ind_type_targ = %1i \n ',ind_type_targ);
      fprintf(fidlisting,'      \n ');    
      if ind_type_targ == 1
         fprintf(fidlisting,' N_r           = %7i   \n ',N_r);
         fprintf(fidlisting,' coNr          = %14.7e\n ',coNr);  
         fprintf(fidlisting,' coNr2         = %14.7e\n ',coNr2);  
         fprintf(fidlisting,' eps_inv       = %14.7e\n ',eps_inv);  
         fprintf(fidlisting,'      \n ');    
      end
      fprintf(fidlisting,' ind_coupling  = %1i    \n ',ind_coupling);
      fprintf(fidlisting,' iter_limit    = %7i    \n ',iter_limit); 
      fprintf(fidlisting,' epsc          = %14.7e \n ',epsc); 
      fprintf(fidlisting,' alpha_relax1  = %8.4f  \n ',alpha_relax1); 
      fprintf(fidlisting,' iter_relax2   = %7i    \n ',iter_relax2); 
      fprintf(fidlisting,' alpha_relax2  = %8.4f  \n ',alpha_relax2);      
      fprintf(fidlisting,'      \n ');
      fprintf(fidlisting,' s             = %14.7e \n ',s); 
      fprintf(fidlisting,' shss          = %14.7e \n ',shss); 
      fprintf(fidlisting,' sh            = %14.7e \n ',sh); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ind_display_screen = %1i \n ',ind_display_screen); 
      fprintf(fidlisting,' ind_print          = %1i \n ',ind_print); 
      fprintf(fidlisting,' ind_plot           = %1i \n ',ind_plot); 
      fprintf(fidlisting,' ind_parallel       = %1i \n ',ind_parallel); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' nbplotHsamples     = %7i \n ',nbplotHsamples);
      fprintf(fidlisting,' nbplotHClouds      = %7i \n ',nbplotHClouds);
      fprintf(fidlisting,' nbplotHpdf         = %7i \n ',nbplotHpdf);
      fprintf(fidlisting,' nbplotHpdf2D       = %7i \n ',nbplotHpdf2D);
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ind_Kullback       = %1i \n ',ind_Kullback); 
      fprintf(fidlisting,' ind_Entropy        = %1i \n ',ind_Entropy); 
      fprintf(fidlisting,' ind_MutualInfo     = %1i \n ',ind_MutualInfo);
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end

   %-----------------------------------------------------------------------------------------------------------------------------------------                           
   %             Generating NnbMC realizations 
   %             For avoiding CPU time generated by " if test" on ind_constraints in sub_solverInverse_Verlet_constraintj.m and
   %             sub_solverInverse_Lrond_constraintj.m, the "if test" is implementent outside
   %-----------------------------------------------------------------------------------------------------------------------------------------  
    
                                                               %--- Generation of random Wienner germs for the Ito equation
   ArrayGauss              = randn(nu,n_d,nbMC);               %    ArrayGauss(nu,n_d,nbMC)
   ArrayWiennerM0transient = randn(nu,n_d,M0transient,nbMC);   %    ArrayWiennerM0transient(nu,n_d,M0transient,nbMC) 
   
   %--- Construction/saving of ArrayWienner(nu,n_d,nbMC), which contains of the nbMC realizations of the (nu,n_d) Wienner matrix for the last 
   %    integration time M0transient. This array is saved for a possible use in postprocessing for the polynomial chaos expansion (PCE) of Z
   ArrayTemp    = permute(ArrayWiennerM0transient,[1 2 4 3]);  % ArrayTemp(nu,n_d,nbMC,M0transient) 
   ArrayWienner = sqrt(Deltar)*ArrayTemp(:,:,:,M0transient);   % ArrayWienner(nu,n_d,nbMC)  
   clear ArrayTemp
  
   %--- constraints are applied on H
   % MatReta_ar(nu,NnbMC),ArrayZ_ar(nu,nbmDMAP,nbMC)

   [MatReta_ar,ArrayZ_ar] = sub_solverInverse_constrainedByTargets(nu,n_d,nbMC,n_ar,nbmDMAP,M0transient,Deltar,f0,shss,sh, ...
                                       MatReta_d,MatRg,MatRa,ArrayWiennerM0transient,ArrayGauss,ind_type_targ,N_r,Rb_targ1,coNr, ...
                                       coNr2,MatReta_targ,eps_inv,Rb_targ2,Rb_targ3,ind_coupling,epsc,iter_limit,Ralpha_relax, ...
                                       ind_display_screen,ind_print,ind_parallel,numfig); 

   clear  ArrayGauss ArrayWiennerM0transient 

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
      disp('------ Concentration of the probability measure learned with imposed targets');
      disp(['       d^2(m_opt)_ar = ', num2str(d2mopt_ar)]); 
      disp(' ');
   end

   if ind_print == 1
      fidlisting = fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' --- Concentration of the probability measure learned with imposed targets \n ');
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
      divKL = sub_solverInverse_Kullback(MatReta_ar,MatReta_d,ind_parallel); 
     
      if ind_display_screen == 1
         disp(' ');
         disp('------ Kullback-Leibler divergence of H_ar with respect to H_d for the inverse problem'); 
         disp(['       divKL = ', num2str(divKL)]); 
         disp(' ');
      end
   
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+');
         fprintf(fidlisting,'      \n ');                                                                
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' --- Kullback-Leibler divergence of H_ar with respect to H_d for the inverse problem \n ');
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
      [entropy_Hd]  = sub_solverInverse_Entropy(MatReta_d,ind_parallel);
      [entropy_Har] = sub_solverInverse_Entropy(MatReta_ar,ind_parallel);
   
      if ind_display_screen == 1
         disp(' ');
         disp('------ Entropy of Hd and Har for the inverse problem'); 
         disp(['       entropy_Hd  = ', num2str(entropy_Hd)]);
         disp(['       entropy_Har = ', num2str(entropy_Har)]);
         disp(' ');
      end
   
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+');
         fprintf(fidlisting,'      \n ');                                                                
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' --- Entropy of Hd and Har for the inverse problem \n ');
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
   if ind_MutualInfo == 1  % Unnormalized mutual informations      
      [iHd]  = sub_solverInverse_Mutual_Information(MatReta_d,ind_parallel);
      [iHar] = sub_solverInverse_Mutual_Information(MatReta_ar,ind_parallel);
      
      if ind_display_screen == 1
         disp(' ');
         disp('------ Mutual Information iHd and iHar for Hd and Har, for the inverse problem'); 
         disp(['       iHd  = ', num2str(iHd)]); 
         disp(['       iHar = ', num2str(iHar)]);
         disp(' ');
      end
   
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+');
         fprintf(fidlisting,'      \n ');                                                                
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' --- Mutual Information iHd and iHar for Hd and Har, for the inverse problem \n ');
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
   
   sub_solverInverse_plot_Hd_Har(n_d,n_ar,MatReta_d,MatReta_ar,nbplotHsamples,nbplotHClouds,nbplotHpdf,nbplotHpdf2D, ...
                                      MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D,numfig);
   SAVERANDendInverse   = rng;  
   ElapsedSolverInverse = toc(TimeStartSolverInverse);  

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'-------   Elapsed time for Task9_SolverInverse \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'Elapsed Time   =  %10.2f\n',ElapsedSolverInverse);   
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task9_SolverInverse')
   end 
   return
end


          
      