function  [MatReta_PCE,SAVERANDendPCE] =  sub_polynomial_chaosZWiener(nu,n_d,nbMC,nbmDMAP,MatRg,MatRa,n_ar,MatReta_ar,ArrayZ_ar, ...
                                    ArrayWienner,icorrectif,coeffDeltar,ind_PCE_ident,ind_PCE_compt,nbMC_PCE,Rmu,RM, ...
                                    mu_PCE,M_PCE,SAVERANDstartPCE,ind_display_screen,ind_print,ind_plot,ind_parallel, ...
                                    MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D) 

   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_polynomial_chaosZWiener
   %  Subject      : Polynomial chos expansion (PCE) [Z_PCE] of random matrix [Z_ar] (manifold), whose learned realizations are 
   %                 ArrayZ_ar(nu,nbmDMAP,nbMC). The learned realizations of [H_ar] = [Z_ar] [g]' (reshaped) are MatReta_ar(nu,n_ar)
   %                 with n_ar = n_d x nbMC. The PCE [H_PCE] = [Z_PCE][g]' of [H_ar] is constructed using the learned realizations 
   %                 ArrayZ_ar(nu,nbmDMAP,nbMC) of [H_ar], and the germ of the PCE is made up of an extraction of the Wienner realizations
   %                 ArrayWienner(nu,n_d,nbMC), which are used by the reduced-order ISDE for computing ArrayZ_ar(nu,nbmDMAP,nbMC). 
   %                 The nar_PCE = n_d x nbMC_PCE  <= n_ar realizations MatReta_PCE(nu,nar_PCE) of H_PCE are the reshaping of the realizations 
   %                 of [H_PCE]. The used theory is presented in [2]
   %
   %  Publications: [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
   %                       Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).
   %                [2] C. Soize, R. Ghanem, Polynomial chaos representation of databases on manifolds, Journal of Computational Physics, 
   %                       doi: 10.1016/j.jcp.2017.01.031, 335, 201-221 (2017).
   %                [3] C. Soize, Uncertainty Quantification. An Accelerated Course with Advanced Applications in Computational Engineering,
   %                       Interdisciplinary Applied Mathematics, doi: 10.1007/978-3-319-54339-0, Springer, New York,2017.
   %                [4] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
   %                       American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
   %                [5] C. Soize, R. Ghanem, Physics-constrained non-Gaussian probabilistic learning on manifolds, 
   %                       International Journal for Numerical Methods in Engineering, doi: 10.1002/nme.6202, 121 (1), 110-145 (2020).
   %                [6] C. Soize, R. Ghanem, Probabilistic learning on manifolds constrained by nonlinear partial differential equations 
   %                       from small datasets, Computer Methods in Applied Mechanics and Engineering, doi:10.1016/j.cma.2021.113777, 
   %                       380, 113777 (2021).
   %                [7] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
   %                       Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).
   %                [8] C. Soize, Probabilistic learning inference of boundary value problem with uncertainties based on Kullback-Leibler 
   %                       divergence under implicit constraints, Computer Methods in Applied Mechanics and Engineering,
   %                       doi:10.1016/j.cma.2022.115078, 395, 115078 (2022). 
   %                [9] C. Soize, Probabilistic learning constrained by realizations using a weak formulation of Fourier transform of 
   %                       probability measures, Computational Statistics, doi:10.1007/s00180-022-01300-w, 38(4),1879–1925 (2023).
   %               [10] C. Soize, R. Ghanem, Probabilistic-learning-based stochastic surrogate model from small incomplete datasets 
   %                       for nonlinear dynamical systems,Computer Methods in Applied Mechanics and Engineering, 
   %                       doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024).
   %               [11] C. Soize, R. Ghanem, Transient anisotropic kernel for probabilistic learning on manifolds, 
   %                       Computer Methods in Applied Mechanics and Engineering, pp.1-44 (2024).
   %               
   %--- INPUTS 
   %
   %--- parameters related to the learning
   %    nu                         : dimension of random vector H_d, H_ar, and H_PCE
   %    n_d                        : number of points in the training set for H_d
   %    nbMC                       : number of learned realizations of (nu,n_d)-valued random matrix [H_ar]   
   %    nbmDMAP                    : dimension of the ISDE-projection basis
   %    MatRg(n_d,nbmDMAP)         : matrix of the ISDE-projection basis 
   %    MatRa(n_d,nbmDMAP)         : MatRa = MatRg*(MatRg'*MatRg)^{-1} 
   %    n_ar                       : number of realizations of H_ar such that n_ar  = nbMC x n_d
   %    MatReta_ar(nu,n_ar)        : n_ar realizations of H_ar  
   %
   %    ArrayZ_ar(nu,nbmDMAP,nbMC) : nbMC learned realizations of the (nu,nbmDMAP)-matrix valued random variable [H_ar]
   %    ArrayWienner(nu,n_d,nbMC)  : nbMC realizations of the (nu,n_d)-matrix-valued of the Wienner process used by the 
   %                                 reduced-order ISDE for computing ArrayZ_ar(nu,nbmDMAP,nbMC). 
   %    icorrectif                  = 0: usual Silveman-bandwidth formulation, the normalization conditions are not exactly satisfied
   %                                = 1: modified Silverman bandwidth, for any value of nu, the normalization conditions are verified  
   %    coeffDeltar                : coefficient > 0 (usual value is 20) for calculating Deltar for ISDE solver
   %
   %    ind_PCE_ident              = 0 : no identification of the PCE 
   %                               = 1 :    identification of the PCE 
   %    ind_PCE_compt              = 0 : no computation of the PCE with plot for given values mu_PCE of mu and M_PCE of M
   %                               = 1:     computation of the PCE with plot for given values mu_PCE of mu and M_PCE of M
   %    nbMC_PCE                   : number of realizations generated for [Z_PCE] and consequently for [H_PCE] with nbMC_PCE <= nbMC
   %                                 (HIGHLY RECOMMENDED TO TAKE nbMC_PCE = nbMC if possible)
   %--- parameters for PCE identification (ind_PCE_ident = 1):
   %    Rmu(nbmu,1)                : nbmu values of the dimension mu of the germ (Xi_1,...,Xi_mu) of the PCE with 1 <= mu <= nu, 
   %                                 explored to find the optimal value muopt
   %    RM(nbM,1)                  : nbM values of the maximum degree M of the PCE with 0 <= M, explored to find the optimal value Mopt
   %                                 M = 0 : multi-index  (0,0,...,0) 
   %                                 M = 1 : multi-indices (0,...,0) and (1,0,...,0), (0,1,...,0),..., (0,...,1): Gaussian representation
   %
   %--- parameters for computing the PCE for given values mu = mu_PCE  and M = M_PCE (ind_PCE_compt = 1):
   %    mu_PCE                      : value of mu for computing the realizations with the PCE with mu >= 1
   %    M_PCE                       : value of M  for computing the realizations with the PCE with M  >= 0 
   %
   %--- Parameters controlling random generator, print, plot, and parallel computation
   %    SAVERANDstartPCE    : state of the random generator at the start
   %    ind_display_screen  : = 0 no display,              = 1 display
   %    ind_print           : = 0 no print,                = 1 print
   %    ind_plot            : = 0 no plot,                 = 1 plot
   %    ind_parallel        : = 0 no parallel computation, = 1 parallel computation
   %
   %--- if ind_PCE_compt = 1 and ind_plot = 1, parameters and data controlling plots for H_ar and H_PCE
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
   %
   %--- OUPUT:
   %          MatReta_PCE(nu,nar_PCE) : nar_PCE = n_d*nbMC_PCE realizations of H_PCE as the reshaping of the nbMC_PCE realizations of [H_PCE]
   %          SAVERANDendPCE          : state of the random generator at the end of the function

   TimeStartPCE = tic; 
   numfig       = 0;    
   nar_PCE      = n_d*nbMC_PCE;   % number of realizations MatReta_PCE(nu,nar_PCE) of H_PCE as the reshaping of 
                                  % the nbMC_PCE realizations of [H_PCE]

   %--- initializing the random generator
   rng(SAVERANDstartPCE);  
   nbplotHsamples = size(MatRplotHsamples,2);   % MatRplotHsamples(1,nbplotHsamples)
   nbplotHClouds  = size(MatRplotHClouds,1);    % MatRplotHClouds(nbplotHClouds,3)
   nbplotHpdf     = size(MatRplotHpdf,2);       % MatRplotHpdf(1,nbplotHpdf)
   nbplotHpdf2D   = size(MatRplotHpdf2D,1);     % MatRplotHpdf2D(nbplotHpdf2D,2)

   if ind_display_screen == 1
      disp(' '); 
      disp('--- beginning Task14_PolynomialChaosZwiener')
      disp(' ');                                                                                                           
      if ind_PCE_ident == 1
         disp(' ind_PCE_ident  = 1: identification of the optimal values Mopt of M, for each given value of mu = Rmu(imu)');
         disp('                     of the PCE H_PCE of H_ar, using the learned realizations MatReta_ar(nu,n_ar) of H_ar ');
      end
      if ind_PCE_compt == 1
         disp(' ind_PCE_compt  = 1: computation of the realizations MatReta_PCE(nu,nar_PCE) of the PCE representation    ');
         disp('                     H_PCE of H_ar  for given mu = mu_PCE and M = M_PCE                                   ');
      end
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task14_PolynomialChaosZwiener \n ');
      fprintf(fidlisting,'                         \n '); 
      if ind_PCE_ident == 1
         fprintf(fidlisting,' ind_PCE_ident  = 1: identification of the optimal values Mopt of M, for each given value of mu = Rmu(imu), \n ');
         fprintf(fidlisting,'                     of the PCE H_PCE of H_ar, using the learned realizations MatReta_ar(nu,n_ar) of H_ar   \n ');
      end
      if ind_PCE_compt == 1
         fprintf(fidlisting,' ind_PCE_compt  = 1: computation of the realizations MatReta_PCE(nu,nar_PCE) of the PCE representation      \n ');
         fprintf(fidlisting,'                     H_PCE of H_ar  for given mu = mu_PCE and M = M_PCE                                     \n ');
      end
      fprintf(fidlisting,'      \n ');  
      fclose(fidlisting);  
   end

   %-----------------------------------------------------------------------------------------------------------------------------------                           
   %       Checking input parameters and input data
   %-----------------------------------------------------------------------------------------------------------------------------------  

   %--- Check parameters and data related to the learning step
   if nu > n_d || nu < 1 || n_d < 1
      error('STOP1 in sub_polynomial_chaosZWiener: nu > n_d or nu < 1 or n_d < 1')
   end  
   if nbMC < 1 
      error('STOP2 in sub_polynomial_chaosZWiener: nbMC < 1')
   end
   if nbmDMAP < 1 || nbmDMAP > n_d
      error('STOP3 in sub_polynomial_chaosZWiener: nbmDMAP < 1 or nbmDMAP > n_d')
   end
   [n1temp,n2temp] = size(MatRg);                                % MatRg(n_d,nbmDMAP) 
   if n1temp ~= n_d || n2temp ~= nbmDMAP
      error('STOP4 in sub_polynomial_chaosZWiener: the dimensions of MatRg are not consistent with n_d and nbmDMAP')
   end  
   [n1temp,n2temp] = size(MatRa);                                % MatRa(n_d,nbmDMAP) 
   if n1temp ~= n_d || n2temp ~= nbmDMAP
      error('STOP5 in sub_polynomial_chaosZWiener: the dimensions of MatRa are not consistent with n_d and nbmDMAP')
   end 
   if n_ar ~= n_d*nbMC
       error('STOP6 in sub_polynomial_chaosZWiener: n_ar must be equal to n_d*nbMC')
   end
  [n1temp,n2temp] = size(MatReta_ar);                            % MatReta_ar(nu,n_ar) 
   if n1temp ~= nu || n2temp ~= n_ar
      error('STOP7 in sub_polynomial_chaosZWiener: the dimensions of MatReta_ar are not consistent with nu and n_ar')
   end 
   [n1temp,n2temp,n3temp] = size(ArrayZ_ar);                      % ArrayZ_ar(nu,nbmDMAP,nbMC)
   if n1temp ~= nu || n2temp ~= nbmDMAP || n3temp ~= nbMC
      error('STOP8 in sub_polynomial_chaosZWiener: the dimensions of ArrayZ_ar are not consistent with nu, nbmDMAP, and nbMC')
   end 
   [n1temp,n2temp,n3temp] = size(ArrayWienner);                   % ArrayWienner(nu,n_d,nbMC)
   if n1temp ~= nu || n2temp ~= n_d || n3temp ~= nbMC
      error('STOP9 in sub_polynomial_chaosZWiener: the dimensions of ArrayWienner are not consistent with nu, n_d, and nbMC')
   end 
   if icorrectif ~= 0 && icorrectif ~= 1  
       error('STOP10 in sub_polynomial_chaosZWiener: icorrectif must be equal to 0 or equal to 1')
   end  
   if coeffDeltar < 1
      error('STOP11 in sub_polynomial_chaosZWiener: coeffDeltar must be greater than or equal to 1')
   end

   %--- Check parameters controlling the type of exec (PCE identification or computation of PCE)
   if ind_PCE_ident ~= 0 && ind_PCE_ident ~= 1  
       error('STOP12 in sub_polynomial_chaosZWiener: ind_PCE_ident must be equal to 0 or equal to 1')
   end  
   if ind_PCE_compt ~= 0 && ind_PCE_compt ~= 1  
       error('STOP13 in sub_polynomial_chaosZWiener: ind_PCE_compt must be equal to 0 or equal to 1')
   end 
   if nbMC_PCE < 1 || nbMC_PCE > nbMC 
      error('STOP14 in sub_polynomial_chaosZWiener: nbMC_PCE < 1 or nbMC_PCE > nbMC')
   end

   %--- Case ind_PCE_ident = 1 (parameters identification of the PCE)
   if ind_PCE_ident == 1
      nbmu  = size(Rmu,1);
      if nbmu < 1
         error('STOP15 in sub_polynomial_chaosZWiener: the dimension of Rmu must be greater than or equal to 1')
      end
      if length(Rmu) ~= length(unique(Rmu))
         error('STOP16 in sub_polynomial_chaosZWiener: there are repetitions in Rmu');  
      end
      if any(Rmu < 1) || any(Rmu > nu)
         error('STOP17 in sub_polynomial_chaosZWiener: at least one integer in Rmu is not within the valid range [1,nu]')                
      end
      nbM  = size(RM,1);
      if nbM < 1
         error('STOP18 in sub_polynomial_chaosZWiener: the dimension of RM must be greater than or equal to 1')
      end
      if length(RM) ~= length(unique(RM))
         error('STOP19 in sub_polynomial_chaosZWiener: there are repetitions in RM');  
      end
      if any(RM < 0) 
         error('STOP20 in sub_polynomial_chaosZWiener: at least one integer in RM is not within the valid range')                
      end
   end

   %--- Case  ind_PCE_compt = 1 (computation of the PCE for given mu = mu_PCE and M = M_PCE)
   if ind_PCE_compt == 1
      if mu_PCE < 1 || mu_PCE > nu 
         error('STOP21 in sub_polynomial_chaosZWiener:  integer mu_PCE is not within the valid range [1,nu]')
      end 
      if M_PCE < 0 
         error('STOP22 in sub_polynomial_chaosZWiener:  integer M_PCE must be greater than or equal to 0')
      end 
   end

   %--- Parameters controlling display, print, plot, and parallel computation
   if ind_display_screen ~= 0 && ind_display_screen ~= 1       
         error('STOP23 in sub_polynomial_chaosZWiener: ind_display_screen must be equal to 0 or equal to 1')
   end
   if ind_print ~= 0 && ind_print ~= 1       
         error('STOP24 in sub_solverDirect: ind_print must be equal to 0 or equal to 1')
   end
   if ind_plot ~= 0 && ind_plot ~= 1       
         error('STOP25 in sub_polynomial_chaosZWiener: ind_plot must be equal to 0 or equal to 1')
   end
   if ind_parallel ~= 0 && ind_parallel ~= 1       
         error('STOP26 in sub_polynomial_chaosZWiener: ind_parallel must be equal to 0 or equal to 1')
   end

   %--- if ind_PCE_compt = 1 and ind_plot = 1, parameters and data controlling plots for H_ar and H_PCE
   if ind_PCE_compt == 1 && ind_plot == 1
      if nbplotHsamples >= 1                          % MatRplotHsamples(1,nbplotHsamples)
         n1temp = size(MatRplotHsamples,1);
         if n1temp ~= 1 
            error('STOP27 in sub_polynomial_chaosZWiener: the first dimension of MatRplotHsamples must be equal to 1') 
         end
         if any(MatRplotHsamples(1,:) < 1) || any(MatRplotHsamples(1,:) > nu)   % at least one integer is not within the valid range
            error('STOP28 in sub_polynomial_chaosZWiener: at least one integer in MatRplotHsamples is not in range [1,nu]') 
         end
      end
      if nbplotHClouds >= 1                           % MatRplotHClouds(nbplotHClouds,3)
         n2temp = size(MatRplotHClouds,2);
         if n2temp ~= 3
            error('STOP29 in sub_polynomial_chaosZWiener: the second dimension of MatRplotHClouds must be equal to 3') 
         end
         if any(MatRplotHClouds(:) < 1) || any(MatRplotHClouds(:) > nu)   % At least one integer is not within the valid range
            error('STOP30 in sub_polynomial_chaosZWiener: at least one integer in MatRplotHClouds is not in range [1,nu]')         
         end
      end
      if nbplotHpdf >= 1                               % MatRplotHpdf(1,nbplotHpdf)
         n1temp = size(MatRplotHpdf,1);
         if n1temp ~= 1 
             error('STOP31 in sub_polynomial_chaosZWiener: the first dimension of MatRplotHpdf must be equal to 1') 
         end
         if any(MatRplotHpdf(1,:) < 1) || any(MatRplotHpdf(1,:) > nu) % at least one integer  is not within the valid range
            error('STOP32 in sub_polynomial_chaosZWiener: at least one integer in MatRplotHpdf is not in [1,nu]')            
         end
      end
      if nbplotHpdf2D >= 1                             % MatRplotHpdf2D(nbplotHpdf2D,2)
         n2temp = size(MatRplotHpdf2D,2);
         if n2temp ~= 2
            error('STOP33 in sub_polynomial_chaosZWiener: the second dimension of MatRplotHpdf2D must be equal to 2') 
         end
         if any(MatRplotHpdf2D(:) < 1) || any(MatRplotHpdf2D(:) > nu)  % at least one integer is not within the valid range
            error('STOP34 in sub_polynomial_chaosZWiener: at least one integer in MatRplotHpdf2D is not in [1,nu]')       
         end
      end
   end 

   %--- for PCE-parameters identification, checking that K < nbMC 
   if ind_PCE_ident == 1 
      MatRK = zeros(nbM,nbmu);
      for imu = 1:nbmu           
           mu    = Rmu(imu);                                                     % mu = germ dimension, Rmu(nbmu,1)
          for iM = 1:nbM
              M  = RM(iM);                                                       % max degree of chaos polynomials
              K  = fix(1.e-12 + factorial(mu + M)/(factorial(mu)*factorial(M))); % number of coefficients in the PCE 
              MatRK(iM,imu) = K;
              if K >= nbMC   
                 if ind_print == 1
                    fidlisting=fopen('listing.txt','a+');                     
                    fprintf(fidlisting,'      \n '); 
                    fprintf(fidlisting,['            mu         M        K        nbMC \n ']);
                    Rprint = [ mu  M  K  nbMC];
                    fprintf(fidlisting,'      %7i    %7i  %7i   %7i \n ', Rprint);
                    fprintf(fidlisting,'      \n '); 
                    fprintf(fidlisting,' STOP37 in sub_polynomial_chaosZWiener: K must be smaller than nbMC  \n '); 
                    fprintf(fidlisting,'      \n '); 
                    fclose(fidlisting); 
                 end         
                 disp(['     mu    M     K    nbMC'])
                 disp([ mu  M  K  nbMC])
                 disp(' K must be smaller than nbMC for the computation of the PCE coefficients')
                 error('STOP37 in sub_polynomial_chaosZWiener: K must be smaller than nbMC')
              end
          end
      end
   end

   %--- for PCE computation, checking that K < nbMC 
   if ind_PCE_compt == 1                
      mu = mu_PCE;                                                       % mu = germ dimension, Rmu(nbmu,1)            
      M  = M_PCE;                                                        % max degree of chaos polynomials
      K  = fix(1.e-12 + factorial(mu + M)/(factorial(mu)*factorial(M))); % number of coefficients in the PCE 
      if K >= nbMC
         if ind_print == 1
            fidlisting=fopen('listing.txt','a+');                     
            fprintf(fidlisting,'      \n '); 
            fprintf(fidlisting,['          mu_PCE     M_PCE    K        nbMC \n ']);
            Rprint = [ mu  M  K  nbMC];
            fprintf(fidlisting,'      %7i    %7i  %7i   %7i \n ', Rprint);
            fprintf(fidlisting,'      \n '); 
            fprintf(fidlisting,' STOP38 in sub_polynomial_chaosZWiener: K must be smaller than nbMC  \n '); 
            fprintf(fidlisting,'      \n '); 
            fclose(fidlisting); 
         end         
         disp(['   mu_PCE M_PCE  K    nbMC'])
         disp([ mu  M  K  nbMC])
         disp(' K must be smaller than nbMC for the computation of the PCE coefficients')
         error('STOP38 in sub_polynomial_chaosZWiener: K must be smaller than nbMC')
      end            
   end
  
   %--- print parameters and data
   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');  
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' nu      = %9i \n ',nu); 
      fprintf(fidlisting,' n_d     = %9i \n ',n_d);  
      fprintf(fidlisting,' nbMC    = %9i \n ',nbMC); 
      fprintf(fidlisting,' n_ar    = %9i \n ',n_ar); 
      fprintf(fidlisting,' nbmDMAP = %9i \n ',nbmDMAP); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' icorrectif    = %1i \n ',icorrectif); 
      fprintf(fidlisting,' coeffDeltar   = %4i   \n ',coeffDeltar);    
      fprintf(fidlisting,' ind_PCE_ident   = %1i \n ',ind_PCE_ident); 
      fprintf(fidlisting,' ind_PCE_compt   = %1i \n ',ind_PCE_compt); 
      fprintf(fidlisting,' nbMC_PCE    = %9i \n ',nbMC_PCE); 
      fprintf(fidlisting,'      \n '); 
      if ind_PCE_ident == 1
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' nbmu    = %9i \n ',nbmu); 
         fprintf(fidlisting,'      \n ');  
         fprintf(fidlisting,'----- Values of mu (germ dimension) and M (max degree) to search the optimal values muopt and Mopt  \n '); 
         fprintf(fidlisting,'                \n ');  
         fprintf(fidlisting,' Rmu =          \n '); 
         fprintf(fidlisting,' %3i %3i %3i %3i  %3i %3i %3i %3i %3i %3i %3i %3i %3i %3i %3i %3i %3i %3i %3i %3i  \n ',Rmu');
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' RM =          \n '); 
         fprintf(fidlisting,' %3i %3i %3i %3i  %3i %3i %3i %3i %3i %3i %3i %3i  %3i %3i %3i %3i %3i %3i %3i %3i  \n ',RM');              
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' MatRK(nbM,nbmu) =      \n '); 
         for i = 1:nbM
             fprintf(fidlisting, '%9i', MatRK(i, :));  
             fprintf(fidlisting, '\n');  
         end
         fprintf(fidlisting,'      \n '); 
      end
      if ind_PCE_compt == 1
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' mu_PCE    = %9i \n ',mu_PCE); 
         fprintf(fidlisting,' M_PCE     = %9i \n ',M_PCE); 
         fprintf(fidlisting,' K         = %9i \n ',K); 
         fprintf(fidlisting,'      \n '); 
      end
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ind_display_screen = %1i \n ',ind_display_screen); 
      fprintf(fidlisting,' ind_print          = %1i \n ',ind_print); 
      fprintf(fidlisting,' ind_plot           = %1i \n ',ind_plot); 
      fprintf(fidlisting,' ind_parallel       = %1i \n ',ind_parallel); 
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting); 
   end

   %----------------------------------------------------------------------------------------------------------------------------------                           
   %       Construction of the realizations MatRxi(nu,nbMC) of the largest dimension germ RXi = (Xi_1,...Xi_nu) (max Rmu <= nu) 
   %       with RXi Gaussian, centered, and with covariance matrix equal to [I_nu], from ArrayWienner(nu,n_d,nbMC). Therefore 
   %       Xi_1,...,Xi_nu are statistically independent
   %-----------------------------------------------------------------------------------------------------------------------------------
                    
   % MatRxi(nu,nbMC),ArrayWienner(nu,n_d,nbMC),MatRa(n_d,nbmDMAP) 
     
   RrondA = (sum(MatRa.^2,1))';                                 % RrondA(nbmDMAP,1),MatRa(n_d,nbmDMAP)
   Rtemp  = 1./sqrt(RrondA);                                    % Rtemp(nbmDMAP,1)
   Rahat  = MatRa*Rtemp;                                        % Rahat(n_d,1)

   %--- Computing the step Deltar used for solving the ISDE
   s = ((4/((nu+2)*n_d))^(1/(nu+4)));                           % usual Silver bandwidth  
   s2 = s*s;   
   if icorrectif == 0                     
      shss = 1;
   end
   if icorrectif == 1
      shss = 1/sqrt(s2+(n_d-1)/n_d);                          
   end     
   sh     = s*shss;  
   Deltar = 2*pi*sh/coeffDeltar;  

   %--- Computing MatRxi(nu,nbMC)
   coef   = 1/((sqrt(Deltar))*norm(Rahat));
   MatRxi = zeros(nu,nbMC);                                     % MatRxi(nu,nbMC)
   for ell=1:nbMC
       MatRxi(:,ell) = coef*ArrayWienner(:,:,ell)*Rahat;        % MatRxi(nu,nbMC),ArrayWienner(nu,n_d,nbMC),Rahat(n_d,1)
   end
   ArrayZhat = permute(ArrayZ_ar,[1 3 2]);                      % ArrayZhat(nu,nbMC,nbmDMAP),ArrayZ_ar(nu,nbmDMAP,nbMC)
   
   %--- Computing the matrix-valued mean value [Zbar_ar] of [Z_ar] 
   MatRZbar_ar     = mean(ArrayZ_ar,3);                         % MatRZbar_ar(nu,nbmDMAP),ArrayZ_ar(nu,nbmDMAP,nbMC)

   %--- Computing the square of the norm : E{||[Z_ar] - [Zbar_ar]||_F^2 } with the nbMC realizations of [Z_ar]
   normMatRZ2_ar   = 0;
   for ell = 1:nbMC
       MatRtemp_ell  = ArrayZ_ar(:,:,ell) - MatRZbar_ar;        % MatRtemp_ell(nu,nbmDMAP),MatRZbar_ar(nu,nbmDMAP),ArrayZ_ar(nu,nbmDMAP,nbMC)
       normMatRZ2_ar = normMatRZ2_ar + sum(MatRtemp_ell(:).^2); 
   end      
   norm2Zc  = normMatRZ2_ar/nbMC;
  
   %-------------------------------------------------------------------------------------------------------------------------------                           
   %       ind_PCE_ident = 1  : identification of the optimal values Mopt of M for each considered value of mu = Rmu(imu) and 
   %                            loading the optimal values in RMopt(imu), imu = 1 : nbmu   
   %-------------------------------------------------------------------------------------------------------------------------------

   if ind_PCE_ident == 1  
      RMopt           = zeros(nbmu,1);       % RMopt(nbmu,1)
      RJopt           = zeros(nbmu,1);       % RJopt(nbmu,1)
      MatRJ           = zeros(nbM,nbmu);     % MatRJ(nbM,nbmu)
      MatRerrorZL2    = zeros(nbM,nbmu);     % MatRerrorZL2(nbM,nbmu)  
      MatReta_PCE     = [];
      
      for imu = 1:nbmu  
          mu            = Rmu(imu);                                                    % mu = germ dimension, Rmu(nbmu,1)
          MatRxi_mu     = MatRxi(1:mu,:);                                              % MatRxi_mu(mu,nbMC),MatRxi(nu,nbMC)
          MatRxi_mu_PCE = randn(mu,nbMC_PCE);                                          % MatRxi_mu_PCE(mu,nbMC_PCE)      
          RJ_mu         = zeros(nbM,1);                                                % RJ_mu(nbM,1)
          
          %--- Vectorized sequence
          if ind_parallel == 0
             for iM = 1:nbM 
                 M        = RM(iM);                                                       % max degree of chaos polynomials
                 K        = fix(1.e-12 + factorial(mu + M)/(factorial(mu)*factorial(M))); % number of coefficients in the PCE 
                                                                                          % (including index (0,0,...,0) 
                 %--- Construction of the realizations MatRPsiK(K,nbMC) of the chaos Psi_{Ralpha^(k)}(Xi) 
                 %
                 %    Xi = (Xi_1,...,Xi_mu)
                 %    Rbeta^(k) = (beta_1^(k) , ... , beta_mu^(k)) in R^mu with k = 1,...,K
                 %    Rbeta^(1) = (     0      , ... ,      0      ) in R^mu 
                 %    0 <= beta_1^(k) + ... + beta_mu^(k) <= M  for k = 1,...,K
                 %    Psi_{Rbeta^(1)}(Xi) = 1
                 %    MatRPsiK(k,ell) =  Psi_{Rbeta^(k)}(xi^ell), xi^ell = MatRxi(1:mu,ell), ell=1:nbMC              
                 %    MatRPsiK(K,nbMC)      
                 [MatRPsiK] = sub_polynomial_chaosZWienner_PCE(K,M,mu,nbMC,MatRxi_mu); % MatRPsiK(K,nbMC)  
   
                 %--- Construction of ArrayRyhat(nu,K,nbmDMAP)   
                 ArrayRyhat = zeros(nu,K,nbmDMAP);               % ArrayRyhat(nu,K,nbmDMAP)
                 co = 1/(nbMC-1);
                 for alpha = 1:nbmDMAP                           % ArrayRyhat(nu,K,nbmDMAP),ArrayZhat(nu,nbMC,nbmDMAP),MatRPsiK(K,nbMC)
                     ArrayRyhat(:,:,alpha) = co*ArrayZhat(:,:,alpha)*MatRPsiK'; 
                 end
   
                 %--- Construction of nbMC_PCE realizations MatRPsiK_PCE(K,nbMC_PCE) of the chaos Psi_{Ralpha^(k)}(Xi) 
                [MatRPsiK_PCE] = sub_polynomial_chaosZWiener_PCE(K,M,mu,nbMC_PCE,MatRxi_mu_PCE); % MatRPsiK_PCE(K,nbMC_PCE),MatRxi_mu_PCE(mu,nbMC_PCE)
   
                 %--- Construction of ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP) 
                 ArrayZhatPCE_mu = zeros(nu,nbMC_PCE,nbmDMAP);           % ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)
                 for alpha = 1:nbmDMAP                                   % ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP),MatRPsiK_PCE(K,nbMC_PCE)
                     ArrayZhatPCE_mu(:,:,alpha) = ArrayRyhat(:,:,alpha)*MatRPsiK_PCE; % ArrayRyhat(nu,K,nbmDMAP) 
                 end        
   
                 %--- Constructing ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
                 ArrayZ_PCE_mu = permute(ArrayZhatPCE_mu,[1 3 2]);      % ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE),ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)  
   
                 %--- Computing MatRerrorZbar(iM,imu)  and RJ1(iM)    
                 MatRZbar_PCE_mu = mean(ArrayZ_PCE_mu,3);               % MatRZbar_PCE_mu(nu,nbmDMAP),ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
   
                 %--- Computing MatRerrorZL2(iM,imu)
                 normMatRZ2_PCE_mu   = 0;
                 for ell = 1:nbMC_PCE                                                    % ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
                     MatRtemp_ell  = ArrayZ_PCE_mu(:,:,ell) - MatRZbar_PCE_mu;           % MatRtemp_ell(nu,nbmDMAP)
                     normMatRZ2_PCE_mu = normMatRZ2_PCE_mu + norm(MatRtemp_ell,'fro')^2;
                 end  
                 norm2Zc_PCE_mu = normMatRZ2_PCE_mu/nbMC_PCE;
                 MatRerrorZL2(iM,imu) = abs(norm2Zc - norm2Zc_PCE_mu)/norm2Zc;           % MatRerrorZL2(nbM,nbmu)  
   
                 %--- Cost function J_mu(iM) with a L2-norm of  ArrayZ_ar(nu,nbmDMAP,nbMC) and ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)                    
                 RJ_mu(iM)  =  MatRerrorZL2(iM,imu); 
             end              % end for iM = 1:nbM
          end

          %--- Parallel computation
          if ind_parallel == 1
             parfor iM = 1:nbM 
                 M        = RM(iM);                                                       % max degree of chaos polynomials
                 K        = fix(1.e-12 + factorial(mu + M)/(factorial(mu)*factorial(M))); % number of coefficients in the PCE 
                                                                                          % (including index (0,0,...,0) 
                 %--- Construction of the realizations MatRPsiK(K,nbMC) of the chaos Psi_{Ralpha^(k)}(Xi) 
                 %
                 %    Xi = (Xi_1,...,Xi_mu)
                 %    Rbeta^(k) = (beta_1^(k) , ... , beta_mu^(k)) in R^mu with k = 1,...,K
                 %    Rbeta^(1) = (     0      , ... ,      0      ) in R^mu 
                 %    0 <= beta_1^(k) + ... + beta_mu^(k) <= M  for k = 1,...,K
                 %    Psi_{Rbeta^(1)}(Xi) = 1
                 %    MatRPsiK(k,ell) =  Psi_{Rbeta^(k)}(xi^ell), xi^ell = MatRxi(1:mu,ell), ell=1:nbMC              
                 %    MatRPsiK(K,nbMC)      
                 [MatRPsiK] = sub_polynomial_chaosZWiener_PCE(K,M,mu,nbMC,MatRxi_mu);     % MatRPsiK(K,nbMC)  
   
                 %--- Construction of ArrayRyhat(nu,K,nbmDMAP)   
                 ArrayRyhat = zeros(nu,K,nbmDMAP);               % ArrayRyhat(nu,K,nbmDMAP)
                 co = 1/(nbMC-1);
                 for alpha = 1:nbmDMAP                           % ArrayRyhat(nu,K,nbmDMAP),ArrayZhat(nu,nbMC,nbmDMAP),MatRPsiK(K,nbMC)
                     ArrayRyhat(:,:,alpha) = co*ArrayZhat(:,:,alpha)*MatRPsiK'; 
                 end
   
                 %--- Construction of nbMC_PCE realizations MatRPsiK_PCE(K,nbMC_PCE) of the chaos Psi_{Ralpha^(k)}(Xi) 
                [MatRPsiK_PCE] = sub_polynomial_chaosZWiener_PCE(K,M,mu,nbMC_PCE,MatRxi_mu_PCE); % MatRPsiK_PCE(K,nbMC_PCE),MatRxi_mu_PCE(mu,nbMC_PCE)
   
                 %--- Construction of ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP) 
                 ArrayZhatPCE_mu = zeros(nu,nbMC_PCE,nbmDMAP);           % ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)
                 for alpha = 1:nbmDMAP                                   % ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP),MatRPsiK_PCE(K,nbMC_PCE)
                     ArrayZhatPCE_mu(:,:,alpha) = ArrayRyhat(:,:,alpha)*MatRPsiK_PCE; % ArrayRyhat(nu,K,nbmDMAP) 
                 end        
   
                 %--- Constructing ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
                 ArrayZ_PCE_mu = permute(ArrayZhatPCE_mu,[1 3 2]);      % ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE),ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)  
   
                 %--- Computing MatRerrorZbar(iM,imu)  and RJ1(iM)    
                 MatRZbar_PCE_mu = mean(ArrayZ_PCE_mu,3);               % MatRZbar_PCE_mu(nu,nbmDMAP),ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
   
                 %--- Computing MatRerrorZL2(iM,imu)
                 normMatRZ2_PCE_mu   = 0;
                 for ell = 1:nbMC_PCE                                                    % ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
                     MatRtemp_ell  = ArrayZ_PCE_mu(:,:,ell) - MatRZbar_PCE_mu;           % MatRtemp_ell(nu,nbmDMAP)
                     normMatRZ2_PCE_mu = normMatRZ2_PCE_mu + norm(MatRtemp_ell,'fro')^2;
                 end  
                 norm2Zc_PCE_mu = normMatRZ2_PCE_mu/nbMC_PCE;
                 MatRerrorZL2(iM,imu) = abs(norm2Zc - norm2Zc_PCE_mu)/norm2Zc;           % MatRerrorZL2(nbM,nbmu)  
   
                 %--- Cost function J_mu(iM) with a L2-norm of  ArrayZ_ar(nu,nbmDMAP,nbMC) and ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)                    
                 RJ_mu(iM)  =  MatRerrorZL2(iM,imu);                  
             end              % end parfor iM = 1:nbM
          end

          MatRJ(:,imu)  = RJ_mu;                                                      % MatRJ(nbM,nbmu), RJ_mu(nbM)  

          %--- find optimal value Mopt(mu) of M for given mu based of RJ_mu
          [Jopt_mu,iMopt_mu] = min(RJ_mu);       % RJ_mu(nbM)  
          RMopt(imu)         = RM(iMopt_mu);     % RMopt(nbmu)
          RJopt(imu)         = Jopt_mu;          % RJopt(nbmu)
      end      % end parfor imu = 1:nbmu   
     
      clear MatRxi_mu_PCE MatRPsiK ArrayRyhat MatRPsiK_PCE ArrayZhatPCE_mu ArrayZ_PCE_mu
      clear RerrorOVL MatRZbar_PCE_mu MatRZbar_ar MatRZ_ar MatRZ_PCE_mu

      
      %--- Print the cost function and the optimal values based on RJ
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+');  
         fprintf(fidlisting,'               \n ');
         fprintf(fidlisting,'               \n ');  
         fprintf(fidlisting,' --- Optimal value Mopt of M as a function of mu based of RJ \n '); 
         fprintf(fidlisting,'     computed using the L2-norm \n ');  
         fprintf(fidlisting,'               \n ');
         fprintf(fidlisting,'   mu  Mopt(mu)  Jopt(mu) \n ');  
         fprintf(fidlisting,'      \n '); 
         for imu = 1:nbmu
             Rprint = [Rmu(imu) RMopt(imu) RJopt(imu)];         
             fprintf(fidlisting,'  %3i  %3i   %14.7e \n ',Rprint);
         end
         fprintf(fidlisting,'               \n ');  
         fprintf(fidlisting,'               \n ');  
         fprintf(fidlisting,' --- Values of the cost function J(mu,M) (second-order moment) and number of PCE coefficients \n ');  
         fprintf(fidlisting,'     computed using the L2-norm \n ');  
         fprintf(fidlisting,'               \n ');
         fprintf(fidlisting,'  mu     M       J(mu,M)             K \n '); 
         fprintf(fidlisting,'      \n ');
         for imu = 1:nbmu
             mu  = Rmu(imu);                                                        % germ dimension
             for iM =  1:nbM
                 M   = RM(iM);                                                      % max degree of chaos polynomials
                 K   = fix(1.e-12 +factorial(mu + M)/(factorial(mu)*factorial(M))); % number of coefficients in the PCE (including index (0,0,...,0) 
                 J   = MatRJ(iM,imu);                                               % MatRJ(nbM,nbmu)
                 Rprint = [mu M J K];  
                 fprintf(fidlisting,' %3i   %3i   %14.7e  %9i \n ',Rprint);
             end
             fprintf(fidlisting,'      \n ');
         end  
         fprintf(fidlisting,'               \n ');  
         fprintf(fidlisting,'               \n ');  
         fprintf(fidlisting,' --- Values of the second-order moment of Z_PCE as a function of mu and M  \n ');                        
         fprintf(fidlisting,'               \n ');
         fprintf(fidlisting,'  mu     M   error_L2(mu,M) \n '); 
         fprintf(fidlisting,'      \n ');
         for imu = 1:nbmu
             mu  = Rmu(imu);                                                        % germ dimension
             for iM =  1:nbM
                 M   = RM(iM);                                                      % max degree of chaos polynomials
                 Rprint = [mu M MatRerrorZL2(iM,imu) ];  
                 fprintf(fidlisting,' %3i   %3i   %14.7e \n ',Rprint);
             end
             fprintf(fidlisting,'      \n ');
         end   
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n ');  
         fclose(fidlisting); 
      end
      
      %--- plot the family of curves { M--> J(M,mu) }_mu 
      if ind_plot == 1
         if nbM >= 2
            h = figure;
            hold on      
            legendEntries = cell(nbmu, 1);                           % Create a cell array to store legend entries      
            for imu = 1:nbmu
                mu = Rmu(imu);
                plot(RM, MatRJ(:,imu), '-')                          % RM(nbM,1),MatRJ(nbM,nbmu)
                legendEntries{imu} = ['$\mu = ', num2str(mu), '$'];  % Store legend entry for this plot
            end  
            title(['Function $M\mapsto J(M,\mu)$ computed with the $L^2$-norm'],'FontSize',16,'Interpreter','latex','FontWeight','normal');
            xlabel('$M$','FontSize',16,'Interpreter','latex')
            ylabel('$J(M,\mu)$','FontSize',16,'Interpreter','latex')      
            legend(legendEntries,'Interpreter','latex','Location','northeast')  % Add legend with entries
            hold off  
            numfig = numfig + 1;
            saveas(h,['figure_PolynomialChaosZWiener_J(M) with L2-norm.fig']);
            close(h);
         end
      end
   end

   %------------------------------------------------------------------------------------------------------------------------------                           
   %       ind_PCE_compt = 1: PCE computation for a given value  mu_PCE and M_PCE of mu and M 
   %------------------------------------------------------------------------------------------------------------------------------
  
   if ind_PCE_compt == 1      
      mu = mu_PCE;
      M  = M_PCE;
      K  = fix(1.e-12 + factorial(mu + M)/(factorial(mu)*factorial(M)));    % number of coefficients in the PCE (including index (0,0,...,0)
       
      %--- re-building the PCE coefficients for mu_PCE and M_PCE
      MatRxi_mu  = MatRxi(1:mu,:);                                           % MatRxi_mu(mu,nbMC), MatRxi_mu(nu,nbMC)
      [MatRPsiK] = sub_polynomial_chaosZWiener_PCE(K,M,mu,nbMC,MatRxi_mu);          % MatRPsiK(K,nbMC)
      co  = 1/(nbMC-1);
      ArrayRyhat = zeros(nu,K,nbmDMAP);                                      % ArrayRyhat(nu,K,nbmDMAP)
      for alpha = 1:nbmDMAP 
          ArrayRyhat(:,:,alpha) = co*ArrayZhat(:,:,alpha)*MatRPsiK';  % ArrayRyhat(nu,K,nbmDMAP), ArrayZhat(nu,nbMC,nbmDMAP),MatRPsiK(K,nbMC)
      end      
      clear MatRPsiK MatRxi_mu ArrayZhat
      
      %%%%%%%%% SURROGATE MODEL DEFINED BY PCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      %--- Construction of nbMC_PCE realizations MatRPsiK_PCE(K,nbMC_PCE) of the chaos Psi_{Ralpha^(k)}(Xi) 
      MatRxi_mu_PCE  = randn(mu,nbMC_PCE);                                  % MatRxi_mu_PCE(mu,nbMC_PCE)      
      [MatRPsiK_PCE] = sub_polynomial_chaosZWiener_PCE(K,M,mu,nbMC_PCE,MatRxi_mu_PCE); % MatRPsiK_PCE(K,nbMC_PCE)
      clear MatRxi_mu_PCE 
 
      %--- Construction of ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)    
      ArrayZhatPCE_mu = zeros(nu,nbMC_PCE,nbmDMAP);                         % ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)
      for alpha = 1:nbmDMAP                                                 % ArrayRyhat(nu,K,nbmDMAP),MatRPsiK_PCE(K,nbMC_PCE) 
          ArrayZhatPCE_mu(:,:,alpha) = ArrayRyhat(:,:,alpha)*MatRPsiK_PCE;  % ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)
      end
      clear ArrayRyhat MatRPsiK_PCE
 
      %--- Constructing ArrayH_PCE(nu,n_d,nbMC_PCE)
      ArrayZ_PCE = permute(ArrayZhatPCE_mu,[1 3 2]);        % ArrayZ_PCE(nu,nbmDMAP,nbMC_PCE),ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)
      ArrayH_PCE = zeros(nu,n_d,nbMC_PCE);                                           
      for ell=1:nbMC_PCE
          ArrayH_PCE(:,:,ell) = ArrayZ_PCE(:,:,ell)*MatRg'; % ArrayH_PCE(nu,n_d,nbMC_PCE),ArrayZ_PCE(nu,nbmDMAP,nbMC_PCE),MatRg(n_d,nbmDMAP) 
      end
      MatReta_PCE = reshape(ArrayH_PCE,nu,nar_PCE);       % MatReta_PCE(nu,nar_PCE)
      clear ArrayZ_PCE  ArrayZhatPCE_mu  ArrayH_PCE  

      %%%%%%%%% END SEQUENCE OF THE SURROGATE PCE MODEL
                    
      %--- plot statistics for H_ar from realizations MatReta_ar(nu,n_ar) (learning) and plot statistics  for H_PCE from realizations 
      %    MatReta_PCE(nu,nar_PCE) computed with the PCE H_PCE of H_ar 
      if ind_plot == 1
         sub_polynomial_chaosZWiener_plot_Har_HPCE(n_ar,nar_PCE,MatReta_ar,MatReta_PCE,nbplotHClouds,nbplotHpdf,nbplotHpdf2D, ...
                                         MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D,numfig);
      end
   end

   SAVERANDendPCE = rng;  
   ElapsedPCE     = toc(TimeStartPCE);  

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'-------   Elapsed time for Task14_PolynomialChaosZwiener \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'Elapsed Time   =  %10.2f\n',ElapsedPCE);   
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task14_PolynomialChaosZwiener')
   end 
   return
end


      