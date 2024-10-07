function [SAVERANDendPolynomialChaosQWU] = sub_polynomialChaosQWU(n_x,n_q,n_w,n_d,n_ar,nbMC,nu,MatRx_d,MatReta_ar,RmuPCA, ...
                                           MatRVectPCA,Indx_real,Indx_pos,Indq_obs,Indw_obs,nx_obs,Indx_obs,ind_scaling, ...
                                           Rbeta_scale_real,Ralpha_scale_real,Rbeta_scale_log,Ralpha_scale_log,nbMC_PCE, ...
                                           Ng,Ndeg,ng,ndeg,MaxIter,SAVERANDstartPolynomialChaosQWU,ind_display_screen,ind_print, ...
                                           ind_plot,ind_parallel)

    %----------------------------------------------------------------------------------------------------------------------------------------
    %
    %  Copyright: Christian Soize, Universite Gustave Eiffel, 05 October 2024
    %
    %  Software     : Probabilistic Learning on Manifolds (PLoM) 
    %  Function name: sub_polynomialChaosQWU
    %  Subject      : Post processing allowing the construction of the polynomial chaos expansion (PCE) 
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
    %               [10] C. Soize, R. Ghanem, Physical systems with random uncertainties: Chaos representation with arbitrary probability 
    %                       measure, SIAM Journal on Scientific Computing, doi:10.1137/S1064827503424505, 26}2), 395-410 (2004).
    %               [11] C. Soize, C. Desceliers, Computational aspects for constructing realizations of polynomial chaos in high 
    %                       dimension}, SIAM Journal On Scientific Computing, doi:10.1137/100787830, 32(5), 2820-2831 (2010).
    %               [12] G. Perrin, C. Soize, D. Duhamel, C. Funfschilling, Identification of polynomial chaos representations in 
    %                       high dimension from a set of realizations, SIAM Journal on Scientific Computing, doi:10.1137/11084950X, 
    %                       34(6), A2917-A2945 (2012).
    %               [13] C. Soize, Q-D. To, Polynomial-chaos-based conditional statistics for probabilistic learning with heterogeneous
    %                       data applied to atomic collisions of Helium on graphite substrate, Journal of Computational Physics,
    %                       doi:10.1016/j.jcp.2023.112582, 496, 112582, pp.1-20 (2024).
    % 
    %--- Algebraic representation of the polynomial chaos for RQ^0 = sum_alpha Rgamma_alpha Psi_alpha(Xi) values in RR^n_q with
    %       RQ^0         = random vector with values in  RR^n_q  
    %       Rgamma_alpha = coefficient with values in RR^n_q
    %       Psi_alpha(Xi)= real-valued polynomial chaos 
    %       Xi           = (Xi_1, ... , Xi_Ng) random vector for the germ
    %       Ng           = dimension of the germ Xi = (Xi_1, ... , Xi_Ng) 
    %       alpha        = (alpha_1,...,alpha_Ng) multi-index of length Ng   
    %       Ndeg         = max degree of the polynomial chaos                                      
    %       K0           = factorial(Ng+Ndeg)/(factorial(Ng)*factorial(Ndeg)) number of terms
    %                      including alpha^0 = (0, ... ,0) for which psi_alpha^0(Xi) = 1
    %
    %--- Algebraic representation of the polynomial chaos for [Gamma_r] = sum_a [g_a] phi_a(U) values in the set of (n_q x n_q) matrices
    %       [Gamma_r]    = random matrix with values in  (n_q x n_q) matrix
    %       [g_a]        = coefficient with values in (n_q x n_q) matrices 
    %       phi_a(U)     = real-valued normalized Hermite polynomial chaos 
    %       U            = (U_1, ... , U_ng) random vector for normalized Gaussian germ
    %       ng           = dimension of the germ U = (U_1, ... , U_ng) 
    %       a            = (a_1,...,a_ng) multi index of length ng   
    %       ndeg         = max degree of the polynomial chaos                                      
    %       KU           = factorial(ng+ndeg)/(factorial(ng)*factorial(ndeg)) = number
    %                      of terms NOT including a^0 = (0, ... ,0) for which phi_a^0(U) = 1
    %
    %--- INPUTS 
    %
    %     n_x                         : dimension of random vectors XX_ar  (unscaled) and X_ar (scaled)  
    %     n_q                         : dimension of random vector QQ (unscaled quantitity of interest)  1 <= n_q    
    %     n_w                         : dimension of random vector WW (unscaled control variable) with 1 <= n_w  
    %     n_d                         : number of points in the training set for XX_d and X_d  
    %     n_ar                        : number of points in the learning set for H_ar, X_obs, and XX_obs
    %     nbMC                        : number of learned realizations of (nu,n_d)-valued random matrix [H_ar]    
    %     nu                          : order of the PCA reduction, which is the dimension of H_ar 
    %     MatRx_d(n_x,n_d)            : n_d realizations of X_d (scaled)
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
    %     nbMC_PCE                    : number of learned realizations used for PCE is nar_PCE= nbMC_PCE x n_d 
    %                                   (HIGHLY RECOMMENDED TO TAKE nbMC_PCE = nbMC if possible) 
    %     Ng                          : dimension of the germ for polynomial chaos Psi_alpha that is such that Ng = nw_obs   
    %     Ndeg                        : max degree of the polynomial chaos Psi_alpha with Ndeg >= 1  
    %     ng                          : dimension of the germ for polynomial chaos phi_a that is such that ng >= 1  
    %     ndeg                        : max degree of the polynomial chaos phi_a with ndeg >= 0 (if ndeg = 0, then KU = 1)  
    %     MaxIter                     : maximum number of iteration used by the quasi-Newton optimization algorithm (example 400)
    %
    %     SAVERANDstartPolynomialChaosQWU : state of the random generator at the beginning
    %     ind_display_screen              : = 0 no display,              = 1 display
    %     ind_print                       : = 0 no print,                = 1 print
    %     ind_plot                        : = 0 no plot,                 = 1 plot
    %     ind_parallel                    : = 0 no parallel computation, = 1 parallel computation
    %
                                          %---- EXAMPLE OF DATA FOR EXPLAINING THE DATA STRUCTURE
                                          %     component numbers of qq     = 1:100
                                          %     component numbers of ww     = 1:20
                                          %     Indq_obs(nq_obs,1)          = [2 4 6 8 80 98]', nq_obs = 6 
                                          %     Indw_obs(nw_obs,1)          = [1 3 8 15 17]'  , nw_obs = 5
                                          %     nx_obs                      = 6 + 5 = 11
                                          %     Indx_obs    = [Indq_obs                     
                                          %                    n_q + Indw_obs] =  [2 4 6 8 80 98  101 103 108 115 117]'  
    %--- OUTPUT
    %          SAVERANDendPolynomialChaosQWU: state of the random generator at the end of the function
    %
    %--- REUSING THE CONSTRUCTED PCE ----------------------------------------------------------------------------------------------------
              % Example of script for reusing the PCE with function sub_polynomialChaosQWU_realization_chaos.m
              % ind_display_screen = 1;
              % ind_print          = 1;
              % ind_plot           = 1;
              % 
              % %--- Load filePolynomialChaosQWU_for_realization.mat
              % 
              % Define the file name
              % fileName = 'filePolynomialChaosQWU_for_realization.mat';
              % the structure of the filename to be uses corresponds to the following save:
              % save(fileName,'nw_obs','nq_obs','Indw_obs','Indq_obs','nar_PCE','MatRww_ar0','MatRqq_ar0','Ralpham1_scale_chaos', ...
              %               'Rbeta_scale_chaos','Ng','K0','MatPower0','MatRa0','ng','KU','MatPowerU','MatRaU','Jmax','n_y', ...
              %               'MatRgamma_opt','Indm','Indk','Ralpha_scale_yy','RQQmean','MatRVectEig1s2','-v7.3');
              % Check if the file exists 
              % if isfile(fileName)
              %     fprintf('The file "%s" exists.\n', fileName);
              %     load(fileName);
              %     fprintf('The file "%s" has been loaded.\n', fileName);
              % else
              %     fprintf('STOP-ERROR: the file "%s" does not exist.\n', fileName);
              % end
              % 
              % MatRqq_chaos = zeros(nq_obs,nar_PCE);
              % for ell = 1:nar_PCE                                      %... loop on the control variable
              %     Rww_ell = MatRww_ar0(:,ell);
              %     [Rqq_chaos_ell] = sub_polynomialChaosQWU_realization_chaos(nw_obs,Rww_ell,Ralpham1_scale_chaos,Rbeta_scale_chaos, ...
              %                       Ng,K0,MatPower0,MatRa0,ng,KU,MatPowerU,MatRaU,Jmax,n_y,MatRgamma_opt,Indm,Indk,Ralpha_scale_yy, ...
              %                       RQQmean,MatRVectEig1s2);
              %     MatRqq_chaos(:,ell) = Rqq_chaos_ell;
              % end
              % 
              % %--- print and plot
              % sub_polynomialChaosQWU_print_plot(nq_obs,nar_PCE,Indq_obs,MatRqq_ar0,MatRqq_chaos,ind_display_screen,ind_print,ind_plot,3); 
    %--------------------------------------------------------------------------------------------------------------------------------------

    if ind_display_screen == 1                              
       disp('--- beginning Task15_PolynomialChaosQWU')
    end

    if ind_print == 1
       fidlisting=fopen('listing.txt','a+');
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,' ------ Task15_PolynomialChaosQWU \n ');
       fprintf(fidlisting,'      \n ');  
       fclose(fidlisting);  
    end
 
    rng(SAVERANDstartPolynomialChaosQWU);  
    TimeStartPolynomialChaosQWU = tic; 
         
    %-------------------------------------------------------------------------------------------------------------------------------------   
    %             Loading MatRxx_obs(nx_obs,n_ar)  and checking parameters and data
    %-------------------------------------------------------------------------------------------------------------------------------------
 
    %--- Checking parameters and data
    if n_x <= 0 
        error('STOP1 in sub_polynomialChaosQWU: n_x <= 0');
    end  
    if n_q <= 0 || n_w <= 0
        error('STOP2 in sub_polynomialChaosQWU: n_q <= 0 or n_w <= 0');
    end
    nxtemp = n_q + n_w;                                                 % dimension of random vector XX = (QQ,WW)
    if nxtemp ~= n_x 
        error('STOP3 in sub_polynomialChaosQWU: n_x not equal to n_q + n_w');
    end
    if n_d <= 0 
        error('STOP4 in sub_polynomialChaosQWU: n_d <= 0');
    end
    if n_ar <= 0 
        error('STOP5 in sub_polynomialChaosQWU: n_ar <= 0');
    end
    if nbMC <= 0 
        error('STOP6 in sub_polynomialChaosQWU: nbMC <= 0');
    end   
 
    if nu <= 0 || nu >= n_d
        error('STOP7 in sub_polynomialChaosQWU: nu <= 0 or nu >= n_d');
    end  
 
    [n1temp,n2temp] = size(MatRx_d);                  %  MatRx_d(n_x,n_d) 
    if n1temp ~= n_x || n2temp ~= n_d
       error('STOP8 in sub_polynomialChaosQWU: dimension error in matrix MatRx_d(n_x,n_d)');
    end
    [n1temp,n2temp] = size(MatReta_ar);               %  MatReta_ar(nu,n_ar) 
    if n1temp ~= nu || n2temp ~= n_ar
       error('STOP9 in sub_polynomialChaosQWU: dimension error in matrix MatReta_ar(nu,n_ar)');
    end
    [n1temp,n2temp] = size(RmuPCA);                   %  RmuPCA(nu,1) 
    if n1temp ~= nu || n2temp ~= 1
       error('STOP10 in sub_polynomialChaosQWU: dimension error in matrix RmuPCA(nu,1)');
    end
    [n1temp,n2temp] = size(MatRVectPCA);                   %  MatRVectPCA(n_x,nu)
    if n1temp ~= n_x || n2temp ~= nu
       error('STOP11 in sub_polynomialChaosQWU: dimension error in matrix MatRVectPCA(n_x,nu)');
    end
 
    nbreal = size(Indx_real,1);                           % Indx_real(nbreal,1) 
    if nbreal >= 1
       [n1temp,n2temp] = size(Indx_real);                  
       if n1temp ~= nbreal || n2temp ~= 1
          error('STOP12 in sub_polynomialChaosQWU: dimension error in matrix Indx_real(nbreal,1)');
       end
    end
 
    nbpos = size(Indx_pos,1);                             % Indx_pos(nbpos,1)
    if nbpos >= 1
       [n1temp,n2temp] = size(Indx_pos);                  
       if n1temp ~= nbpos || n2temp ~= 1
          error('STOP13 in sub_polynomialChaosQWU: dimension error in matrix Indx_pos(nbpos,1)');
       end
    end
 
    nxtemp = nbreal + nbpos;
    if nxtemp ~= n_x 
        error('STOP14 in sub_polynomialChaosQWU: n_x not equal to nreal + nbpos');
    end
 
    % Loading dimension nq_obs of Indq_obs(nq_obs,1)
    nq_obs = size(Indq_obs,1);     %  Indq_obs(nq_obs,1)
    
    % Checking input data and parameters of Indq_obs(nq_obs,1) 
    if nq_obs < 1 || nq_obs > n_q
       error('STOP15 in sub_polynomialChaosQWU: nq_obs < 1 or nq_obs > n_q');
    end
    [n1temp,n2temp] = size(Indq_obs);                      % Indq_obs(nq_obs,1)
    if n1temp ~= nq_obs || n2temp ~= 1
       error('STOP16 in sub_polynomialChaosQWU: dimension error in matrix Indq_obs(nq_obs,1)');
    end   
    if length(Indq_obs) ~= length(unique(Indq_obs))
       error('STOP17 in sub_polynomialChaosQWU: there are repetitions in Indq_obs');  
    end
    if any(Indq_obs < 1) || any(Indq_obs > n_q)
       error('STOP18 in sub_polynomialChaosQWU: at least one integer in Indq_obs is not within the valid range');
    end
 
    % Loading dimension nw_obs of Indw_obs(nw_obs,1)
    nw_obs = size(Indw_obs,1);     %  Indw_obs(nw_obs,1)  
 
    % Checking input data and parameters of Indw_obs(nw_obs,1)
    if nw_obs < 1 || nw_obs > n_w
       error('STOP19 in sub_polynomialChaosQWU: nw_obs < 1 or nw_obs > n_w');
    end
    [n1temp,n2temp] = size(Indw_obs);                      % Indw_obs(nw_obs,1)
    if n1temp ~= nw_obs || n2temp ~= 1
       error('STOP20 in sub_polynomialChaosQWU: dimension error in matrix Indw_obs(nw_obs,1)')
    end   
    if length(Indw_obs) ~= length(unique(Indw_obs))
       error('STOP21 in sub_polynomialChaosQWU: there are repetitions in Indw_obs');  
    end
    if any(Indw_obs < 1) || any(Indw_obs > n_w)
       error('STOP22 in sub_polynomialChaosQWU: at least one integer in Indw_obs is not within the valid range');
    end
 
    if nx_obs <= 0 
        error('STOP23 in sub_polynomialChaosQWU: nx_obs <= 0');
    end
    [n1temp,n2temp] = size(Indx_obs);                      % Indx_obs(nx_obs,1)                
    if n1temp ~= nx_obs || n2temp ~= 1
       error('STOP24 in sub_polynomialChaosQWU: dimension error in matrix Indx_obs(nx_obs,1)');
    end
    if ind_scaling ~= 0 && ind_scaling ~= 1
       error('STOP25 in sub_polynomialChaosQWU: ind_scaling must be equal to 0 or to 1');
    end
    if nbreal >= 1 
       [n1temp,n2temp] = size(Rbeta_scale_real);                   % Rbeta_scale_real(nbreal,1)              
       if n1temp ~= nbreal || n2temp ~= 1
          error('STOP26 in sub_polynomialChaosQWU: dimension error in matrix Rbeta_scale_real(nbreal,1) ');
       end
       [n1temp,n2temp] = size(Ralpha_scale_real);                   % Ralpha_scale_real(nbreal,1)              
       if n1temp ~= nbreal || n2temp ~= 1
          error('STOP27 in sub_polynomialChaosQWU: dimension error in matrix Ralpha_scale_real(nbreal,1) ');
       end                    
    end
    if nbpos >= 1 
       [n1temp,n2temp] = size(Rbeta_scale_log);                     % Rbeta_scale_log(nbpos,1)              
       if n1temp ~= nbpos || n2temp ~= 1
          error('STOP28 in sub_polynomialChaosQWU: dimension error in matrix Rbeta_scale_log(nbpos,1) ');
       end
       [n1temp,n2temp] = size(Ralpha_scale_log);                    % Ralpha_scale_log(nbpos,1)              
       if n1temp ~= nbpos || n2temp ~= 1
          error('STOP29 in sub_polynomialChaosQWU: dimension error in matrix Ralpha_scale_log(nbpos,1) ');
       end                    
    end
 
    % Number of learned realizations used for the PCE expansion
    if nbMC_PCE <= 0  || nbMC_PCE > nbMC
        error('STOP30 in sub_polynomialChaosQWU: nbMC_PCE <= 0  or  nbMC_PCE > nbMC');
    end
    nar_PCE = n_d*nbMC_PCE;               
 
    %--- PCA back: MatRx_obs(nx_obs,n_ar)
    [MatRx_obs] = sub_polynomialChaosQWU_PCAback(n_x,n_d,nu,n_ar,nx_obs,MatRx_d,MatReta_ar,Indx_obs,RmuPCA,MatRVectPCA, ...
                              ind_display_screen,ind_print);
    
    %--- Scaling back: MatRxx_obs(nx_obs,n_ar)
    [MatRxx_obs] = sub_polynomialChaosQWU_scalingBack(nx_obs,n_x,n_ar,MatRx_obs,Indx_real,Indx_pos,Indx_obs,Rbeta_scale_real,Ralpha_scale_real, ...
                                   Rbeta_scale_log,Ralpha_scale_log,ind_display_screen,ind_print,ind_scaling); 
    clear MatRx_obs

    %-------------------------------------------------------------------------------------------------------------------------------------   
    %             Loading MatRqq_ar0(nq_obs,nar_PCE) and MatRww_ar0(nw_obs,nar_PCE)
    %-------------------------------------------------------------------------------------------------------------------------------------
  
    % Loading MatRqq_ar0(nq_obs,nar_PCE) and MatRww_ar0(nw_obs,nar_PCE) from MatRxx_obs(nx_obs,n_ar) 
    MatRqq_ar0 = MatRxx_obs(1:nq_obs,1:nar_PCE);               % MatRqq_ar0(nq_obs,nar_PCE),MatRxx_obs(nx_obs,n_ar) 
    MatRww_ar0 = MatRxx_obs(nq_obs+1:nq_obs+nw_obs,1:nar_PCE); % MatRww_ar0(nw_obs,nar_PCE),MatRxx_obs(nx_obs,n_ar) 
    clear MatRxx_obs 
 
    %-------------------------------------------------------------------------------------------------------------------------------------   
    %             Checking information related to the PCE and checking the control parameters
    %-------------------------------------------------------------------------------------------------------------------------------------
 
    %     Ng                          : dimension of the germ for polynomial chaos Psi_alpha that is such that Ng = n_w   
    %     Ndeg                        : max degree of the polynomial chaos Psi_alpha with Ndeg >= 1  
    %     ng                          : dimension of the germ for polynomial chaos phi_a that is such that ng >= 1  
    %     ndeg                        : max degree of the polynomial chaos phi_a with ndeg >= 0 (if ndeg = 0, then KU = 1)  
    
    if Ng ~= nw_obs
       error('STOP31 in sub_polynomialChaosQWU: Ng must be equal to nw_obs');
    end    
    if Ndeg <= 0
       error('STOP32 in sub_polynomialChaosQWU: Ndeg must be greater than or equal to 1');
    end    
    if ng <= 0
       error('STOP33 in sub_polynomialChaosQWU: ng must be greater than or equal to 1');
    end 
    if ndeg <= -1
       error('STOP34 in sub_polynomialChaosQWU: ndeg must be greater than or equal to 0');
    end 
    if MaxIter <= 0
       error('STOP35 in sub_polynomialChaosQWU: MaxIter must be greater than or equal to 1');
    end
 
    if ind_display_screen ~= 0 && ind_display_screen ~= 1       
          error('STOP36 in sub_polynomialChaosQWU: ind_display_screen must be equal to 0 or equal to 1')
    end
    if ind_print ~= 0 && ind_print ~= 1       
          error('STOP37 in sub_polynomialChaosQWU: ind_print must be equal to 0 or equal to 1')
    end
    if ind_plot ~= 0 && ind_plot ~= 1       
          error('STOP38 in sub_polynomialChaosQWU: ind_plot must be equal to 0 or equal to 1')
    end
    if ind_parallel ~= 0 && ind_parallel ~= 1       
          error('STOP39 in sub_polynomialChaosQWU: ind_parallel must be equal to 0 or equal to 1')
    end
 
    %--- Computation of K0 and KU and checking the consistency
    K0 = fix(1e-12 + factorial(Ng+Ndeg)/(factorial(Ng)*factorial(Ndeg))); 
    KU = fix(1e-12 + factorial(ng+ndeg)/(factorial(ng)*factorial(ndeg))); 
 
    if K0 >= nar_PCE
       disp(['K0 = ',num2str(K0)]);
       disp(['nar_PCE = ',num2str(nar_PCE)]);
       error('STOP40 in sub_polynomialChaosQWU: K0 must be less than nar_PCE')
    end  
 
    if KU >= nar_PCE
       disp(['KU = ',num2str(KU)]);
       disp(['nar_PCE = ',num2str(nar_PCE)]);
       error('STOP41 in sub_polynomialChaosQWU: KU must be less than nar_PCE')
    end  
     
    %--- Components of QQ_obs that are selected for the cost function. In this code version all the components are kept
    Ind_qqC   = (1:1:nq_obs)';   % Ind_qqC(nbqqC,1): components of QQ_obs that are selected for the cost function
    nbqqC     = nq_obs; 
    
    %--- display
    if ind_display_screen == 1
       disp('    [Ng   Ndeg  K0  ]')
       disp([Ng Ndeg K0])
     
       disp('    [ng   ndeg  KU  ]')
       disp([ng ndeg KU])
    end 
    
    %--- print  
    if ind_print == 1
       fidlisting=fopen('listing.txt','a+'); 
       fprintf(fidlisting,'                                \n ');      
       fprintf(fidlisting,' n_q           = %9i \n ',n_q);   
       fprintf(fidlisting,' n_w           = %9i \n ',n_w);
       fprintf(fidlisting,' n_x           = %9i \n ',n_x);   
       fprintf(fidlisting,' nbreal        = %9i \n ',nbreal);
       fprintf(fidlisting,' nbpos         = %9i \n ',nbpos);
       fprintf(fidlisting,' nx_obs        = %9i \n ',nx_obs);
       fprintf(fidlisting,' ind_scaling   = %9i \n ',ind_scaling);    
       fprintf(fidlisting,'                                \n '); 
       fprintf(fidlisting,' n_d           = %9i \n ',n_d);
       fprintf(fidlisting,' nbMC          = %9i \n ',nbMC);
       fprintf(fidlisting,' n_ar          = %9i \n ',n_ar);    
       fprintf(fidlisting,' nu            = %9i \n ',nu);
       fprintf(fidlisting,'                                \n '); 
       fprintf(fidlisting,' nbMC_PCE      = %9i \n ',nbMC_PCE);
       fprintf(fidlisting,' nar_PCE       = %9i \n ',nar_PCE);
       fprintf(fidlisting,'                                \n '); 
       fprintf(fidlisting,' Ng            = %9i \n ',Ng);
       fprintf(fidlisting,' Ndeg          = %9i \n ',Ndeg);
       fprintf(fidlisting,' K0            = %9i \n ',K0);
       fprintf(fidlisting,'                                \n '); 
       fprintf(fidlisting,' ng            = %9i \n ',ng);
       fprintf(fidlisting,' ndeg          = %9i \n ',ndeg);
       fprintf(fidlisting,' KU            = %9i \n ',KU);
       fprintf(fidlisting,'                                \n '); 
       fprintf(fidlisting,' MaxIter       = %9i \n ',MaxIter);
       fprintf(fidlisting,'                                \n '); 
       fprintf(fidlisting,' ind_display_screen = %1i \n ',ind_display_screen); 
       fprintf(fidlisting,' ind_print          = %1i \n ',ind_print); 
       fprintf(fidlisting,' ind_plot           = %1i \n ',ind_plot); 
       fprintf(fidlisting,' ind_parallel       = %1i \n ',ind_parallel); 
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,'      \n '); 
       fclose(fidlisting); 
    end
             

    %--- construction of MatRxipointOVL(nq_obs,nbpoint) of the points used by ksdensity in OVL
    nbpoint = 200;
    MatRxipointOVL = zeros(nq_obs,nbpoint);
    for k = 1:nq_obs
        maxqk               = max(MatRqq_ar0(k,:));          % MatRqq_ar0(nq_obs,nar_PCE)
        minqk               = min(MatRqq_ar0(k,:));
        MatRxipointOVL(k,:) = linspace(minqk,maxqk,nbpoint); % QQ_k
    end
    MatRxipointOVLC =  MatRxipointOVL(Ind_qqC,:);

    %--- construction of RbwOVL(nq_obs,1) for the bandwidth used by ksdensity in OVL
    Rbw    = zeros(nq_obs,1);                                % Rbw(nq_obs,1);
    RbwOVL = zeros(nq_obs,1);
    for k = 1:nq_obs 
        [~,~,bwk]   = ksdensity(MatRqq_ar0(k,:));            % MatRqq_ar0(nq_obs,nar_PCE)
        Rbw(k,1)    = bwk;
        RbwOVL(k,1) = 1.*bwk;
    end
    RbwOVLC = RbwOVL(Ind_qqC,:);
   
    %--- computing second-order moment using the learned dataset for QQ_ar0
    %    MatRmom2QQ_ar0 = (MatRqq_ar0*(MatRqq_ar0'))/(nar_PCE-1);        % MatRmom2QQ_ar0(nq_obs,nq_obs)

    %--- Construction of MatRxiNg0(Ng,nar_PCE) (we have Ng = nw_obs) by scaling MatRww_ar0(Ng,nar_PCE) between cmin and cmax 
    cmax = 1.0; 
    cmin = -1.0;   
    Rmax = max(MatRww_ar0,[],2);                            %    MatRww_ar0(Ng,nar_PCE), Ng = nw_obs               
    Rmin = min(MatRww_ar0,[],2);
    Rbeta_scale_chaos    = zeros(Ng,1);                     %    Rbeta_scale_chaos(Ng,1);
    Ralpha_scale_chaos   = zeros(Ng,1);                     %    Ralpha_scale_chaos(Ng,1);
    Ralpham1_scale_chaos = zeros(Ng,1);                     %    Ralpham1_scale_chaos(Ng,1);
    for k = 1:Ng           
        A = (Rmax(k)-Rmin(k))/(cmax-cmin);
        B = Rmin(k) - A*cmin;
        Rbeta_scale_chaos(k)    = B;
        Ralpha_scale_chaos(k)   = A;
        Ralpham1_scale_chaos(k) = 1/Ralpha_scale_chaos(k);
    end 
    clear cmax cmin Rmax Rmin                                                            
    MatRxiNg0  = Ralpham1_scale_chaos.*(MatRww_ar0 - Rbeta_scale_chaos);  % MatRxiNg0(Ng,nar_PCE),MatRww_ar0(Ng,nar_PCE), Ng = nw_obs   
        
    %--- Normalization of QQ_ar0 using a PCA: MatRqq_ar0  = repmat(RQQmean,1,nar_PCE) + MatRVectEig1s2*MatRyy_ar0  
    RQQmean            = mean(MatRqq_ar0,2);                    % RQQmean(nq_obs,1)
    MatRQQcov          = cov(MatRqq_ar0');                      % MatRQQcov(nq_obs,nq_obs),MatRqq_ar0(nq_obs,nar_PCE)
    MatRQQcov          = 0.5*(MatRQQcov + (MatRQQcov'));     
    [MatRVect,MatREig] = eig(MatRQQcov);                        % MatRVect(nq_obs,nq_obs),MatREig(nq_obs,nq_obs) 
    MatRVectEig1s2     = MatRVect*diag(sqrt(diag(MatREig)));    % MatRVectEig1s2(nq_obs,nq_obs)  

    n_y = nq_obs;                                               % no reduction applied

    MatRyy_ar0         = diag((sqrt(1./diag(MatREig))))*(MatRVect')*(MatRqq_ar0 - RQQmean); % MatRyy_ar0(n_y,nar_PCE)
    clear MatRQQcov MatRVect MatReig

    %--- scaling MatRyy_ar0 in MatRy_ar0  in order that  the maximum of Y on the realizations be normalized to 1
    Rmax              = max(abs(MatRyy_ar0),[],2);                        % Rmax(n_y,1)
    Ralpha_scale_yy   = Rmax;                                             % Ralpha_scale_yy(n_y,1)
    Ralpham1_scale_yy = 1./Ralpha_scale_yy;                               % Ralpham1_scale_yy(n_y,1); 
    MatRy_ar0         = Ralpham1_scale_yy.*MatRyy_ar0;                    % MatRy_ar0(n_y,nar_PCE)
    
    %--- Computing matrices: MatRPsi0(K0,nar_PCE),MatPower0(K0,Ng),MatRa0(K0,K0)
    %    of the polynomial chaos Psi_{alpha^(k)}(Xi) with alpha^(k) = (alpha_1^(k),...,alpha_Ng^(k)) in R^Ng with k=1,...,K0
    %    MatRxiNg0(Ng,nar_PCE) 
    [MatRPsi0,MatPower0,MatRa0] = sub_polynomialChaosQWU_chaos0(K0,Ndeg,Ng,nar_PCE,MatRxiNg0);
    clear MatRxiNg0

    %--- Computing matrices: MatRphiU(KU,nar_PCE),MatPowerU(KU,Ng),MatRaU(KU,KU)
    %    of the polynomial chaos phi_{a^(m)}(U) with a^(m) = (a_1^(m),...,a_ng^(m)) in R^ng with m=1,...,KU    
    MatRU = randn(ng,nar_PCE);                                                                % MatRU(ng,nar_PCE)
    [MatRphiU,MatPowerU,MatRaU] = sub_polynomialChaosQWU_chaosU(KU,ndeg,ng,nar_PCE,MatRU);    % MatRphiU(KU,nar_PCE)
    clear MatRU

    %--- Computing the global index j = (m,k), m = 1,...,KU and k = 1,...,K0, 
    %                               j = 1,..., Jmax  with Jmax = KU*K0
    Jmax = KU*K0;
                                    % Indm = zeros(Jmax,1);    % m = Indm(j)
                                    % Indk = zeros(Jmax,1);    % k = Indk(j)
                                    % j = 0;
                                    % for m = 1:KU
                                    %     for k = 1:K0
                                    %         j = j+1;
                                    %         Indm(j) = m;
                                    %         Indk(j) = k;
                                    %     end
                                    % end
    Indm = repelem((1:KU)', K0);
    Indk = repmat((1:K0)', KU, 1);

    if n_y > Jmax
       error('STOP42 in sub_polynomialChaosQWU: Jmax must be greater than or equal to n_y; increase Ndeg and/or ng and/or ndeg')
    end  

    %--- construction of MatRb(Jmax,nar_PCE) such that MatRb(j,ell) = MatRphiU(m,ell)*MatRPsi0(k,ell)
    MatRb = zeros(Jmax,nar_PCE);
    for j=1:Jmax
        m = Indm(j);
        k = Indk(j);
        MatRb(j,:) = MatRphiU(m,:).*MatRPsi0(k,:);               % MatRb(Jmax,nar_PCE)
    end

    %--- Computing MatRMrondY(n_y,n_y)
    MatRMrondY = MatRy_ar0*MatRy_ar0'/(nar_PCE-1);               %   MatRy_ar0(n_y,nar_PCE)

    %--- Computing an initial value MatRgamma_bar of MatRgamma for fmincon 
    MatRgamma_bar  = MatRy_ar0*MatRb'/(nar_PCE-1);               % MatRgamma_bar(n_y,Jmax)
    
    %--- Pre-computation and loading
    MatRchol    = chol(MatRMrondY);                              % MatRchol(n_y,n_y),MatRMrondY(n_y,n_y) 
    MatRones    = ones(n_y,Jmax);                                % MatRones(n_y,Jmax)
    MatRqq_ar0C = MatRqq_ar0(Ind_qqC,:);
    
    %--- computing MatRsample_opt(n_y,Jmax) by using fminunc
    MatRsample0         = zeros(n_y,Jmax);                       % MatRsample0(n_y,Jmax): initial value   
    options.Algorithm   = 'quasi-newton';           
    options.Display     = 'iter-detailed';  
    options.MaxFunEvals = 800*n_y*Jmax; % Maximum number of function evaluations allowed, a positive integer. Default is 100*numberOfVariables.
    options.MaxIter     = MaxIter;      % Maximum number of iterations allowed, a positive integer. The default is 400.
    options.TolFun      = 1e-6;         % Termination tolerance on the function value, a positive scalar. The default is 1e-6.
    options.TolX        = 1e-6;         % Termination tolerance on x, a positive scalar. The default is 1e-6.        
    % options.PlotFcns  = {@optimplotfunccount,@optimplotfval,@optimplotresnorm,@optimplotstepsize};  % plots 
    options.UseParallel = 'always'; 
    % Define the custom output function to log iterations
    options.OutputFcn = @logIterationsToFile;

    J = @(MatRsample)sub_polynomialChaosQWU_Jcost(MatRsample,n_y,Jmax,nar_PCE,nq_obs,nbqqC,nbpoint,MatRgamma_bar, ...
                                                  MatRchol,MatRb,Ralpha_scale_yy,RQQmean,MatRVectEig1s2,Ind_qqC, ...
                                                  MatRqq_ar0C,MatRxipointOVLC,RbwOVLC,MatRones);  

    MatRsample_opt = fminunc(J,MatRsample0,options); % MatRsample_opt(n_y,Jmax)
        
            
    %--- computing MatRgamma_opt(n_y,Jmax) for the polynomial chaos expansion  
    MatRtilde     = MatRgamma_bar.*(ones(n_y,Jmax) + MatRsample_opt); % MatRtilde(n_y,Jmax),MatRgamma_bar(n_y,Jmax) 
    MatRFtemp     = chol(MatRtilde*(MatRtilde'));                     % MatRFtemp(n_y,n_y)
    MatRAtemp     = (inv(MatRFtemp))';                                % MatRAtemp(n_y,n_y)
    MatRhat       = MatRAtemp*MatRtilde;                              % MatRhat(n_y,Jmax),MatRtilde(n_y,Jmax)
    MatRgamma_opt = MatRchol'*MatRhat;                                % MatRgamma(n_y,Jmax)
    clear MatRtilde MatRFtemp MatRAtemp MatRhat
   
    %-----------------------------------------------------------------------------------------------------------------------
    %                      Polynomial-chaos representation for MatRww_ar0(nw_obs,nar_PCE) 
    %-----------------------------------------------------------------------------------------------------------------------
    
    seeds_temp = randi(2^32,1,1);     % Random initialization of the generator with the seed 
    rng(seeds_temp);  
    MatRU = randn(ng,nar_PCE);                                                                    % MatRU(ng,nar_PCE)
    [MatRy_PolChaos_ar0] = sub_polynomialChaosQWU_surrogate(nar_PCE,nw_obs,n_y,MatRww_ar0,MatRU,Ralpham1_scale_chaos, ...
                                                            Rbeta_scale_chaos,Ng,K0,MatPower0,MatRa0,ng,KU,MatPowerU,MatRaU,Jmax, ...
                                                            MatRgamma_opt,Indm,Indk);
    MatRyy_PolChaos_ar0     = Ralpha_scale_yy.*MatRy_PolChaos_ar0;                                % MatRy_PolChaos_ar0(n_y,nar_PCE)   
    MatRqq_PolChaos_ar0     = repmat(RQQmean,1,nar_PCE) + MatRVectEig1s2*MatRyy_PolChaos_ar0;     % MatRqq_PolChaos_ar0(nq_obs,nar_PCE)
  % MatRmom2QQ_PolChaos_ar0 = MatRqq_PolChaos_ar0*(MatRqq_PolChaos_ar0')/(nar_PCE - 1);           % MatRmom2QQ_PolChaos_ar0(nq_obs,nq_obs) 
    [RerrorOVL]             = sub_polynomialChaosQWU_OVL(nbqqC,nar_PCE,MatRqq_ar0(Ind_qqC,:),nar_PCE,MatRqq_PolChaos_ar0(Ind_qqC,:), ...
                                                         nbpoint,MatRxipointOVL(Ind_qqC,:),RbwOVL(Ind_qqC,1));
    errorOVL_PolChaos_ar0C  = sum(RerrorOVL)/nbqqC;  

    %--- display
    if ind_display_screen == 1
       disp(['errorOVL_PolChaos_ar0C = ',num2str(errorOVL_PolChaos_ar0C)]);
    end

    %--- print
    if ind_print == 1
       fidlisting=fopen('listing.txt','a+');
       fprintf(fidlisting,'      \n ');  
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,' errorOVL_PolChaos_ar0C =  =  %14.7e \n',errorOVL_PolChaos_ar0C);   
       fprintf(fidlisting,'      \n ');                                                                
       fprintf(fidlisting,'      \n '); 
       fclose(fidlisting);   
    end

    %--- debug sequence
    ind_plot_debug = 0;
    if ind_plot_debug == 1
       hold off      
       for i=1:nq_obs
           h = figure; 
           plot((1:1:nar_PCE),MatRqq_PolChaos_ar0(i,:),'-')     % MatRqq_PolChaos_ar0(nq_obs,nar_PCE)                                            
           title(['$\ell\mapsto qq_{',num2str(i),',{\rm{chaos,ar0s}}}^\ell$'],'FontSize',16,'Interpreter','latex','FontWeight','normal')                                         
           xlabel('$\ell$','FontSize',16,'Interpreter','latex')                                                                
           ylabel(['$qq_{',num2str(i),',{\rm{chaos,ar0s}}}^\ell$'],'FontSize',16,'Interpreter','latex')
           saveas(h,['figure_sub_polynomialChaosQWU_qq_',num2str(i),'_chaos_ar0.fig']);       
           close(h);  
       end
    end

    %--- print and plot
    sub_polynomialChaosQWU_print_plot(nq_obs,nar_PCE,Indq_obs,MatRqq_ar0,MatRqq_PolChaos_ar0,ind_display_screen,ind_print,ind_plot,1);  

    %-----------------------------------------------------------------------------------------------------------------------
    %                          Polynomial-chaos validation for MatRww_o(nw_obs,nar_PCE)
    %-----------------------------------------------------------------------------------------------------------------------

    MatRww_o = MatRww_ar0;                                   % MatRww_o(nw_obs,nar_PCE),MatRww_ar0(nw_obs,nar_PCE)

    seeds_temp = randi(2^32,1,1);                            % Random initialization of the generator with the seed 
    rng(seeds_temp);  

    MatRqq_PolChaos_o = zeros(nq_obs,nar_PCE);               % MatRqq_PolChaos_o(nq_obs,nar_PCE);
    for kno = 1:nar_PCE                                      %... loop on the control variable
        Rww_kno = MatRww_o(:,kno); 
        RU_kno  = randn(ng,1);
        [Ry_PolChaos_o] = sub_polynomialChaosQWU_surrogate(1,nw_obs,n_y,Rww_kno,RU_kno,Ralpham1_scale_chaos,Rbeta_scale_chaos, ...
                                                           Ng,K0,MatPower0,MatRa0,ng,KU,MatPowerU,MatRaU,Jmax,MatRgamma_opt,Indm,Indk);
        Ryy_PolChaos_o           = Ralpha_scale_yy.*Ry_PolChaos_o;            % Ryy_PolChaos_o(n_y,1)   
        Rqq_PolChaos_o           = RQQmean + MatRVectEig1s2*Ryy_PolChaos_o;   % Rqq_PolChaos_o(nq_obs,1)
        MatRqq_PolChaos_o(:,kno) = Rqq_PolChaos_o;                            % MatRqq_PolChaos_o(nq_obs,nar_PCE), Rqq_PolChaos_o(nq_obs,1)
    end  

    %--- print and plot
    sub_polynomialChaosQWU_print_plot(nq_obs,nar_PCE,Indq_obs,MatRqq_ar0,MatRqq_PolChaos_ar0,ind_display_screen,ind_print,ind_plot,2); 

    %--- save the file.mat:  "filePolynomialChaosQWU_for_realization"

    % Define the file name
    fileName = 'filePolynomialChaosQWU_for_realization.mat';

    % Save on filename.mat file
    save(fileName,'nw_obs','nq_obs','Indw_obs','Indq_obs','nar_PCE','MatRww_ar0','MatRqq_ar0','Ralpham1_scale_chaos', ...
                  'Rbeta_scale_chaos','Ng','K0','MatPower0','MatRa0','ng','KU','MatPowerU','MatRaU','Jmax','n_y', ...
                  'MatRgamma_opt','Indm','Indk','Ralpha_scale_yy','RQQmean','MatRVectEig1s2','-v7.3');
    fprintf('The file "%s" has been saved.\n', fileName);
    
    %----------------------------------------------------------------------------------------------------------------------------------
    %                                                           end
    %----------------------------------------------------------------------------------------------------------------------------------

    SAVERANDendPolynomialChaosQWU = rng;  
    ElapsedPolynomialChaosQWU     = toc(TimeStartPolynomialChaosQWU);  

    if ind_print == 1
       fidlisting=fopen('listing.txt','a+');
       fprintf(fidlisting,'      \n ');                                                                
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,'-------   Elapsed time for Task15_PolynomialChaosQWU \n ');
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,'Elapsed Time   =  %10.2f\n',ElapsedPolynomialChaosQWU);   
       fprintf(fidlisting,'      \n '); 
       fclose(fidlisting);  
    end
    if ind_display_screen == 1   
       disp('--- end Task15_PolynomialChaosQWU')
    end    
    return
 end
 
 %----------------------------------------------------------------------------------------------------------------------------------
 %                                function for printing the iterations from fminunc
 %----------------------------------------------------------------------------------------------------------------------------------

 function stop = logIterationsToFile(~, optimValues, state)
    stop = false;
    fidlisting = fopen('listing.txt', 'a+');
    if fidlisting == -1
        error('Cannot open file for writing.');
    end
    
    switch state
        case 'init'
            fprintf(fidlisting, 'Iteration\tFunc-count\tf(x)\t\tStep-size\tFirst-order optimality\n');
        case 'iter'
            fprintf(fidlisting, '%d\t\t%d\t\t%.6g\t%.6g\t%.6g\n', ...
                optimValues.iteration, optimValues.funccount, optimValues.fval, ...
                optimValues.stepsize, optimValues.firstorderopt);
        case 'done'
            fprintf(fidlisting, 'Optimization completed.\n');
        otherwise
            % do nothing
    end
    
    fclose(fidlisting);
end









         