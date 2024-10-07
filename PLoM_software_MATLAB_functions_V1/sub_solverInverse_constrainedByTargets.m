function  [MatReta_ar,ArrayZ_ar] = sub_solverInverse_constrainedByTargets(nu,n_d,nbMC,n_ar,nbmDMAP,M0transient,Deltar,f0,shss,sh, ...
                                       MatReta_d,MatRg,MatRa,ArrayWiennerM0transient,ArrayGauss,ind_type_targ,N_r,Rb_targ1,coNr, ...
                                       coNr2,MatReta_targ,eps_inv,Rb_targ2,Rb_targ3,ind_coupling,epsc,iter_limit,Ralpha_relax, ...
                                       ind_display_screen,ind_print,ind_parallel,numfig) 

   %------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 08 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverInverse_constrainedByTargets
   %  Subject      : solver for ind_type_targ = 1, 2, or 3: Constraints on H^c
   %
   %--- INPUT  
   %      nu                  : dimension of random vector H = (H_1, ... H_nu)
   %      n_d                 : number of points in the training set for H
   %      nbMC                : number of realizations of (nu,n_d)-valued random matrix [H_ar] 
   %      n_ar                : number of realizations of H_ar such that n_ar  = nbMC x n_d
   %      nbmDMAP             : dimension of the ISDE-projection basis
   %      M0transient         : number of steps for reaching the stationary solution
   %      Deltar              : ISDE integration step by Verlet scheme
   %      f0                  : dissipation coefficient in the ISDE   
   %      shss,sh             : parameters of the GKDE of pdf of H_d (training) 
   %      MatReta_d(nu,n_d)   : n_d realizations of H_d (training)    
   %      MatRg(n_d,nbmDMAP)  : matrix of the ISDE-projection basis
   %      MatRa(n_d,nbmDMAP)  : related to MatRg(n_d,nbmDMAP) 
   %      ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)  : realizations of the matrix-valued normalized Wienner process
   %      ArrayGauss(nu,n_d,nbMC)                           : realizations of the Gaussian matrix-valued random variable
   %
   %      ind_type_targ       : = 1, targets defined by giving N_r realizations
   %                          : = 2, targets defined by giving target mean-values 
   %                          : = 3, targets defined by giving target mean-values and target covariance matrix
   %                          --- ind_type_targ = 1: targets defined by giving N_r realizations of XX_targ 
   %      N_r                     nummber of realizations of the targets     
   %      Rb_targ1(N_r,1)         E{h_targ1(H^c)} = b_targ1  with h_targ1 = (h_{targ1,1}, ... , h_{targ1,N_r})
   %      coNr                    parameter used for evaluating  E{h^c_targ(H^c)}               
   %      coNr2                   parameter used for evaluating  E{h^c_targ(H^c)} 
   %      MatReta_targ(nu,N_r)    N_r realizations of the projection of XX_targ on the model
   %      eps_inv                 : tolerance for computing the pseudo-inverse of matrix MatRGammaS_iter with 
   %                                sub_solverInverse_pseudo_inverse(MatRGammaS_iter,eps_inv) for ind_type_targ = 1,
   %                                in sub_solverInverse_constrainedByTargets. An adapted value is 0.001. If problems occurs
   %                                increase the value to 0.01.
   %                          --- ind_type_targ = 2 or 3: targets defined by giving mean value of XX_targ
   %      Rb_targ2(nu,1)          yielding the constraint E{H^c} = b_targ2 
   %                          --- ind_type_targ = 3: targets defined by giving target covariance matrix of XX_targ
   %      Rb_targ3(nu,1)          yielding the constraint diag(E{H_c H_c'}) = b_targ3  
   %
   %      ind_coupling        : 0 (no coupling). = 1, used if ind_type_targ = 3 and consisting by keeping the extradiagobal blocks 
   %                            in matrix MatRGammaS_iter(2*nu,2*nu) related to [H^c ; diag(H_c H_c')] for the computation
   %                            Lagrange mutipliers by using the iteration algorithm
   %      epsc                : relative tolerance for the iteration convergence of the constraints on H 
   %      iter_limit          : maximum number of iteration for computing the Lagrange multipliers
   %      Ralpha_relax        : Ralpha_relax(iter_limit,1) relaxation parameter in ] 0 , 1 ] for the iterations. 
   %      ind_display_screen  : = 0 no display,              = 1 display
   %      ind_print           : = 0 no print,                = 1 print
   %      ind_parallel        : = 0 no parallel computation, = 1 parallel computation
   %
   %--- OUTPUT
   %      MatReta_ar(nu,n_ar)
   %      ArrayZ_ar(nu,nbmDMAP,nbMC);    % ArrayZ_ar(nu,nbmDMAP,nbMC), this array is used as ouput for possible use in a postprocessing 
   %                                     % of Z in order to construct its polynomial chaos expansion (PCE)
 
   %--- Initialization and preallocation  
   if ind_type_targ == 1                     %--- Constraints E{h^c_r(H^c)} = Rb_targ1(r,1) for r =1,...,N_r 
      mhc = N_r;
      Rbc = Rb_targ1;                        % Rbc(mhc,1) = Rb_targ1(N_r,1)
      Rlambda_iter_m1 = zeros(mhc,1);        % Rlambda_iter_m1(mhc,1) = (lambda_1,...,lambda_N_r)
   end
   if ind_type_targ == 2                     %--- Constraints E{H^c_j} = Rb_targ2(j,1) for j =1,...,nu 
      mhc = nu;
      Rbc = Rb_targ2;                        % Rbc(mhc,1) = Rb_targ2(nu,1)
      Rlambda_iter_m1 = zeros(mhc,1);        % Rlambda_iter_m1(mhc,1) = (lambda_1,...,lambda_nu)
   end
   if ind_type_targ  == 3                    %--- constraints E{H^c_j} = Rb_targ2(j,1) and E{H^c_j^2} = Rb_targ3(j,1) for j =1,...,nu 
      mhc              = 2*nu;               % Rbc(mhc,1) = [Rb_targ2 ; Rb_targ3]
      Rbc(1:nu,1)      = Rb_targ2;           % Rb_targ2(nu,1)
      Rbc(1+nu:2*nu,1) = Rb_targ3;           % Rb_targ3(nu,1)
      Rlambda_iter_m1  = zeros(mhc,1);       % Rlambda_iter_m1(mhc,1) = (lambda_1,...,lambda_nu,lambda_{1+nu},...,lambda_{nu+nu})
   end
   
   %--- Print parameters
   if ind_print == 1
      fidlisting=fopen('listing.txt','a+'); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'--- Parameters for solver Inverse constrained by targets    \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'    ind_type_targ = %7i \n ',ind_type_targ); 
      fprintf(fidlisting,'    nu            = %7i \n ',nu); 
      fprintf(fidlisting,'    mhc           = %7i \n ',mhc); 
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end
          
   %--- Preallocation for the Lagrange-multipliers computation by the iteration algorithm and generation  of nbMC realizations                                                        
   Rerr               = zeros(iter_limit,1);   % Rerr(iter_limit,1)      
   RnormRlambda       = zeros(iter_limit,1);   % RnormRlambda(iter_limit,1)
   RcondGammaS        = zeros(iter_limit,1);   % RcondGammaS(iter_limit,1)
   Rtol               = zeros(iter_limit,1);   % Rtol(iter_limit,1) 
   Rtol(1)            = 1;
   ind_conv           = 0;
   
   %--- Loop of the iteration algorithm (iter: lambda_iter given and compute lambda_iterplus1

   for iter = 1:iter_limit   

       if ind_display_screen == 1
          disp(['------------- iter number: ', num2str(iter)]);
       end  

       % Constraints E{h^c_r(H^c)} = Rbc(r,1) for r =1,...,N_r 
       if ind_type_targ == 1 
          ArrayZ_ar_iter = zeros(nu,nbmDMAP,nbMC);      % ArrayZ_ar_iter(nu,nbmDMAP,nbMC)

          % Vectorized sequence
          if ind_parallel == 0
             for ell = 1:nbMC 
                MatRGauss_ell               = ArrayGauss(:,:,ell);                   % ArrayGauss(nu,n_d,nbMC)
                ArrayWiennerM0transient_ell = ArrayWiennerM0transient(:,:,:,ell);    % ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                [MatRZ_ar_iter] = sub_solverInverse_Verlet_target1(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh, ...
                                           ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1,coNr,coNr2,MatReta_targ);                                                   
                ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(nu,nbmDMAP,nbMC)                                
             end
          end

          % Parallel sequence
          if ind_parallel == 1
             parfor ell = 1:nbMC 
                MatRGauss_ell               = ArrayGauss(:,:,ell);                   % ArrayGauss(nu,n_d,nbMC)
                ArrayWiennerM0transient_ell = ArrayWiennerM0transient(:,:,:,ell);    % ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                [MatRZ_ar_iter] = sub_solverInverse_Verlet_target1(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh, ...
                                           ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1,coNr,coNr2,MatReta_targ); 
                ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(nu,nbmDMAP,nbMC)                                
             end
          end
          clear MatRZ_ar_iter MatRGauss_ell ArrayWiennerM0transient_ell 
       end

       % Constraints E{H^c_j} = Rb_targ2(j,1) for j =1,...,nu 
       if ind_type_targ == 2                           
          ArrayZ_ar_iter = zeros(nu,nbmDMAP,nbMC);      % ArrayZ_ar_iter(nu,nbmDMAP,nbMC)

          % Vectorized sequence
          if ind_parallel == 0
             for ell = 1:nbMC 
                MatRGauss_ell               = ArrayGauss(:,:,ell);                   % ArrayGauss(nu,n_d,nbMC)
                ArrayWiennerM0transient_ell = ArrayWiennerM0transient(:,:,:,ell);    % ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                [MatRZ_ar_iter] = sub_solverInverse_Verlet_target2(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh, ...
                                                                   ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1);
                ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(nu,nbmDMAP,nbMC)                                
             end
          end

          % Parallel sequence
          if ind_parallel == 1
             parfor ell = 1:nbMC 
                MatRGauss_ell               = ArrayGauss(:,:,ell);                   % ArrayGauss(nu,n_d,nbMC)
                ArrayWiennerM0transient_ell = ArrayWiennerM0transient(:,:,:,ell);    % ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)
   
                [MatRZ_ar_iter] = sub_solverInverse_Verlet_target2(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh, ...
                                                                   ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1);                                                   
   
                ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(nu,nbmDMAP,nbMC)                                
             end
          end
          clear MatRZ_ar_iter MatRGauss_ell ArrayWiennerM0transient_ell 
       end
       
       % Constraints E{H^c_j} = Rb_targ2(j,1) and E{H^c_j^2} = Rb_targ3(j,1) for j =1,...,nu
       if ind_type_targ == 3                          
          ArrayZ_ar_iter = zeros(nu,nbmDMAP,nbMC);                                   % ArrayZ_ar_iter(nu,nbmDMAP,nbMC)

          % Vectorized sequence
          if ind_parallel == 0
             for ell = 1:nbMC 
                 MatRGauss_ell               = ArrayGauss(:,:,ell);                   % ArrayGauss(nu,n_d,nbMC)
                 ArrayWiennerM0transient_ell = ArrayWiennerM0transient(:,:,:,ell);    % ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                 [MatRZ_ar_iter] = sub_solverInverse_Verlet_target3(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh, ...
                                                                   ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1);                                                 
                 ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(nu,nbmDMAP,nbMC)                                
             end
          end

          % Parallel sequence
          if ind_parallel == 1
             parfor ell = 1:nbMC 
                 MatRGauss_ell               = ArrayGauss(:,:,ell);                   % ArrayGauss(nu,n_d,nbMC)
                 ArrayWiennerM0transient_ell = ArrayWiennerM0transient(:,:,:,ell);    % ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                 [MatRZ_ar_iter] = sub_solverInverse_Verlet_target3(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh, ...
                                                                   ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1);                                                
                 ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(nu,nbmDMAP,nbMC)                                
             end
          end
          clear MatRZ_ar_iter MatRGauss_ell ArrayWiennerM0transient_ell 
       end
       
       % Computing ArrayH_ar_iter(nu,n_d,nbMC)   
       ArrayH_ar_iter = zeros(nu,n_d,nbMC);                                                            
       for ell=1:nbMC                                                    % ArrayZ_ar_iter(nu,nbmDMAP,nbMC), MatRg(n_d,nbmDMAP) 
              ArrayH_ar_iter(:,:,ell) = ArrayZ_ar_iter(:,:,ell)*MatRg';  % ArrayH_ar_iter(nu,n_d,nbMC)
       end
       
       % Computing h^c(H) loaded in MatRhc_iter(mhc,n_ar)
       MatReta_ar_iter  = reshape(ArrayH_ar_iter,nu,n_ar);              % MatReta_ar_iter(nu,n_ar)
       clear ArrayH_ar_iter   
       
       % Test if there NaN or Inf is obtained in the ISDE solver  
       testNaN =isnan(norm(MatReta_ar_iter,'fro')); 
       if testNaN  >= 1  
          disp(' ');
          disp('----- STOP in sub_solverInverse_constrainedByTargets: NaN or Inf is obtained in the ISDE solver')
          disp(' ');
          disp(['iter         ', num2str(iter)]);  
          disp(' ');
          disp(' If NaN or Inf is obtained after a small number of iterations, decrease the value of alpha_relax1');
          disp(' If NaN or Inf is still obtained, decrease alpha_relax2 and/or increase iter_relax2');
          disp(' ');
          disp('Rlambda_iter_m1 = ');
          disp(Rlambda_iter_m1');

          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n ');
          fprintf(fidlisting,'      \n ');
          fprintf(fidlisting,'----- STOP in sub_solverInverse_constrainedByTargets: NaN or Inf is obtained in the ISDE solver \n '); 
          fprintf(fidlisting,'      \n ');             
          fprintf(fidlisting,'      iter             = %7i \n ',iter);
          fprintf(fidlisting,'      \n ');  
          fprintf(fidlisting,'      If indetermination is reached after a small number of iterations, decrease the value of alpha_relax1. \n');
          fprintf(fidlisting,'       If indetermination is still reached, decrease alpha_relax2 and/or increase iter_relax2. \n');
          fprintf(fidlisting,'      \n ');  
          fprintf(fidlisting,'      Rlambda_iter_m1  = \n');
          fprintf(fidlisting,'                         %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e \n ',Rlambda_iter_m1');
          fprintf(fidlisting,'      \n ');
          fprintf(fidlisting,'      \n ');
          fclose(fidlisting); 

          error('STOP:divergence of ISDE in sub_solverInverse_constrainedByTargets') 
       end
       
       % Computing and loading MatRhc_iter(mhc,n_ar);
       if ind_type_targ == 1                              % Constraints E{h^c_r(H^c)}=Rb_targ1(r,1), r = 1,...,N_r 
          MatRhc_iter = zeros(N_r,n_ar);                  % MatRhc_iter(N_r,n_ar), mhc = N_r
          % Vectorized sequence
          if ind_parallel == 0
             for r = 1:N_r 
                 MatRexp_r        = MatReta_ar_iter - MatReta_targ(:,r); % MatRexp_r(nu,n_ar),MatReta_ar_iter(nu,n_ar),MatReta_targ(nu,N_r)
                 Rexp_r           = exp(-coNr*(sum(MatRexp_r.^2,1)));    % Rexp_r(1,n_ar) 
                 MatRhc_iter(r,:) = Rexp_r;                              % MatRhc_iter(N_r,n_ar)   
             end   
          end
          % Parallel sequence
          if ind_parallel == 1
             parfor r = 1:N_r                            
                 MatRexp_r        = MatReta_ar_iter - MatReta_targ(:,r); % MatRexp_r(nu,n_ar),MatReta_ar_iter(nu,n_ar),MatReta_targ(nu,N_r)
                 Rexp_r           = exp(-coNr*(sum(MatRexp_r.^2,1)));    % Rexp_r(1,n_ar) 
                 MatRhc_iter(r,:) = Rexp_r;                              % MatRhc_iter(N_r,n_ar)   
             end   
          end
       end
       if ind_type_targ == 2                              % Constraints E{H_{ar,j}} = Rb_targ2(j,1) for j = 1,...,nu
          MatRhc_iter              = zeros(mhc,n_ar);     % MatRhc_iter(mhc,n_ar);
          MatRhc_iter(1:nu,:)      = MatReta_ar_iter;     % MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)
       end
       if ind_type_targ == 3                              % Constraints E{H_{ar,j}}=Rb_targ2(j,1) and E{H_{ar,j}^2}=Rb_targ3(j,1), j=1,...,nu
          MatRhc_iter              = zeros(mhc,n_ar);     % MatRhc_iter(mhc,n_ar);
          MatRhc_iter(1:nu,:)      = MatReta_ar_iter;     % MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)
          MatRhc_iter(1+nu:2*nu,:) = MatReta_ar_iter.^2;  % MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)
       end
            
       % Computing the values of the quantities at iteration iter
       Rmeanhc_iter          = mean(MatRhc_iter,2);                   %  Rmeanhc_iter(mhc,1),MatRhc_iter(mhc,n_ar)           
       RGammaP_iter          = Rbc - Rmeanhc_iter;                    %  RGammaP_iter(mhc,1),Rbc(mhc,1),Rmeanhc_iter(mhc,1) 
       MatRGammaS_iter_temp  = cov(MatRhc_iter');                     %  MatRGammaS_iter_temp(mhc,mhc) 
       
       % Updating MatRGammaS_iter_temp(mhc,mhc) if ind_coupling = 0: there is no coupling between the extradiagonal blocks in 
       % matrix  MatRGammaS_iter
       if ind_coupling == 0                                                  
          if ind_type_targ == 1 || ind_type_targ == 2                 % Constraints E{h^c(H_{ar})} = Rbc (no extradiagonal blocks)
              MatRGammaS_iter  = MatRGammaS_iter_temp;                % MatRGammaS_iter(mhc,mhc)   
          end
          if ind_type_targ == 3                                       % constraints E{H_{ar,j}} = 0 and E{H_{ar,j}^2} = 1 for j =1,...,nu
             MatRGammaS_iter_temp(1:nu,nu+1:nu+nu) = zeros(nu,nu);    % there are two extradiagonal blocks: [E{H_{ar}} , E{H_{ar}^.2}}]
             MatRGammaS_iter_temp(nu+1:nu+nu,1:nu) = zeros(nu,nu);    % and its symmetric block:  [E{H_{ar}^.2} , E{H_{ar}} ]
             MatRGammaS_iter  = MatRGammaS_iter_temp;                 % MatRGammaS_iter(mhc,mhc)  
          end   
       end
       if ind_coupling == 1
          MatRGammaS_iter  = MatRGammaS_iter_temp;                    % MatRGammaS_iter(mhc,mhc)  
       end
       clear MatRGammaS_iter_temp 

       % Testing the convergence at iteration iter
       normbc = norm(Rbc);
       if normbc == 0
          normbc = 1;
       end
       Rerr(iter)          = norm(RGammaP_iter)/normbc;               %  Rerr(iter_limit,1)  
       RnormRlambda(iter)  = norm(Rlambda_iter_m1);                   %  RnormRlambda(iter_limit,1) 
       RcondGammaS(iter)   = cond(MatRGammaS_iter);                   %  RcondGammaS(iter_limit,1) 
       if ind_display_screen == 1          
          disp(['err_iter         = ', num2str(Rerr(iter))]);
          disp(['norm_lambda_iter = ', num2str(RnormRlambda(iter))]);
       end
       if iter >= 2
          denom = (abs(Rerr(iter)+Rerr(iter-1)))/2;
          if denom == 0
             denom = 1;
          end
          Rtol(iter) = abs(Rerr(iter)-Rerr(iter-1))/denom;
          if ind_display_screen == 1   
             disp(['tol_iter         = ', num2str(Rtol(iter))]);
          end
       end  
       if ind_print == 1
          fidlisting=fopen('listing.txt','a+'); 
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,'         --- iter number = %7i \n ',iter);
          fprintf(fidlisting,'             err_iter    = %14.7e \n ',Rerr(iter));
          fprintf(fidlisting,'             tol_iter    = %14.7e \n ',Rtol(iter));
          fprintf(fidlisting,'      \n '); 
          fclose(fidlisting); 
       end
       
       % Criterion 1: if iter > 10 and if Rerr(iter)-Rerr(iter-1) > 0, there is a local minimum, obtained for iter - 1.
       % convergence is then assumed to be reached and then, exit from the loop on iter  
       if (iter > 10) && (Rerr(iter)-Rerr(iter-1) > 0)                     
          ind_conv    = 1; 
          MatReta_ar  = MatReta_ar_iter_m1;                         %   MatReta_ar(nu,n_ar), MatReta_ar_iter_m1(nu,n_ar)
          ArrayZ_ar   = ArrayZ_ar_iter_m1;                          %   ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter_m1(nu,nbmDMAP,nbMC)   
          iter_plot   = iter-1;
          if ind_display_screen == 1  
             disp('Convergence with criterion 1: local minimum reached')
          end
          if ind_print == 1
             fidlisting=fopen('listing.txt','a+'); 
             fprintf(fidlisting,'      \n '); 
             fprintf(fidlisting,' --- Convergence with criterion 1: local minimum reached \n ');
             fprintf(fidlisting,'      \n '); 
             fclose(fidlisting); 
          end
          clear MatReta_ar_iter_m1 MatReta_ar_iter
          clear MatRhc_iter Rmeanhc_iter RGammaP_iter Rlambda_iter_m1
          break    % exit from the loop on iter  
       end

       % Criterion 2: if the relative error of the constraint satisfaction is less than or equal to the tolerance. The convergence 
       %              is reached, and then exit from the loop on iter.
       if Rerr(iter) <= epsc  
          ind_conv    = 1;  
          MatReta_ar  = MatReta_ar_iter;                                    %   MatReta_ar(nu,n_ar), MatReta_ar_iter(nu,n_ar)
          ArrayZ_ar   = ArrayZ_ar_iter;                                     %   ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
          iter_plot   = iter;
          if ind_display_screen == 1  
             disp('Convergence with criterion 2: convergence obtained either with variance-values of H-components satisfied');
             disp('                              or relative error of the constraint satisfaction is less than the tolerance');
          end
          if ind_print == 1
             fidlisting=fopen('listing.txt','a+'); 
             fprintf(fidlisting,'      \n ');              
             fprintf(fidlisting,' --- Convergence with criterion 2: convergence obtained either with variance-values \n');
             fprintf(fidlisting,'                                    of H-components satisfied or relative error of the \n');
             fprintf(fidlisting,'                                    constraint satisfaction is less than the tolerance \n');
             fprintf(fidlisting,'                \n '); 
             fclose(fidlisting); 
          end
          clear MatReta_ar_iter_m1 MatReta_ar_iter
          clear  MatRhc_iter Rmeanhc_iter RGammaP_iter Rlambda_iter_m1
          break                          % exit from the loop on iter
       end   

      % Criterion 3: if iter > min(20,iter_limit) and Rtol(iter) < epsc,  the error is stationary and thus
      % the convergence is assumed to be reached and then, exit from the loop on iter  
      if (iter > min(20,iter_limit) && Rerr(iter) < epsc ) &&  (Rtol(iter) < epsc)    
         ind_conv    = 1; 
         MatReta_ar  = MatReta_ar_iter;                                    %   MatReta_ar(nu,n_ar), MatReta_ar_iter(nu,n_ar)
         ArrayZ_ar   = ArrayZ_ar_iter;                                     %   ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
         iter_plot   = iter;
         if ind_display_screen == 1  
            disp('Convergence with criterion 3: iteration number greater that iter_limit and the error is stationary');
          end
         if ind_print == 1
             fidlisting=fopen('listing.txt','a+'); 
             fprintf(fidlisting,'      \n '); 
             fprintf(fidlisting,' --- Convergence with criterion 3: iteration number greater that iter_limit and the error is stationary \n ');
             fprintf(fidlisting,'      \n '); 
             fclose(fidlisting); 
         end         
         clear MatReta_ar_iter_m1 MatReta_ar_iter
         clear MatRhc_iter Rmeanhc_iter RGammaP_iter Rlambda_iter_m1
         break                         % exit from the loop on iter
      end
      
      % Convergence not reached
      if ind_conv == 0
         if ind_type_targ == 1
            [MatRGammaSinv_iter] = sub_solverInverse_pseudo_inverse(MatRGammaS_iter,eps_inv); 
            Rtemp_iter           = MatRGammaSinv_iter*RGammaP_iter;            % Rtemp_iter = pseudo_inv(MatRGammaS_iter)*RGammaP_iter 
         end
         if ind_type_targ == 2 || ind_type_targ == 3
            Rtemp_iter = MatRGammaS_iter\RGammaP_iter;                         % Rtemp_iter = inv(MatRGammaS_iter)*RGammaP_iter
         end   
         Rlambda_iter       = Rlambda_iter_m1 - Ralpha_relax(iter)*Rtemp_iter; % Rlambda_iter(mhc,1), Rlambda_iter_m1(mhc,1)
         Rlambda_iter_m1    = Rlambda_iter;
         MatReta_ar_iter_m1 = MatReta_ar_iter;
         ArrayZ_ar_iter_m1  = ArrayZ_ar_iter;                                    
         clear MatRhc_iter Rlambda_iter Rmeanhc_iter RGammaP_iter MatRGammaS_iter  Rtemp_iter
      end 
   end                         %--- end for iter = 1:iter_limit  
   
   %--- if ind_conv = 0, then iter_limit is reached without convergence
   if ind_conv == 0 
      MatReta_ar  = MatReta_ar_iter;                                    %   MatReta_ar(nu,n_ar), MatReta_ar_iter(nu,n_ar)
      ArrayZ_ar   = ArrayZ_ar_iter;                                     %   ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
      iter_plot   = iter_limit;    
      if ind_display_screen == 1  
         disp('------ No convergence of the iteration algorithm in sub_solverInverse_constrainedByTargets');
         disp(['       iter_plot = ', num2str(iter_plot)]);
         disp('        If convergence is not reached after a small number of iterations, decrease the value of alpha_relax1. \n');
         disp('        If convergence is still not reached, decrease alpha_relax2 and/or increase iter_relax2. \n');
      end
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+'); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' --- No convergence of the iteration algorithm in sub_solverInverse_constrainedByTargets  \n ');
         fprintf(fidlisting,'                                                  \n ');
         fprintf(fidlisting,'     iter             = %7i    \n ',iter_plot);
         fprintf(fidlisting,'     err_iter         = %14.7e \n ',Rerr(iter_plot));
         fprintf(fidlisting,'     tol_iter         = %14.7e \n ',Rtol(iter_plot));
         fprintf(fidlisting,'     norm_lambda_iter = %14.7e \n ',RnormRlambda(iter_plot));
         fprintf(fidlisting,'     condGammaS_iter  = %14.7e \n ',RcondGammaS(iter_plot));
         fprintf(fidlisting,'                                                  \n '); 
         fprintf(fidlisting,'                                                  \n '); 
         fprintf(fidlisting,'     If convergence is not reached after a small number of iterations, decrease the value of alpha_relax1. \n');
         fprintf(fidlisting,'      If convergence is still not reached, decrease alpha_relax2 and/or increase iter_relax2. \n');
         fprintf(fidlisting,'      \n '); 
         fclose(fidlisting); 
      end        
   end

   %--- if ind_conv = 1, then convergence is reached
   if ind_conv == 1 
      if ind_display_screen == 1  
         disp('------ Convergence of the iteration algorithm in sub_solverInverse_constrainedByTargets ');
         disp(['       iter_plot = ', num2str(iter_plot)]);
      end
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+'); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n ');
         fprintf(fidlisting,' --- Convergence of the iteration algorithm in sub_solverInverse_constrainedByTargets   \n '); 
         fprintf(fidlisting,'                                                    \n ');
         fprintf(fidlisting,'     iter             = %7i    \n ',iter_plot);
         fprintf(fidlisting,'     err_iter         = %14.7e \n ',Rerr(iter_plot));
         fprintf(fidlisting,'     tol_iter         = %14.7e \n ',Rtol(iter_plot));
         fprintf(fidlisting,'     norm_lambda_iter = %14.7e \n ',RnormRlambda(iter_plot));
         fprintf(fidlisting,'     condGammaS_iter  = %14.7e \n ',RcondGammaS(iter_plot));         
         fprintf(fidlisting,'      \n '); 
         fclose(fidlisting); 
      end
   end
   
   %--- Plot
   h = figure;      
   plot((1:1:iter_plot)',Rerr(1:iter_plot),'b-')                                                 
   title('Graph of function $\rm{err}(\iota)$','FontSize',16,'Interpreter','latex','FontWeight','normal')                                         
   xlabel('$\iota$','FontSize',16,'Interpreter','latex')                                                                
   ylabel('$\rm{err}(\iota)$','FontSize',16,'Interpreter','latex')  
   numfig = numfig + 1;
   saveas(h,['figure_sub_solverInverse_constrainedByTargets_',num2str(numfig),'_Rerr.fig']); 
   close(h);	

   h = figure;      
   plot((1:1:iter_plot)',RnormRlambda(1:iter_plot),'b-')                                                 
   title('Graph of function $\Vert \lambda_{\iota}\Vert $','FontSize',16,'Interpreter','latex','FontWeight','normal')                                         
   xlabel('$\iota$','FontSize',16,'Interpreter','latex')                                                                
   ylabel('$\Vert \lambda_{\iota}\Vert $','FontSize',16,'Interpreter','latex')  
   numfig = numfig + 1;
   saveas(h,['figure_sub_solverInverse_constrainedByTargets_',num2str(numfig),'_RnormRlambda.fig']); 
   close(h);	
   
   h = figure;      
   plot((1:1:iter_plot)',RcondGammaS(1:iter_plot),'b-')                                                 
   title('Graph of function $\rm{cond} [\Gamma"(\lambda_{\iota})]$','FontSize',16,'Interpreter','latex','FontWeight','normal')                                         
   xlabel('$\iota$','FontSize',16,'Interpreter','latex')                                                                
   ylabel('$\rm{cond} [\Gamma"(\lambda_{\iota})]$','FontSize',16,'Interpreter','latex')  
   numfig = numfig + 1;
   saveas(h,['figure_sub_solverInverse_constrainedByTargets_',num2str(numfig),'_RcondGammaS.fig']); 
   close(h);
   
   %--- Print The relative norm of the extradiagonal term that as to be close to 0R
   %    and print Hmean_ar and diag(MatRHcov_ar)
   %                                                     
   if ind_print == 1                              
      RHmean_ar      = mean(MatReta_ar,2);
      MatRHcov_ar    = cov(MatReta_ar'); 
      RdiagHcov_ar   = diag(MatRHcov_ar);
      normExtra_diag = norm(MatRHcov_ar - diag(RdiagHcov_ar),'fro')/norm(diag(RdiagHcov_ar),'fro');
      fidlisting=fopen('listing.txt','a+'); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n ');
      fprintf(fidlisting,'----- RHmean_ar =          \n '); 
      fprintf(fidlisting,'                 %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e \n ',RHmean_ar');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'----- diag(MatRHcov_ar) =          \n '); 
      fprintf(fidlisting,'                 %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e \n ',RdiagHcov_ar');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'----- Relative Frobenius norm of the extra-diagonal terms of MatRHcov_ar = %14.7e \n ',normExtra_diag); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting); 
   end  
   return
end
 
   


 

 