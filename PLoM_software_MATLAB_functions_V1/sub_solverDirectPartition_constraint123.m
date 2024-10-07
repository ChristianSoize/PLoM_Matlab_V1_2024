function  [MatReta_arj] = sub_solverDirectPartition_constraint123(j,mj,n_d,nbMC,n_ar,nbmDMAPj,M0transient,Delatarj,f0j,shssj,shj, ...
                                       MatReta_dj,MatRgj,MatRaj,ArrayWiennerM0transientj,ArrayGaussj,ind_constraints,ind_coupling, ...
                                       epsc,iter_limit,Ralpha_relax,minVarH,maxVarH,ind_display_screen,ind_print,ind_parallel,numfig) 

   %-------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 2 July 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverDirect_constraint123
   %  Subject      : ind_constraints = 1, 2, or 3: solver PLoM for direct predictions with the partition and with constraints 
   %                 of normalization for Y^j for each group j. The notations of the partition are
   %                 H    = (H_1,...,H_r,...,H_nu)    in ngroup subsets (groups) H^1,...,H^j,...,H^ngroup
   %                 H    = (Y^1,...,Y^j,...,Y^ngroup) 
   %                 Y^j  = = (Y^j_1,...,Y^j_mj) = (H_rj1,...,Hrjmj)   with j = 1,...,ngroup and with n1 + ... + nngroup = nu
   %  Comment      : this function is derived from sub_solverDirect_constraint123.m
   %
   %--- INPUT 
   %      For each group j
   %      j                   : group number in {1,...,ngroup}
   %      mj                  : dimension of random vector Y^j
   %      n_d                 : number of points in the training set for Y^j_d (also of H_d)
   %      nbMC                : number of realizations of (mj,n_d)-valued random matrix [Y^j_ar] 
   %      n_ar                : number of realizations of H_ar = (Y^1_ar,...Y^ngroup_ar) such that n_ar  = nbMC x n_d
   %      nbmDMAPj            : dimension of the ISDE-projection basis
   %      Delatarj            : ISDE integration step by Verlet scheme
   %      f0j                 : dissipation coefficient in the ISDE   
   %      shssj,shj           : parameters of the GKDE of pdf of Y^j_d (training) 
   %      MatReta_dj(mj,n_d)  : n_d realizations of Y^j_d (training)    
   %      MatRgj(n_d,nbmDMAPj): matrix of the ISDE-projection basis
   %      MatRaj(n_d,nbmDMAPj): related to MatRgj(n_d,nbmDMAPj) 
   %      ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)  : realizations of the matrix-valued normalized Wienner process
   %      ArrayGaussj(mj,n_d,nbMC)                           : realizations of the Gaussian matrix-valued random variable
   %
   %      For all the groups (independent of j)
   %      ind_constraints     : type of constraints (= 1,2, or 3)
   %      ind_coupling        : 0 (no coupling), = 1 (coupling for the Lagrange mutipliers computation)
   %      epsc                : relative tolerance for the iteration convergence of the constraints on Y^j_ar  for any j
   %      iter_limit          : maximum number of iteration for computing the Lagrange multipliers
   %      Ralpha_relax        : Ralpha_relax(iter_limit,1) relaxation parameter in ] 0 , 1 ] for the iterations. 
   %      minVarH             : minimum imposed on E{(Y^j_k)^2} with respect to 1 (for instance 0.999)  (for any j)
   %      maxVarH             : maximum imposed on E{(Y^j_k)^2} with respect to 1 (for instance 1.001)  (for any j)
   %      ind_display_screen  : = 0 no display,              = 1 display
   %      ind_print           : = 0 no print,                = 1 print
   %      ind_parallel        : = 0 no parallel computation, = 1 parallel computation
   %
   %--- OUTPUT
   %      MatReta_arj(mj,n_ar)
   %
   %--- COMMENTS
   %    ind_constraints     = 0 : no constraints concerning E{Y^j] = 0 and E{Y^j (Y^j)'} = [I_mj]
   %                        = 1 : constraints E{(Y^j_k)^2} = 1 for k =1,...,mj   
   %                        = 2 : constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
   %                        = 3 : constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj]  
   %      ind_coupling      = 0 : for ind_constraints = 2 or 3, no coupling between the Lagrange multipliers in matrix MatRGammaS_iter 
   %                        = 1 : for ind_constraints = 2 or 3, coupling, all the extra-diagonal blocs in matrix MatRGammaS_iter are kept
 
   %--- Initialization and preallocation    
   if ind_constraints == 1                   %--- Constraints E{(Y^j_k)^2} = 1 for k =1,...,mj  are used
      mhc = mj;
      Rbc = ones(mj,1);                      % Rbc(mhc,1):  E{h^c(Y^j)} = b^c corresponds to  E{(Y^j_k)^2} = 1 for k =1,...,mj
      Rlambda_iter_m1    = zeros(mhc,1);     % Rlambda_iter_m1(mhc,1) = (lambda_1,...,lambda_mj)
      MatRlambda_iter_m1 = [];               % MatRlambda_iter_m1(mj,mj) is not used
   end
   if ind_constraints == 2                   %--- constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj are used
      mhc = 2*mj;
      Rbc = zeros(mhc,1);                    % Rbc(mhc,1):  E{h^c(Y^j)} = b^c corresponds to E{H} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
      Rbc(1+mj:2*mj,1)   = ones(mj,1); 
      Rlambda_iter_m1    = zeros(mhc,1);     % Rlambda_iter_m1(mhc,1) = (lambda_1,...,lambda_mj,lambda_{1+mj},...,lambda_{mj+mj})
      MatRlambda_iter_m1 = [];               % MatRlambda_iter_m1(mj,mj) is not used
   end
   if ind_constraints == 3                   %--- constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj] are used
      mhc = 2*mj + mj*(mj-1)/2;              % note that for mj = 1, we have mhc = 2*mj
                                             % if mj = 1: Rbc(mhc,1) = (bc_1,...,bc_mj , bc_{1+mj},...,bc_{mj+mj} )
                                             % if mj > 1: Rbc(mhc,1) = (bc_1,...,bc_mj , bc_{1+mj},...,bc_{mj+mj} , {bc_{(j,i) + 2 mj} 
                                             %            for 1 <= j < i <= mj} )
      Rbc              = zeros(mhc,1);       % Rbc(mhc,1):  E{h^c(Y^j)} = b^c corresponds to E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj] 
      Rbc(1+mj:2*mj,1) = ones(mj,1); 
      Rlambda_iter_m1  = zeros(mhc,1);       % Rlambda_iter_m1(mhc,1)      
      if mj == 1                             % Rlambda_iter(mhc,1) = (lambda_1,...,lambda_mj,lambda_{1+mj},...,lambda_{mj+mj})
         MatRlambda_iter_m1 = [];            % MatRlambda_iter_m1(mj,mj) is not used  
      end    
      if mj >=2                              % Rlambda_iter(mhc,1) = (lambda_1,...,lambda_mj , lambda_{1+mj},...,lambda_{mj+mj} ,
                                             % {lambda_{(j,i) + 2 mj} for 1 <= j < i <= mj} ), and MatRlambda_iter_m1(mj,mj) = symmetric 
         MatRlambda_iter_m1 = zeros(mj,mj);  % with zero diagonal and for which upper part is Rlambda_iter row rise
         ind = 0;
         for k = 1:mj-1
             for i = k+1:mj
                 ind = ind + 1;
                 MatRlambda_iter_m1(k,i) = Rlambda_iter_m1(2*mj+ind); % Rlambda_iter_m1(mhc,1); 
                 MatRlambda_iter_m1(i,k) = MatRlambda_iter_m1(k,i);   % MatRlambda_iter_m1(mj,mj)
             end
         end
      end
   end

   %--- Print parameters
   if ind_print == 1
      fidlisting=fopen('listing.txt','a+'); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'--- Constraints parameters in solverDirect with constraints \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'    ind_constraints = %7i \n ',ind_constraints); 
      fprintf(fidlisting,'    mj              = %7i \n ',mj); 
      fprintf(fidlisting,'    mhc             = %7i \n ',mhc); 
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
   
   %--- Loop of the iteration algorithm (iter: lambda_iter given and compute lambda_iterplus1)

   for iter = 1:iter_limit   

       if ind_display_screen == 1
          disp(['------------- iter number: ', num2str(iter)]);
       end  

       % Constraints E{(Y^j_k)^2} = 1 for k =1,...,mj
       if ind_constraints == 1                           
          ArrayZ_ar_iter = zeros(mj,nbmDMAPj,nbMC);      % ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)

          % Vectorized sequence
          if ind_parallel == 0
             for ell = 1:nbMC 
                MatRGauss_ell               = ArrayGaussj(:,:,ell);                   % ArrayGaussj(mj,n_d,nbMC)
                ArrayWiennerM0transient_ell = ArrayWiennerM0transientj(:,:,:,ell);    % ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)  
                [MatRZ_ar_iter] = sub_solverDirect_Verlet_constraint1(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj,MatRgj,shssj,shj, ...
                                                                      ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1); 
                ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)                                
             end
          end

          % Parallel sequence
          if ind_parallel == 1
             parfor ell = 1:nbMC 
                MatRGauss_ell               = ArrayGaussj(:,:,ell);                   % ArrayGaussj(mj,n_d,nbMC)
                ArrayWiennerM0transient_ell = ArrayWiennerM0transientj(:,:,:,ell);    % ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)
                [MatRZ_ar_iter] = sub_solverDirect_Verlet_constraint1(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj,MatRgj,shssj,shj, ...
                                                                      ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1); 
                ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)                                
             end
          end

          clear MatRZ_ar_iter MatRGauss_ell ArrayWiennerM0transient_ell 
       end
       
       % Constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
       if ind_constraints == 2                          
          ArrayZ_ar_iter = zeros(mj,nbmDMAPj,nbMC);                                   % ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)

          % Vectorized sequence
          if ind_parallel == 0
             for ell = 1:nbMC 
                 MatRGauss_ell               = ArrayGaussj(:,:,ell);                   % ArrayGaussj(mj,n_d,nbMC)
                 ArrayWiennerM0transient_ell = ArrayWiennerM0transientj(:,:,:,ell);    % ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)   
                 [MatRZ_ar_iter] = sub_solverDirect_Verlet_constraint2(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj,MatRgj,shssj,shj,  ...
                                                                       ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1);                                                  
                 ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)                                
             end
          end

          % Parallel sequence
          if ind_parallel == 1
             parfor ell = 1:nbMC 
                 MatRGauss_ell               = ArrayGaussj(:,:,ell);                   % ArrayGaussj(mj,n_d,nbMC)
                 ArrayWiennerM0transient_ell = ArrayWiennerM0transientj(:,:,:,ell);    % ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)   
                [MatRZ_ar_iter] = sub_solverDirect_Verlet_constraint2(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj,MatRgj,shssj,shj,  ...
                                                                       ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1);                                                 
                 ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)                                
             end
          end

          clear MatRZ_ar_iter MatRGauss_ell ArrayWiennerM0transient_ell 
       end
       
       % Constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj]  
       if ind_constraints == 3                          
          ArrayZ_ar_iter = zeros(mj,nbmDMAPj,nbMC);                                   % ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)

          % Vectorized sequence
          if ind_parallel == 0 
             for ell = 1:nbMC 
                 MatRGauss_ell               = ArrayGaussj(:,:,ell);                   % ArrayGaussj(mj,n_d,nbMC)
                 ArrayWiennerM0transient_ell = ArrayWiennerM0transientj(:,:,:,ell);    % ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)

                 [MatRZ_ar_iter] = sub_solverDirect_Verlet_constraint3(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj, ...
                                                                        MatRgj,shssj,shj,ArrayWiennerM0transient_ell,MatRGauss_ell, ...
                                                                        Rlambda_iter_m1,MatRlambda_iter_m1); 
                 ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)  
             end
          end

          % Parallel sequence
          if ind_parallel == 1 
             parfor ell = 1:nbMC 
                 MatRGauss_ell               = ArrayGaussj(:,:,ell);                   % ArrayGaussj(mj,n_d,nbMC)
                 ArrayWiennerM0transient_ell = ArrayWiennerM0transientj(:,:,:,ell);    % ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)

                 [MatRZ_ar_iter] = sub_solverDirect_Verlet_constraint3(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj, ...
                                                                        MatRgj,shssj,shj,ArrayWiennerM0transient_ell,MatRGauss_ell, ...
                                                                        Rlambda_iter_m1,MatRlambda_iter_m1);   
                 ArrayZ_ar_iter(:,:,ell) = MatRZ_ar_iter;                             % ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)  
             end
          end

          clear MatRZ_ar_iter MatRGauss_ell ArrayWiennerM0transient_ell 
       end
       
       % Computing ArrayH_ar_iter(mj,n_d,nbMC)   
       ArrayH_ar_iter = zeros(mj,n_d,nbMC);                                                            
       for ell=1:nbMC                                                     % ArrayZ_ar_iter(mj,nbmDMAPj,nbMC), MatRgj(n_d,nbmDMAPj) 
              ArrayH_ar_iter(:,:,ell) = ArrayZ_ar_iter(:,:,ell)*MatRgj';  % ArrayH_ar_iter(mj,n_d,nbMC)
       end
       
       % Computing h^c(Y^j) loaded in MatRhc_iter(mhc,n_ar)
       MatReta_ar_iter  = reshape(ArrayH_ar_iter,mj,n_ar);              % MatReta_ar_iter(mj,n_ar)
       clear ArrayH_ar_iter   
       
       % Test if there NaN or Inf is obtained in the ISDE solver  
       testNaN = isnan(norm(MatReta_ar_iter,'fro')); 
       if testNaN  >= 1  
          disp(' ');
          disp('----- STOP in sub_solverDirect_constraint123: NaN or Inf is obtained in the ISDE solver ')
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
          fprintf(fidlisting,'----- STOP in sub_solverDirect_constraint123: NaN or Inf is obtained in the ISDE solver \n '); 
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

          error('STOP: divergence of ISDE in sub_solverDirect_constraint123') 
       end
       
       % Computing and loading MatRhc_iter(mhc,n_ar);
       if ind_constraints == 1                                            % Constraints E{(Y^j_k)^2} = 1 for k =1,...,mj
          MatRhc_iter              = zeros(mhc,n_ar);                     % MatRhc_iter(mhc,n_ar);
          MatRhc_iter(1:mj,:)      = MatReta_ar_iter.^2;                  % MatReta_ar_iter(mj,n_ar),MatRhc_iter(mhc,n_ar)
       end
       if ind_constraints == 2                                            % Constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
          MatRhc_iter              = zeros(mhc,n_ar);                     % MatRhc_iter(mhc,n_ar);
          MatRhc_iter(1:mj,:)      = MatReta_ar_iter;                     % MatReta_ar_iter(mj,n_ar),MatRhc_iter(mhc,n_ar)
          MatRhc_iter(1+mj:2*mj,:) = MatReta_ar_iter.^2;                  % MatReta_ar_iter(mj,n_ar),MatRhc_iter(mhc,n_ar)
       end
       if ind_constraints == 3                                            % Constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj]  
          MatRhc_iter              = zeros(mhc,n_ar);                     % MatRhc_iter(mhc,n_ar);
          MatRhc_iter(1:mj,:)      = MatReta_ar_iter;                     % MatReta_ar_iter(mj,n_ar),MatRhc_iter(mhc,n_ar)
          MatRhc_iter(1+mj:2*mj,:) = MatReta_ar_iter.^2;                  % MatReta_ar_iter(mj,n_ar),MatRhc_iter(mhc,n_ar)
          if mj >=2  
             ind = 0;
             for k = 1:mj-1
                 for i = k+1:mj
                     ind = ind + 1;
                     MatRhc_iter(2*mj+ind,:) = MatReta_ar_iter(k,:).*MatReta_ar_iter(i,:);
                 end
             end
          end
       end
     
       % Computing the values of the quantities at iteration iter
       RVarH_iter          = var(MatReta_ar_iter,0,2);
       minVarH_iter        = min(RVarH_iter );
       maxVarH_iter        = max(RVarH_iter );
       clear RVarH_iter
                                               
       Rmeanhc_iter          = mean(MatRhc_iter,2);                           %  Rmeanhc_iter(mhc,1),MatRhc_iter(mhc,n_ar)           
       RGammaP_iter          = Rbc - Rmeanhc_iter;                            %  RGammaP_iter(mhc,1),Rbc(mhc,1),Rmeanhc_iter(mhc,1) 
       MatRGammaS_iter_temp  = cov(MatRhc_iter');                             %  MatRGammaS_iter_temp(mhc,mhc) 
       
       % Updating MatRGammaS_iter(mhc,mhc) if ind_coupling = 0:  For ind_constraints = 2 or 3, if ind_coupling = 0, there is no coupling 
       % between the Lagrange multipliers in the matrix MatRGammaS_iter
       if ind_coupling == 0                                                  
          if ind_constraints == 1                                             % Constraints  E{(Y^j_k)^2} = 1 for k =1,...,mj
              MatRGammaS_iter  = MatRGammaS_iter_temp;                        % MatRGammaS_iter(mhc,mhc)   
          end
          if ind_constraints == 2                                             % Constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
             MatRGammaS_iter_temp(1:mj,mj+1:mj+mj) = zeros(mj,mj);
             MatRGammaS_iter_temp(mj+1:mj+mj,1:mj) = zeros(mj,mj);
             MatRGammaS_iter  = MatRGammaS_iter_temp;                         % MatRGammaS_iter(mhc,mhc)  
          end
          if ind_constraints == 3                                             % Constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj] 
             MatRGammaS_iter_temp(1:mj,1+mj:mhc) = zeros(mj,mhc-mj);
             MatRGammaS_iter_temp(1+mj:mhc,1:mj) = zeros(mhc-mj,mj);
             MatRGammaS_iter_temp(1+mj:mj+mj,1+2*mj:mhc) = zeros(mj,mhc-2*mj);
             MatRGammaS_iter_temp(1+2*mj:mhc,1+mj:mj+mj) = zeros(mhc-2*mj,mj);
             MatRGammaS_iter  = MatRGammaS_iter_temp;                         % MatRGammaS_iter(mhc,mhc)  
          end 
       end
       if ind_coupling == 1
          MatRGammaS_iter = MatRGammaS_iter_temp;                             % MatRGammaS_iter(mhc,mhc)  
       end
       clear MatRGammaS_iter_temp 
       
       % Testing the convergence at iteration iter
       Rerr(iter)          = norm(RGammaP_iter)/norm(Rbc);                    %  Rerr(iter_limit,1)  
       RnormRlambda(iter)  = norm(Rlambda_iter_m1);                           %  RnormRlambda(iter_limit,1) 
       RcondGammaS(iter)   = cond(MatRGammaS_iter);                           %  RcondGammaS(iter_limit,1) 
       if ind_display_screen == 1          
          disp(['err_iter         = ', num2str(Rerr(iter))]);
          disp(['norm_lambda_iter = ', num2str(RnormRlambda(iter))]);
       end
       if iter >= 2
          Rtol(iter) = 2*abs(Rerr(iter)-Rerr(iter-1))/abs(Rerr(iter)+Rerr(iter-1));
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
          MatReta_arj  = MatReta_ar_iter_m1;                         %   MatReta_arj(mj,n_ar), MatReta_ar_iter_m1(mj,n_ar)
          iter_plot   = iter-1;
          if ind_display_screen == 1  
             disp('Convergence with criterion 1: local minimum reached')
             disp('If convergence is not sufficiently good, decrease the value of alpha_relax2');
          end
          if ind_print == 1
             fidlisting=fopen('listing.txt','a+'); 
             fprintf(fidlisting,'      \n '); 
             fprintf(fidlisting,' --- Convergence with criterion 1: local minimum reached \n ');
             fprintf(fidlisting,'     If convergence is not sufficiently good, decrease the value of alpha_relax2 \n ');
             fprintf(fidlisting,'      \n '); 
             fclose(fidlisting); 
          end
          clear MatReta_ar_iter_m1 MatReta_ar_iter
          clear MatRhc_iter Rmeanhc_iter RGammaP_iter Rlambda_iter_m1
          break    % exit from the loop on iter  
       end

       % Criterion 2: if {minVarH_iter >= minVarH and maxVarH_iter <= maxVarH} or Rerr(iter) <= epsc, the variance of each component is greater 
       % than or equal to minVarH and less than or equal to maxVarH, or the relative error of the constraint satisfaction is less 
       % than or equal to the tolerance. The convergence is reached, and then exit from the loop on iter.
       if ((minVarH_iter >= minVarH) && (maxVarH_iter <= maxVarH)) || Rerr(iter) <= epsc  
          ind_conv    = 1;  
          MatReta_arj  = MatReta_ar_iter;                                    %   MatReta_arj(mj,n_ar), MatReta_ar_iter(mj,n_ar)
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
          clear MatRhc_iter Rmeanhc_iter RGammaP_iter Rlambda_iter_m1
          break                          % exit from the loop on iter
       end   

      % Criterion 3: if iter > min(20,iter_limit) and Rtol(iter) < epsc,  the error is stationary and thus
      % the convergence is assumed to be reached and then, exit from the loop on iter  
      if (iter > min(20,iter_limit) && Rerr(iter) < epsc ) &&  (Rtol(iter) < epsc)    
         ind_conv    = 1; 
         MatReta_arj  = MatReta_ar_iter;                                    %   MatReta_arj(mj,n_ar), MatReta_ar_iter(mj,n_ar)
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
      
      % Convergence not reached: the variance of each component is less than minVarH or greater than maxVarH, 
      % the relative error of the constraint satisfaction is greater than the tolerance, there is no local minimum,
      % and there is no stationary point.  
      if ind_conv == 0
          Rtemp_iter      = MatRGammaS_iter\RGammaP_iter;                            % Rtemp_iter = inv(MatRGammaS_iter)*RGammaP_iter
          Rlambda_iter    = Rlambda_iter_m1 - Ralpha_relax(iter)*Rtemp_iter;         % Rlambda_iter(mhc,1), Rlambda_iter_m1(mhc,1)
          MatRlambda_iter = [];
          if ind_constraints == 3 && mj >= 2                                         % constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj] 
             MatRlambda_iter = zeros(mj,mj);                                         % MatRlambda_iter(mj,mj) 
             ind = 0;
             for k = 1:mj-1
                 for i = k+1:mj
                     ind = ind + 1;
                     MatRlambda_iter(k,i) = Rlambda_iter(2*mj+ind);
                     MatRlambda_iter(i,k) = MatRlambda_iter(k,i);
                 end
             end
          end 
          Rlambda_iter_m1    = Rlambda_iter;
          MatRlambda_iter_m1 = MatRlambda_iter;  
          MatReta_ar_iter_m1 = MatReta_ar_iter;                               
          clear MatRhc_iter Rlambda_iter MatRlambda_iter Rmeanhc_iter RGammaP_iter MatRGammaS_iter  Rtemp_iter
      end 
   end                         %--- end for iter = 1:iter_limit  
   
   %--- if ind_conv = 0, then iter_limit is reached without convergence
   if ind_conv == 0 
      MatReta_arj  = MatReta_ar_iter;                                    %   MatReta_arj(mj,n_ar), MatReta_ar_iter(mj,n_ar)
      iter_plot   = iter_limit;    
      if ind_display_screen == 1  
         disp('------ No convergence of the iteration algorithm in sub_solverDirect_constraint123');
         disp(['       iter_plot = ', num2str(iter_plot)]);
         disp('        If convergence is not reached after a small number of iterations, decrease the value of alpha_relax1. \n');
         disp('        If convergence is still not reached, decrease alpha_relax2 and/or increase iter_relax2. \n');
      end
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+'); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' --- No convergence of the iteration algorithm in sub_solverDirect_constraint123 \n ');
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
         disp('------ Convergence of the iteration algorithm in sub_solverDirect_constraint123');
         disp(['       iter_plot = ', num2str(iter_plot)]);
      end
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+'); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n ');
         fprintf(fidlisting,' --- Convergence of the iteration algorithm in sub_solverDirect_constraint123   \n '); 
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
   title(['Graph of function $\rm{err}(\iota)$ for group ',num2str(j)],'FontSize',16,'Interpreter','latex','FontWeight','normal')                                         
   xlabel('$\iota$','FontSize',16,'Interpreter','latex')                                                                
   ylabel('$\rm{err}(\iota)$','FontSize',16,'Interpreter','latex')  
   numfig = numfig + 1;
   saveas(h,['figure_solverDirectPartition_constraint123_',num2str(numfig),'_Rerr_group',num2str(j),'.fig']); 
   close(h);	

   h = figure;      
   plot((1:1:iter_plot)',RnormRlambda(1:iter_plot),'b-')                                                 
   title(['Graph of function $\Vert \lambda_{\iota}\Vert$ for group ',num2str(j)],'FontSize',16,'Interpreter','latex','FontWeight','normal')                                         
   xlabel('$\iota$','FontSize',16,'Interpreter','latex')                                                                
   ylabel('$\Vert \lambda_{\iota}\Vert $','FontSize',16,'Interpreter','latex')  
   numfig = numfig + 1;
   saveas(h,['figure_solverDirectPartition_constraint123_',num2str(numfig),'_RnormRlambda_group',num2str(j),'.fig']); 
   close(h);	
   
   h = figure;      
   plot((1:1:iter_plot)',RcondGammaS(1:iter_plot),'b-')                                                 
   title(['Graph of function $\rm{cond} [\Gamma"(\lambda_{\iota})]$ for group ',num2str(j)],'FontSize',16,'Interpreter','latex','FontWeight','normal')                                         
   xlabel('$\iota$','FontSize',16,'Interpreter','latex')                                                                
   ylabel('$\rm{cond} [\Gamma"(\lambda_{\iota})]$','FontSize',16,'Interpreter','latex')  
   numfig = numfig + 1;
   saveas(h,['figure_solverDirectPartition_constraint123_',num2str(numfig),'_RcondGammaS_group',num2str(j),'.fig']); 
   close(h);
   
   %--- Print The relative norm of the extradiagonal term that as to be close to 0R
   %    and print Hmean_ar and diag(MatRHcov_ar)
   %                                                     
   if ind_print == 1                              
      RHmean_ar      = mean(MatReta_arj,2);
      MatRHcov_ar    = cov(MatReta_arj'); 
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
 
   


 

 