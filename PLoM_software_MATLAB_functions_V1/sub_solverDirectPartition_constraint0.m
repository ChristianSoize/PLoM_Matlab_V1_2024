function  [MatReta_arj] = sub_solverDirectPartition_constraint0(mj,n_d,nbMC,n_ar,nbmDMAPj,M0transient,Deltarj,f0j,shssj,shj, ...
                                                  MatReta_dj,MatRgj,MatRaj,ArrayWiennerM0transient,ArrayGauss,ind_parallel,ind_print) 

   %-------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 2 July 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverDirectPartition_constraint0.m
   %  Subject      : ind_constraints = 0: solver PLoM for direct predictions with the partition and without constraints 
   %                 of normalization, for Y^j for each group j. The notations of the partition are
   %                 H    = (H_1,...,H_r,...,H_nu)    in ngroup subsets (groups) H^1,...,H^j,...,H^ngroup
   %                 H    = (Y^1,...,Y^j,...,Y^ngroup) 
   %                 Y^j  = = (Y^j_1,...,Y^j_mj) = (H_rj1,...,Hrjmj)   with j = 1,...,ngroup and with n1 + ... + nngroup = nu
   %  Comment      : this function is derived from sub_solverDirect_constraint0.m
   %
   %--- INPUT  
   %      mj                  : dimension of random vector Y^j  = = (Y^j_1,...,Y^j_mj) = (H_rj1,...,Hrjmj) 
   %      n_d                 : number of points in the training dataset for H_d
   %      nbMC                : number of realizations of (mj,n_d)-valued random matrix [Y^j] 
   %      n_ar                : number of realizations of Y^j such that n_ar  = nbMC x n_d
   %      nbmDMAPj            : dimension of the ISDE-projection basis
   %      M0transient         : mjmber of steps for reaching the stationary solution
   %      Deltarj             : ISDE integration step by Verlet scheme
   %      f0j                 : dissipation coefficient in the ISDE   
   %      shssj,shj           : parameters of the GKDE of pdf of Y^j_d (training) 
   %      MatReta_dj(mj,n_d)  : n_d realizations of Y^j_d (training)    
   %      MatRgj(n_d,nbmDMAPj): matrix of the ISDE-projection basis
   %      MatRaj(n_d,nbmDMAPj): related to MatRgj(n_d,nbmDMAPj) 
   %      ArrayWiennerM0transient(mj,n_d,M0transient,nbMC)  : realizations of the matrix-valued normalized Wienner process
   %      ArrayGauss(mj,n_d,nbMC)                           : realizations of the Gaussian matrix-valued random variable
   %      ind_parallel        : = 0 no parallel computation, = 1 parallel computation
   %      ind_print           : = 0 no print,                = 1 print
   %
   %--- OUTPUT
   %      MatReta_arj(mj,n_ar)
    
   %--- Initialization
   ArrayZ_ar = zeros(mj,nbmDMAPj,nbMC);

   %--- Vectorized computation of ArrayZ_ar(mj,nbmDMAPj,nbMC)
   if ind_parallel == 0        
      for ell = 1:nbMC 
         MatRGauss_ell               = ArrayGauss(:,:,ell);                   %  ArrayGauss(mj,n_d,nbMC)
         ArrayWiennerM0transient_ell = ArrayWiennerM0transient(:,:,:,ell);    %  ArrayWiennerM0transient(mj,n_d,M0transient,nbMC)
           
         [MatRZ_ar_ell] = sub_solverDirect_Verlet_constraint0(mj,n_d,M0transient,Deltarj,f0j,MatReta_dj,MatRgj,MatRaj,shssj,shj, ...
                                                              ArrayWiennerM0transient_ell,MatRGauss_ell); 
         ArrayZ_ar(:,:,ell) = MatRZ_ar_ell;   % ArrayZ_ar(mj,nbmDMAPj,nbMC), MatRZ_ar_ell(mj,nbmDMAPj)                                  
      end
   end

   %--- Parallel computation of ArrayZ_ar(mj,nbmDMAPj,nbMC)
   if ind_parallel == 1        
      parfor ell = 1:nbMC 
         MatRGauss_ell               = ArrayGauss(:,:,ell);                   %  ArrayGauss(mj,n_d,nbMC)
         ArrayWiennerM0transient_ell = ArrayWiennerM0transient(:,:,:,ell);    %  ArrayWiennerM0transient(mj,n_d,M0transient,nbMC)
           
         [MatRZ_ar_ell] = sub_solverDirect_Verlet_constraint0(mj,n_d,M0transient,Deltarj,f0j,MatReta_dj,MatRgj,MatRaj,shssj,shj, ...
                                                              ArrayWiennerM0transient_ell,MatRGauss_ell); 
         ArrayZ_ar(:,:,ell) = MatRZ_ar_ell;   % ArrayZ_ar(mj,nbmDMAPj,nbMC), MatRZ_ar_ell(mj,nbmDMAPj)                                  
      end
   end
  
   clear MatRZ_ar_ell MatRGauss_ell ArrayWiennerM0transient_ell 

   %--- Computing MatReta_arj(mj,n_ar) with a Vectorized sequence.
   %    Parallel computation with a parfor loop is not implemented due to possible RAM limitations
   %    and considering that the CPU time is not significant with vectorized computation.
   ArrayH_ar = zeros(mj,n_d,nbMC);                                           
   for ell=1:nbMC
       ArrayH_ar(:,:,ell) = ArrayZ_ar(:,:,ell)*MatRgj';  % ArrayH_ar(mj,n_d,nbMC), ArrayZ_ar(mj,nbmDMAPj,nbMC), MatRgj(n_d,nbmDMAPj) 
   end
   MatReta_arj  = reshape(ArrayH_ar,mj,n_ar);            % MatReta_arj(mj,n_ar)

   %--- Print The relative norm of the extradiagonal term that as to be close to 0
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
  


 

 