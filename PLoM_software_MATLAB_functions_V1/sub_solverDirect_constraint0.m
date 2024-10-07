function  [MatReta_ar,ArrayZ_ar] = sub_solverDirect_constraint0(nu,n_d,nbMC,n_ar,nbmDMAP,M0transient,Deltar,f0,shss,sh, ...
                                                  MatReta_d,MatRg,MatRa,ArrayWiennerM0transient,ArrayGauss,ind_parallel,ind_print) 

   %-------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverDirect_constraint0
   %  Subject      : ind_constraints = 0 : No constraints on H  
   %
   %--- INPUT  
   %      nu                  : dimension of random vector H_d = (H_1, ... H_nu) and H_ar
   %      n_d                 : number of points in the training dataset for H_d
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
   %      ind_parallel        : = 0 no parallel computation, = 1 parallel computation
   %      ind_print           : = 0 no print,                = 1 print
   %
   %--- OUTPUT
   %      MatReta_ar(nu,n_ar)
   %      ArrayZ_ar(nu,nbmDMAP,nbMC);    % ArrayZ_ar(nu,nbmDMAP,nbMC), this array is used as ouput for possible use in a postprocessing 
   %                                     % of Z in order to construct its polynomial chaos expansion (PCE)
  
   %--- Initialization
   ArrayZ_ar = zeros(nu,nbmDMAP,nbMC);

   %--- Vectorized computation of ArrayZ_ar(nu,nbmDMAP,nbMC)
   if ind_parallel == 0        
      for ell = 1:nbMC 
         MatRGauss_ell               = ArrayGauss(:,:,ell);                   %  ArrayGauss(nu,n_d,nbMC)
         ArrayWiennerM0transient_ell = ArrayWiennerM0transient(:,:,:,ell);    %  ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)
           
         [MatRZ_ar_ell] = sub_solverDirect_Verlet_constraint0(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRg,MatRa,shss,sh, ...
                                                              ArrayWiennerM0transient_ell,MatRGauss_ell); 
         ArrayZ_ar(:,:,ell) = MatRZ_ar_ell;   % ArrayZ_ar(nu,nbmDMAP,nbMC), MatRZ_ar_ell(nu,nbmDMAP)                                  
      end
   end

   %--- Parallel computation of ArrayZ_ar(nu,nbmDMAP,nbMC)
   if ind_parallel == 1        
      parfor ell = 1:nbMC 
         MatRGauss_ell               = ArrayGauss(:,:,ell);                   %  ArrayGauss(nu,n_d,nbMC)
         ArrayWiennerM0transient_ell = ArrayWiennerM0transient(:,:,:,ell);    %  ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)
           
         [MatRZ_ar_ell] = sub_solverDirect_Verlet_constraint0(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRg,MatRa,shss,sh, ...
                                                              ArrayWiennerM0transient_ell,MatRGauss_ell); 
         ArrayZ_ar(:,:,ell) = MatRZ_ar_ell;   % ArrayZ_ar(nu,nbmDMAP,nbMC), MatRZ_ar_ell(nu,nbmDMAP)                                  
      end
   end
  
   clear MatRZ_ar_ell MatRGauss_ell ArrayWiennerM0transient_ell 

   %--- Computing MatReta_ar(nu,n_ar) with a Vectorized sequence.
   %    Parallel computation with a parfor loop is not implemented due to possible RAM limitations
   %    and considering that the CPU time is not significant with vectorized computation.
   ArrayH_ar = zeros(nu,n_d,nbMC);                                           
   for ell=1:nbMC
       ArrayH_ar(:,:,ell) = ArrayZ_ar(:,:,ell)*MatRg';  % ArrayH_ar(nu,n_d,nbMC), ArrayZ_ar(nu,nbmDMAP,nbMC), MatRg(n_d,nbmDMAP) 
   end
   MatReta_ar  = reshape(ArrayH_ar,nu,n_ar);            % MatReta_ar(nu,n_ar)

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
  


 

 