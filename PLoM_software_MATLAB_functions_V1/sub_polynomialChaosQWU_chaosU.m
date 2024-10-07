function [MatRphiU,MatPowerU,MatRaU] = sub_polynomialChaosQWU_chaosU(KU,ndeg,ng,NnbMC0,MatRU)

   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 23 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_polynomialChaosQWU_chaosU
   %  Subject      : Computing matrices: MatRphiU(KU,NnbMC0),MatPowerU(KU,ng),MatRaU(KU,KU) of the polynomial chaos 
   %                     phi_{a^(m)}(U) with a^(m) = (a_1^(m),...,a_ng^(m)) in R^ng with m=1,...,KU
   %                     U            = (U_1, ... , U_ng) random vector for normalized Gaussian germ
   %                     ng           = dimension of the germ U = (U_1, ... , U_ng) 
   %                     m            = 1,...,KU indices of multi index (a_1^(m),...,a_ng^(m)) of length ng   
   %                     ndeg         = max degree of the polynomial chaos                                      
   %                     KU           = factorial(ng+ndeg)/(factorial(ng)*factorial(ndeg)) = number
   %                                    of terms including a^(0) = (0, ... ,0) for which phi_a^(0)(U) = 1
   %                 The algorithm used is the one detailed in the following reference:
   %                 [3] C. Soize, Uncertainty Quantification. An Accelerated Course with Advanced Applications in Computational Engineering,
   %                     Interdisciplinary Applied Mathematics, doi: 10.1007/978-3-319-54339-0, Springer, New York,2017.
   %
   %---INPUT variables
   %         KU           = factorial(ng+ndeg)/(factorial(ng)*factorial(ndeg)) = number
   %                         of terms including a^(0) = (0, ... ,0) for which phi_a^(0)(U) = 1
   %         ng           = dimension of the germ U = (U_1, ... , U_ng) 
   %         ndeg         = max degree of the polynomial chaos  
   %         NnbMC0       = number of realizations
   %         MatRU(ng,NnbMC0) = NnbMC0 independent realizations of normalized Gaussian random vector U = (U_1, ... , U_ng) 
   %
   %---OUTPUT variable
   %         MatRphiU(KU,NnbMC0) = MatRaU(KU,KU)*MatRMM(KU,NnbMC0)
   %         MatRaU(KU,KU) such that  MatRphiU(KU,NnbMC0) = MatRaU(KU,KU)*MatRMM(KU,NnbMC0)
   %         MatPowerU(KU,ng)
   %         NOTE: one has 1/(NnbMC0-1)*MatRphiU(KU,nbMC0)*MatRphiU(KU,nbMC0)' = [I_KU] with
   %               MatRphiU(KU,NnbMC0) = MatRaU(KU,KU)*MatRMM(KU,NnbMC0)


   %--- construction of MatPowerU(KU,ng): MatPowerU(k,:) = (alpha_1^(k), ... , alpha_ng^(k))
   %    which includes the multi-index (0,...,0) lacated at index  k = KU
   if ndeg == 0    
      MatRphiU  = ones(KU,NnbMC0);      % MatRphiU(KU,NnbMC0): it is (1,...,1) NnbMC0 times
      MatPowerU = eye(KU,ng);           % MatPowerU(KU,ng) 
      MatRaU    = eye(KU,KU);           % MatRaU(KU,KU)
   end
   if ndeg >= 1
      MatPowerU = eye(KU,ng); 
      Ind1 = 1; 
      Ind2 = ng ;
      I = ng;
      for p = 2:ndeg
          for L = 1:ng
              Rtest = zeros(1,ng);
              Rtest(L) = p-1;
              iL = Ind1;
              while (  ~isequal(MatPowerU(iL,:), Rtest) )
                  iL = iL + 1;
              end
              for k = iL:Ind2
                  I = I + 1;
                  MatPowerU(I, : ) = MatPowerU(L,:) + MatPowerU(k,:);   % MatPowerU(KU,ng)
              end
          end
          Ind1 = Ind2+1;
          Ind2 = I;
      end
   
      %--- Construction of monomials MatRMM(KU,NnbMC0) including  multi-index (0,...,0)
      MatRMM      = zeros(KU,NnbMC0);                         % MatRMM(KU,NnbMC0)
      MatRMM(1,:) = 1;      
      for k = 1:(KU-1)
          Rtemp = ones(1,NnbMC0);                             % Rtemp(1,NnbMC0);
          for j = 1:ng
              Rtemp = Rtemp.*MatRU(j,:).^MatPowerU(k,j);  % MatRU(ng,NnbMC0), MatPowerU(KU,ng)
          end
          MatRMM(k+1,:) = Rtemp;                              % MatRMM(KU,NnbMC0)
      end      

      %--- Construction of MatRphiU(KU,NnbMC0)
      MatRFF               = (MatRMM)*(MatRMM')/(NnbMC0-1);   % MatRFF(KU,KU), MatRMM(KU,NnbMC0)
      MatRFF               = 0.5*(MatRFF+MatRFF');            % MatRFF(KU,KU)
      % Removing the null space of MatRFF indice by the numerical noise
      MaxD = max(diag(MatRFF));
      tolrel = 1e-12;
      MatRFFtemp = MatRFF + tolrel*MaxD*eye(KU,KU);           % MatRFFtemp(KU,KU)
      % Chol decomposition
      [MatRLLtemp,indExecChol] = chol(MatRFFtemp,'lower');    % MatRLLtemp(KU,KU) is a lower triangular matrix    
      if indExecChol == 0
         MatRLL = sparse(MatRLLtemp);                         % MatRLL(KU,KU)
      else
         % matrix not positive definite
         error('STOP in sub_polynomialChaosQWU_chaosU: The matrix MatRFF must be positive definite');
      end
      MatRaU   = inv(MatRLL);                                 % MatRa(KU,KU): sparse matrix
      MatRphiU = full(MatRaU*MatRMM);                         % MatRphiU(KU,NnbMC0),MatRaU(KU,KU),MatRMM(KU,NnbMC0)
   end   
   return 
end

