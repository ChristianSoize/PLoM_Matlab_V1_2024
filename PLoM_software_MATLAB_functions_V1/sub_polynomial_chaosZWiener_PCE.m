function [MatRPsi0] = sub_polynomial_chaosZWiener_PCE(K0,Ndeg,Ng,NnbMC0,MatRxiNg0)

   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 10 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_polynomial_chaosZWiener_PCE
   %  Subject      : Computing matrices: MatRPsi0(K0,NnbMC0),MatPower0(K0,Ng),MatRa0(K0,K0) of the polynomial chaos 
   %                     Psi_{alpha^(k)}(Xi) with alpha^(k) = (alpha_1^(k),...,alpha_Ng^(k)) in R^Ng with k=1,...,K0
   %                     Xi           = (Xi_1, ... , Xi_Ng) random vector for the germ
   %                     Ng           = dimension of the germ Xi = (Xi_1, ... , Xi_Ng)  
   %                     k            = 1,...,K0 indices of multi index (alpha_1^(k),...,alpha_Ng^(k)) of length Ng   
   %                     Ndeg         = max degree of the polynomial chaos   
   %                     K0           = factorial(Ng+Ndeg)/(factorial(Ng)*factorial(Ndeg)) number of terms
   %                                    including alpha^0 = (0, ... ,0) for which psi_alpha^0(Xi) = 1
   %                 The algorithm used is the one detailed in the following reference:
   %                 [3] C. Soize, Uncertainty Quantification. An Accelerated Course with Advanced Applications in Computational Engineering,
   %                     Interdisciplinary Applied Mathematics, doi: 10.1007/978-3-319-54339-0, Springer, New York,2017.
   %
   %---INPUT variables
   %         K0 = number of polynomial chaos including (0,...,0): K0 =  fix(1e-12 + factorial(Ng+Ndeg)/(factorial(Ng)*factorial(Ndeg))); 
   %         Ndeg       = maximum degree of polynomial chaos
   %         Ng         = dimension of the germ Xi = (Xi_1, ... , Xi_Ng)   
   %         NnbMC0     = number of samples
   %         MatRxiNg0(Ng,NnbMC0) = NnbMC0 independent realizations of normalized random vector Xi = (Xi_1, ... , Xi_Ng) 
   %
   %---OUTPUT variable
   %         MatRPsi0(K0,NbMC0) such that  MatRPsi0(K0,NnbMC0) = MatRa0(K0,K0)*MatRMM(K0,NnbMC0)
   %         NOTE: one has (1/(NnbMC0-1))*MatRPsi0(K0,NnbMC0)*MatRPsi0(K0,NnbMC0)' = [I_K0] 

   %--- Construction of MatPower0(K0,Ng): MatPower0(k,:) = (alpha_1^(k), ... , alpha_Ng^(k)),
   %    which includes the multi-index (0,...,0) lacated at index  k = K0
   if Ndeg == 0    
      MatRPsi0  = ones(K0,NnbMC0);      % MatRPsi0(K0,NnbMC0): it is (1,...,1) NnbMC0 times
   end
   if Ndeg >= 1
      MatPower0 = eye(K0,Ng); 
      Ind1 = 1; 
      Ind2 = Ng ;
      I = Ng;
      for p = 2:Ndeg
          for L = 1:Ng
              Rtest = zeros(1,Ng);
              Rtest(L) = p-1;
              iL = Ind1;
              while (  ~isequal(MatPower0(iL,:), Rtest) )
                  iL = iL + 1;
              end
              for k = iL:Ind2
                  I = I + 1;
                  MatPower0(I, : ) = MatPower0(L,:) + MatPower0(k,:);   % MatPower0(K0,Ng)
              end
          end
          Ind1 = Ind2+1;
          Ind2 = I;
      end
   
      %--- Construction of monomials MatRMM(K0,NnbMC0) including  multi-index (0,...,0)
      MatRMM      = zeros(K0,NnbMC0);                         % MatRMM(K0,NnbMC0)
      MatRMM(1,:) = 1;      
      for k = 1:(K0-1)
          Rtemp = ones(1,NnbMC0);                             % Rtemp(1,NnbMC0);
          for j = 1:Ng
              Rtemp = Rtemp.*MatRxiNg0(j,:).^MatPower0(k,j);  % MatRxiNg0(Ng,NnbMC0), MatPower0(K0,Ng)
          end
          MatRMM(k+1,:) = Rtemp;                              % MatRMM(K0,NnbMC0)
      end      

      %--- Construction of MatRPsi0(K0,NnbMC0)
      MatRFF               = (MatRMM)*(MatRMM')/(NnbMC0-1);   % MatRFF(K0,K0), MatRMM(K0,NnbMC0)
      MatRFF               = 0.5*(MatRFF+MatRFF');            % MatRFF(K0,K0)
      % Removing the null space of MatRFF indice by the numerical noise
      MaxD = max(diag(MatRFF));
      tolrel = 1e-12;
      MatRFFtemp = MatRFF + tolrel*MaxD*eye(K0,K0);           % MatRFFtemp(K0,K0)
      % Chol decomposition
      [MatRLLtemp,indExecChol] = chol(MatRFFtemp,'lower');    % MatRLLtemp(K0,K0) is a lower triangular matrix    
      if indExecChol == 0
         MatRLL = sparse(MatRLLtemp);                         % MatRLL(K0,K0)
      else
         % matrix not positive definite
         error('STOP in sub_polynomial_chaosZWiener_PCE: The matrix MatRFF must be positive definite');
      end
      MatRPsi0 = full(MatRLL\MatRMM);                         % MatRPsi0(K0,NnbMC0) 
   end   
   return 
end

