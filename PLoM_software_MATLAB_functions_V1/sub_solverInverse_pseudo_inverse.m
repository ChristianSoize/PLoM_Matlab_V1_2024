function [MatRinv] = sub_solverInverse_pseudo_inverse(MatR,eps_inv)  
        
   %------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 09 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverInverse_pseudo_inverse
   %  Subject      : compute the pseudo inverse MatRinv(n,n) of a positive definite matrix MatR(n,n): MatRinv = inv(MatR); 
   %
   % INPUT
   %      n                   
   %      MatR(n,n)
   %      eps_inv : tolerance for inverting the eigenvalue (for instance 1e-4)
   %
   % OUTPUT
   %      MatRinv(n,n)
   %
   % INTERNAL PARAMETER
   %      n
         MatRS       = 0.5*(MatR+MatR');
         [MatRPsiTemp,MatRxiTemp] = eig(MatRS);                 % MatRPsiTemp(n,n),MatRxiTemp(n,n),MatRS(n,n)
         RxiTemp     = diag(MatRxiTemp);                        % RxiTemp(n,1)
         [Rxi,Index] = sort(RxiTemp,'descend');                 % Rxi(n,1)         
         MatRPsi     = MatRPsiTemp(:,Index);                    % MatRPsi(n,n)   
         xiMin       = Rxi(1)*eps_inv;
         Ind         = find(Rxi >= xiMin);
         m           = size(Ind,1);                             % m is the number of eigenvalues greater than xiMin
         MatRPsim    = MatRPsi(:,1:m);                          % MatRPsim(n,m)
         Rxim        = Rxi(1:m);                                % Rxim(m,1)
         MatRinv     = (MatRPsim)*(diag(1./Rxim))*(MatRPsim');
         return
end