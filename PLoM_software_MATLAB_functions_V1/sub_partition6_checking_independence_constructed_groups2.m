
function [INDEPGaussRefch,INDEPch,tau] = sub_partition6_checking_independence_constructed_groups2(NKL,nr,ngroup,Igroup, ...
                                                                                                  MatIgroup,MatRHexp,MatRHexpGauss)
   %
   % Copyright C. Soize 24 May 2024 
   %
   % Checking that the constructed groups are independent 

   %--- Constructing the reference entropy criterion with Gauss distribution
   [RlogpdfGAUSSch]  = sub_partition9_log_ksdensity_mult(NKL,nr,NKL,nr,MatRHexpGauss,MatRHexpGauss);  % RlogpdfGAUSSch(nr,1)     
   SGAUSSch = - mean(RlogpdfGAUSSch);                                                                 % entropy of HGAUSSgrgamma
   SGAUSSchgr = 0;
   for j = 1:ngroup 
       mj     = Igroup(j);                                                                % Igroup(ngroup)        
       Indgrj = (MatIgroup(j,1:mj))';                                                     % MatIgroup(j,1:Igroup(j)), j = 1,...,ngroup    
       MatRHexpGaussgrj = MatRHexpGauss(Indgrj,:);         
       [RlogpdfGAUSSchgrj]  = sub_partition9_log_ksdensity_mult(NKL,nr,mj,nr,MatRHexpGaussgrj,MatRHexpGaussgrj);  % RlogpdfGAUSSchgrj(nr,1)    
       SGAUSSchgrj = - mean(RlogpdfGAUSSchgrj);                                                                   % entropy of HGAUSS
       SGAUSSchgr = SGAUSSchgr + SGAUSSchgrj;
       clear Indgrj  MatRHexpGaussgrj RlogpdfGAUSSchgrj SGAUSSchgrj
   end
   INDEPGaussRefch = SGAUSSchgr - SGAUSSch;   % INDEPGaussRefch should be equal to 0 because the groups are independent
                                              % as nr is small and is finite,INDEPGaussRefch is strictly positive and is chosen as the reference
                                              
   clear  MatRHexpGauss RlogpdfGAUSSch SGAUSSch SGAUSSchgr j mj 
   
                                              %--- Constructing the criterion INDEPch for checking the independence of the groups

   [Rlogpdfch]  = sub_partition9_log_ksdensity_mult(NKL,nr,NKL,nr,MatRHexp,MatRHexp);    % Rlogpdfch(nr,1)     
   Sch = - mean(Rlogpdfch);                                                              % entropy of H
   Schgr = 0;
   for j = 1:ngroup 
       mj     = Igroup(j);                          % Igroup(ngroup)                                        
       Indgrj = (MatIgroup(j,1:mj))';               % MatIgroup(j,1:Igroup(j)), j = 1,...,ngroup    
       MatRHexpgrj = MatRHexp(Indgrj,:);         
       [Rlogpdfchgrj]  = sub_partition9_log_ksdensity_mult(NKL,nr,mj,nr,MatRHexpgrj,MatRHexpgrj);  % Rlogpdfchgrj(nr,1)    
       Schgrj = - mean(Rlogpdfchgrj);                                                              % entropy of Y^j
       Schgr = Schgr + Schgrj;
       clear Indgrj  MatRHexpgrj Rlogpdfchgrj Schgrj
   end
   INDEPch = Schgr - Sch; 
   
   clear  Rlogpdfch Sch Schgr j mj 
   
   tau = 0;
   if INDEPch > 1e-10 && INDEPGaussRefch > 1e-10
       tau = 1- INDEPch/INDEPGaussRefch;
   end
   
return                                                                        
