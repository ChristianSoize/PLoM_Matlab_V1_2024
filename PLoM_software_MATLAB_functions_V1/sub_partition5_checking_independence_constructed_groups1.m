
function [tau] = sub_partition5_checking_independence_constructed_groups1(NKL,nr,ngroup,Igroup,MatIgroup,MatRHexp,MatRHexpGauss)

   % Copyright C. Soize 21 January 2016 
   
   % SUJECT
   %           Checking that the constructed groups are independent
   %
   % INPUTS
   %           NKL                   : dimension of random vector H
   %           nr                    : number of independent realizations of random vector H
   %           ngroup                : number of constructed independent groups  
   %           Igroup(ngroup)        : vector Igroup(ngroup,1), mj = Igroup(j),  mj is the number of components of Y^j = (H_jr1,... ,H_jrmj)  
   %           MatIgroup(ngroup,mmax): MatIgroup1(j,r) = rj, in which rj is the component of H in group j such that Y^j_r = H_jrj with 
   %                                   mmax = max_j mj for j = 1, ... , ngroup
   %           MatRHexp(NKL,nr)      : nr realizations of H = (H_1,...,H_NKL)
   %           MatRHexpGauss(NKL,nr) : nr independent realizations of HGauss = (HGauss_1,...,HGauss_NKL)
   %
   %--- OUPUT
   %           tau : rate

   %--- Constructing the reference entropy criterion with Gauss distribution
   [RlogpdfGAUSSch] = sub_partition9_log_ksdensity_mult(NKL,nr,NKL,nr,MatRHexpGauss,MatRHexpGauss); % RlogpdfGAUSSch(nr,1)     
   SGAUSSch         = - mean(RlogpdfGAUSSch);                                                       % entropy of HGAUSSgrgamma
   SGAUSSchgr       = 0;
   for j = 1:ngroup 
       mj                  = Igroup(j);                                                             % Igroup(ngroup)        
       Indgrj              = (MatIgroup(j,1:mj))';                                                  % MatIgroup(j,1:Igroup(j)), j = 1,...,ngroup    
       MatRHexpGaussgrj    = MatRHexpGauss(Indgrj,:);         
       [RlogpdfGAUSSchgrj] = sub_partition9_log_ksdensity_mult(NKL,nr,mj,nr,MatRHexpGaussgrj,MatRHexpGaussgrj); % RlogpdfGAUSSchgrj(nr,1)    
       SGAUSSchgrj         = - mean(RlogpdfGAUSSchgrj);                                             % entropy of HGAUSS
       SGAUSSchgr          = SGAUSSchgr + SGAUSSchgrj;
       clear Indgrj  MatRHexpGaussgrj RlogpdfGAUSSchgrj SGAUSSchgrj
   end
   INDEPGaussRefch = SGAUSSchgr - SGAUSSch; % INDEPGaussRefch should be equal to 0 because the groups are independent
                                            % as nr is small and is finite, INDEPGaussRefch is strictly positive and is chosen as the reference
   clear  MatRHexpGauss RlogpdfGAUSSch SGAUSSch SGAUSSchgr j mj 
   
   %--- Constructing the criterion INDEPch for checking the independence of the groups
   [Rlogpdfch] = sub_partition9_log_ksdensity_mult(NKL,nr,NKL,nr,MatRHexp,MatRHexp);                % Rlogpdfch(nr,1)     
   Sch         = - mean(Rlogpdfch);                                                                 % entropy of H
   Schgr       = 0;
   for j = 1:ngroup  
       mj             = Igroup(j);                                                                  % Igroup(ngroup)                                        
       Indgrj         = (MatIgroup(j,1:mj))';                                                       % MatIgroup(j,1:Igroup(j)), j = 1,...,ngroup    
       MatRHexpgrj    = MatRHexp(Indgrj,:);         
       [Rlogpdfchgrj] = sub_partition9_log_ksdensity_mult(NKL,nr,mj,nr,MatRHexpgrj,MatRHexpgrj);    % Rlogpdfchgrj(nr,1)    
       Schgrj         = - mean(Rlogpdfchgrj);                                                       % entropy of Y^j
       Schgr          = Schgr + Schgrj;
       clear Indgrj  MatRHexpgrj Rlogpdfchgrj Schgrj
   end
   INDEPch = Schgr - Sch;    
   clear  Rlogpdfch Sch Schgr j mj 
   
   %--- Constructing the rate
   tau = 0;
   if INDEPch > 1e-10 && INDEPGaussRefch > 1e-10
       tau = 1- INDEPch/INDEPGaussRefch;
   end
   return
end
   
  