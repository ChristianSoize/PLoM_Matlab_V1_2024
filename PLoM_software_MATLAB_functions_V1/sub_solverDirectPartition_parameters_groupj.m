function  [s,sh,shss,f0,Deltar,M0estim] = sub_solverDirectPartition_parameters_groupj(nu,n_d,icorrectif,f0_ref,ind_f0,coeffDeltar) 
   
   %--------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 2 July 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverDirectPartition_parameters_groupj.m
   %  Subject      : calculating and loading the parameters for groupj
   %  Comments     : this function is the adaptation of function sub_solverDirect_parameters.m
   %
   %--- INPUT  
   %        nu              : dimension or H = (H_1,...,H_nu)
   %        n_d             : number of realizations of H in the training dataset
   %        icorrectif      = 0: usual Silveman-bandwidth formulation for which the normalization conditions are not exactly satisfied
   %                        = 1: modified Silverman bandwidth implying that, for any value of nu, the normalization conditions are verified  
   %        f0_ref          : reference value (recommended value f0_ref = 4)
   %        ind_f0          : indicator for generating f0 (recommended value ind_f0 = 0): 
   %                          if ind_f0 = 0, then f0 = f0_ref, and if ind_f0 = 1, then f0 = f0_ref/sh    
   %        coeffDeltar     : coefficient > 0 (usual value is 20) for calculating Deltar
   %
   %--- OUTPUT
   %          s       : usual Silver bandwidth for the GKDE estimate (with the n_d points of the training dataset) 
   %                    of the pdf p_H of H, having to satisfy the normalization condition E{H} = 0_nu and E{H H'} = [I_nu] 
   %          sh      : modified Silver bandwidth for wich the normalization conditions are satisfied for any value of nu >= 1 
   %          shss    : = 1 if icorrectif  = 0, and = sh/s if icorrectif = 1
   %          f_0     : damping parameter in the ISDE, which controls the speed to reach the stationary response of the ISDE
   %          Deltar  : Stormer-Verlet integration-step of the ISDE        
   %          M0estim : estimate of M0transient provided as a reference to the user
   
   %--- Generation of parameters for the GKDE estimate, with the training dataset containing n_d points, of the pdf p_H of H,
   %    with H = (H_1,...,H_nu) having to satisfy the normalization condition E{H} = 0_nu and E{H H'} = [I_nu] 
   %    icorrectif = 0: usual Silveman-bandwidth formulation for which the normalization conditions are not exactly satisfied
   %               = 1: modified Silverman bandwidth implying that, for any value of nu, the normalization conditions are verified 
   
   s = ((4/((nu+2)*n_d))^(1/(nu+4)));                             % usual Silver bandwidth  
   s2 = s*s;   
   if icorrectif == 0                     
      shss = 1;
   end
   if icorrectif == 1
      shss = 1/sqrt(s2+(n_d-1)/n_d);                          
   end     
   sh = s*shss;    

   %--- Generation of the damping parameter f_0 in the ISDE, which controls the speed to reach the stationary response
   %    f0_ref:   reference value (recommended value f0_ref = 4)
   %    ind_f0:   indicator for generating f0 (recommended value ind_f0 = 0): 
   %              if ind_f0 = 0, then f0=f0_ref, and if ind_f0 = 1, then f0=f0_ref/sh  

   if ind_f0 == 0         
      f0 = f0_ref;        
   end
   if ind_f0 == 1
      f0 = f0_ref/sh;        
   end  
   
   %--- Generation of the Stormer-Verlet integration-step Deltar of the reduced-order ISDE.
   %    The positive coefficient, coeffDeltar, allows for calculating Deltar 
   %    An usual value for coeff_Deltar is 20

   Deltar = 2*pi*sh/coeffDeltar;  

   %--- Generation of user information related to the value of M0transient
   %    M0transient : the end-integration value, M0transient, at which the stationary response of the ISDE is reached
   %                  is given by the user. The corresponding final time at which the realization is extrated from 
   %                  solverDirect_Verlet is M0transient*Deltar
   %    M0estim     : estimate of M0transient provided as a reference to the user

   M0estim = 2*log(100)*coeffDeltar/(pi*f0*sh);    % estimated value of M0transient   
   return
end
   

 

 