function [MatRx,Rbeta_scale,Ralpha_scale,Ralpham1_scale] = sub_scaling_standard(nbx,nbsim,MatRxx) 
               
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
   %
   % Subject:  
   %         For the standard scaling, compute Rbeta_scale(nbx,1),Ralpha_scale(nbx,1),Ralpham1_scale(nbx,1) for the scaling and the backscaling  
   %         MatRx  = Ralpham1_scale.*(MatRxx - repmat(Rbeta_scale,1,nbsim)) : scaling MatRxx in MatRx   
   %         MatRxx = Ralpha_scale.*MatRx + repmat(Rbeta_scale,1,nbsim)      : back scaling MatRx in MatRxx  
   %
   %--- INPUTS
   %          nbx                     : dimension of random vector XX (unscaled) and X (scaled)
   %          nbsim                   : number of realizations for XX
   %          MatRxx(nbx,nbsim)       : nbsim realizations of XX
   %---OUTPUTS
   %          MatRx(nbx,nbsim)        : nbsim realizations of X
   %          Rbeta_scale(nbx,1);                     
   %          Ralpha_scale(nbx,1);  
   %          Ralpham1_scale(nbx,1);
   
   Rmax           = max(MatRxx,[],2);                 %    MatRxx(nbx,nbsim)
   Rmin           = min(MatRxx,[],2);
   Rbeta_scale    = zeros(nbx,1);                     %    Rbeta_scale(nbx,1);
   Ralpha_scale   = zeros(nbx,1);                     %    Ralpha_scale(nbx,1);
   Ralpham1_scale = zeros(nbx,1);                     %    Ralpham1_scale(nbx,1);
   for k = 1:nbx 
       if Rmax(k)-Rmin(k) ~= 0
          Rbeta_scale(k)    = Rmin(k);
          Ralpha_scale(k)   = Rmax(k)-Rmin(k);
          Ralpham1_scale(k) = 1/Ralpha_scale(k);
       else
          Ralpha_scale(k)   = 1;
          Ralpham1_scale(k) = 1; 
          Rbeta_scale(k)    = Rmin(k);
       end
   end 
   MatRx  = Ralpham1_scale.*(MatRxx - repmat(Rbeta_scale,1,nbsim));
   return
end
       
        

