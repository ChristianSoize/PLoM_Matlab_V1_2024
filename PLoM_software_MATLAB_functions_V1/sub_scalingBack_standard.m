function [MatRxx] = sub_scalingBack_standard(nbx,nbsim,MatRx,Rbeta_scale,Ralpha_scale) 
               
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
   %
   % Subject:  
   %         For the standard scaling, back scaling: MatRxx from MatRx  (inversion of  function sub_scaling2_standard)   
   %         MatRxx = Ralpha_scale.*MatRx + repmat(Rbeta_scale,1,nbsim)  
   %
   %--- INPUTS
   %          nbx                     : dimension of random vector XX (unscaled) and X (scaled)
   %          nbsim                   : number of points in the training set for XX
   %          MatRx(nbx,nbsim)       : nbsim realizations of XX
   %          Rbeta_scale(nbx,1);                     
   %          Ralpha_scale(nbx,1);   
   %---OUTPUTS
   %          MatRxx(nbx,nbsim)        : nbsim realizations of 
      
   MatRxx = Ralpha_scale.*MatRx + repmat(Rbeta_scale,1,nbsim); %  MatRxx(nbx,nbsim),Ralpha_scale(nbx,1),MatRx(nbx,nbsim),Rbeta_scale(nbx,1)
   return
end
       
        

