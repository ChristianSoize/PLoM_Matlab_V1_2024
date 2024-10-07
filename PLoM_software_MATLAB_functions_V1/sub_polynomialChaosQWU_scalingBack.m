function [MatRxx_obs] =  sub_polynomialChaosQWU_scalingBack(nx_obs,n_x,n_ar,MatRx_obs,Indx_real,Indx_pos,Indx_obs,Rbeta_scale_real,Ralpha_scale_real, ...
                                        Rbeta_scale_log,Ralpha_scale_log,ind_display_screen,ind_print,ind_scaling)  

   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_scalingBack
   %  Subject      : constructing the n_ar realizations MatRxx_obs(nx_obs,n_ar) of the nx_obs <= n_x unscaled observarions XX_obs from the 
   %                 the n_ar  realizations MatRx_obs(nx_obs,n_ar) of the nx_obs scaled observations X_obs 
   %                 This is the inverse transform of the scaling done by the function sub_scaling1_main, restricted to the observations
   %
   %  Publications: 
   %               [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
   %                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
   %               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
   %                          American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
   %
   %  Function definition: 
   %                     Indx_real(nbreal,1) : contains the nbreal component numbers of X_ar and XX_ar that are real (positive, negative, 
   %                                           or zero) with 0 <= nbreal <=  n_x, and for which a "standard scaling" is used.
   %                     Indx_pos(nbpos,1)   : contains the nbpos component numbers of X_ar and XX_ar that are strictly positive, 
   %                                           with  0 <= nbpos <=  n_x , and for which the scaling {log + "standard scaling"} is used.
   %                     Indx_obs(nx_obs,1)  : contains the nx_obs components of XX_ar (unscaled) and X_ar (scaled) that are observed 
   %                                           with nx_obs <= n_x
   %
   %--- INPUTS
   %          nx_obs                      : dimension of random vectors XX_obs (unscaled) and X_obs (scaled)
   %          n_x                         : dimension of random vectors XX_ar  (unscaled) and X_ar (scaled)
   %          n_ar                        : number of points in the learning set for X_obs and XX_obs
   %          MatRx_obs(nx_obs,n_ar)      : n_ar realizations of X_obs
   %          Indx_real(nbreal,1)         : nbreal component numbers of XX_ar that are real (positive, negative, or zero) 
   %          Indx_pos(nbpos,1)           : nbpos component numbers of XX_ar that are strictly positive 
   %          Indx_obs(nx_obs,1)          : nx_obs component numbers of XX_ar that are observed with nx_obs <= n_x
   %          Rbeta_scale_real(nbreal,1)  : loaded if nbreal >= 1 or = [] if nbreal  = 0               
   %          Ralpha_scale_real(nbreal,1) : loaded if nbreal >= 1 or = [] if nbreal  = 0    
   %          Rbeta_scale_log(nbpos,1)    : loaded if nbpos >= 1  or = [] if nbpos = 0                 
   %          Ralpha_scale_log(nbpos,1)   : loaded if nbpos >= 1  or = [] if nbpos = 0   
   %          ind_display_screen          : = 0 no display, = 1 display
   %          ind_print                   : = 0 no print,   = 1 print
   %          ind_scaling                 : = 0 no scaling
   %                                      : = 1    scaling
   %
   %--- OUPUTS
   %          MatRxx_obs(nx_obs,n_ar)      : n_ar realizations of XX_obs
         
   %--- Checking input data and parameters concerning Indx_obs(nx_obs,1)
   if nx_obs > n_x
      error('STOP1 in sub_polynomialChaosQWU_scalingBack: nx_obs > n_x')
   end
   [nobstemp,nartemp] = size(MatRx_obs);                %  MatRx_obs(nx_obs,n_ar) 
   if nobstemp ~= nx_obs || nartemp ~= n_ar
      error('STOP2 in sub_polynomialChaosQWU_scalingBack: dimension errors in MatRx_obs(nx_obs,n_ar) ');
   end
   if length(Indx_obs) ~= length(unique(Indx_obs))
      error('STOP3 in sub_polynomialChaosQWU_scalingBack: there are repetitions in Indx_obs');   % There are repetitions in Indx_obs
   end
   if any(Indx_obs < 1) || any(Indx_obs > n_x)
      error('STOP4 in sub_polynomialChaosQWU_scalingBack: at least one  integer in Indx_obs is not within the valid range');  % At least one  integer in Indx_obs is not within the valid range.
   end
 
   %--- Loading 
   nbreal = size(Indx_real,1);    %  Indx_real(nbreal,1)
   nbpos  = size(Indx_pos,1);     %  Indx_pos(nbpos,1)  

   %--- Initialization
   nbpos_obs = 0;
   nbreal_obs = 0;
   
   if nbpos >= 1                                    % List_pos(nbpos_obs,1) contains the positive-valued observation numbers 
      List_pos             = zeros(nbpos,1);        % nbpos_obs  <= nbpos, nbpos_obs is unknown, then initialization: List_pos(nbpos,1) 
      Rbeta_scale_log_obs  = zeros(nbpos,1);        % Rbeta_scale_log_obs(nbpos_obs,1)
      Ralpha_scale_log_obs = zeros(nbpos,1);        % Ralpha_scale_log_obs(nbpos_obs,1) 
      for ipos = 1:nbpos
          jXpos = Indx_pos(ipos,1);                 % Indx_pos(nbpos,1)
          for iobs = 1:nx_obs
              if jXpos == Indx_obs(iobs)            % Indx_obs(nx_obs,1)
                 nbpos_obs = nbpos_obs + 1;
                 List_pos(nbpos_obs,1) = iobs;
                 Rbeta_scale_log_obs(nbpos_obs,1)  = Rbeta_scale_log(ipos,1);  % Rbeta_scale_log_obs(nbpos_obs,1)
                 Ralpha_scale_log_obs(nbpos_obs,1) = Ralpha_scale_log(ipos,1); % Ralpha_scale_log_obs(nbpos_obs,1)                 
              end
          end
      end
      if nbpos_obs < nbpos
          List_pos(nbpos_obs+1:nbpos)             = [];   % List_pos(nbpos_obs,1)  
          Rbeta_scale_log_obs(nbpos_obs+1:nbpos)  = [];   % Rbeta_scale_log_obs(nbpos_obs,1)
          Ralpha_scale_log_obs(nbpos_obs+1:nbpos) = [];   % Ralpha_scale_log_obs(nbpos_obs,1)        
      end
      MatRx_obs_log = MatRx_obs(List_pos,:);        % MatRx_obs_log(nbpos_obs,n_ar),MatRx_obs(nx_obs,n_ar),List_pos(nbpos_obs,1)   
   end

   if nbreal >= 1                                   % List_real(nbreal_obs,1) contains the real-valued observation numbers 
      List_real             = zeros(nbreal,1);      % nbreal_obs  <= nbreal, nbreal_obs is unknown, then initialization: List_real(nbreal,1) 
      Rbeta_scale_real_obs  = zeros(nbreal,1);      % Rbeta_scale_real_obs(nbreal_obs,1)
      Ralpha_scale_real_obs = zeros(nbreal,1);      % Ralpha_scale_real_obs(nbreal_obs,1) 
      for ireal = 1:nbreal
          jXreal = Indx_real(ireal,1);              % Indx_real(nbreal,1)
          for iobs = 1:nx_obs
              if jXreal == Indx_obs(iobs)           % Indx_obs(nx_obs,1)
                 nbreal_obs = nbreal_obs + 1;
                 List_real(nbreal_obs,1) = iobs; 
                 Rbeta_scale_real_obs(nbreal_obs,1)  = Rbeta_scale_real(ireal,1);  % Rbeta_scale_real_obs(nbreal_obs,1)
                 Ralpha_scale_real_obs(nbreal_obs,1) = Ralpha_scale_real(ireal,1); % Ralpha_scale_real_obs(nbreal_obs,1)                 
              end
          end
      end
      if nbreal_obs < nbreal
          List_real(nbreal_obs+1:nbreal)             = [];   % List_real(nbreal_obs,1)  
          Rbeta_scale_real_obs(nbreal_obs+1:nbreal)  = [];   % Rbeta_scale_real_obs(nbreal_obs,1)
          Ralpha_scale_real_obs(nbreal_obs+1:nbreal) = [];   % Ralpha_scale_real_obs(nbreal_obs,1)        
      end
      MatRx_obs_real = MatRx_obs(List_real,:);      % MatRx_obs_real(nbreal_obs,n_ar), MatRx_obs(nx_obs,n_ar),List_real(nbreal_obs,1)            
   end

   %--- Print
   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'n_x        = %9i \n ',n_x); 
      fprintf(fidlisting,'nbreal     = %9i \n ',nbreal); 
      fprintf(fidlisting,'nbpos      = %9i \n ',nbpos); 
      fprintf(fidlisting,'      \n ');  
      fprintf(fidlisting,'nx_obs      = %9i \n ',nx_obs); 
      fprintf(fidlisting,'nbreal_obs = %9i \n ',nbreal_obs); 
      fprintf(fidlisting,'nbpos_obs  = %9i \n ',nbpos_obs); 
      fprintf(fidlisting,'      \n ');  
      fclose(fidlisting); 
   end
  
   if nbreal_obs + nbpos_obs ~= nx_obs
      error('STOP5 in sub_polynomialChaosQWU_scalingBack: one must have nbreal_obs + nbpos_obs = nx_obs')  
   end

   %--- Scaling 
   if nbreal >= 1
      if ind_scaling == 0
         MatRxx_obs_real = MatRx_obs_real; 
      end
      if ind_scaling == 1
         [MatRxx_obs_real] = sub_scalingBack_standard(nbreal_obs,n_ar,MatRx_obs_real,Rbeta_scale_real_obs,Ralpha_scale_real_obs); 
      end
   end
   if nbpos >= 1 
      if ind_scaling == 0
         MatRxx_obs_log = MatRx_obs_log; 
      end
      if ind_scaling == 1
         [MatRxx_obs_log] = sub_scalingBack_standard(nbpos_obs,n_ar,MatRx_obs_log,Rbeta_scale_log_obs,Ralpha_scale_log_obs); 
      end
      MatRxx_obs_pos   = exp(MatRxx_obs_log);        % MatRxx_obs_pos(nbpos_obs,n_ar), MatRxx_pos_log(nbpos_obs,n_ar)
   end

   %--- Construction of MatRxx_obs(nx_obs,n_ar)
   MatRxx_obs = zeros(nx_obs,n_ar);
   if nbreal_obs >= 1
      MatRxx_obs(List_real,:) = MatRxx_obs_real;   % MatRxx_obs(nx_obs,n_ar),MatRxx_obs_real(nbreal_obs,n_ar),List_real(nbreal_obs,1)  
   end
   if nbpos_obs >= 1
      MatRxx_obs(List_pos,:)  = MatRxx_obs_pos;    % MatRxx_obs(nx_obs,n_ar),MatRxx_obs_pos(nbpos_obs,n_ar),List_pos(nbpos_obs,1)  
   end  
   return
end
      

