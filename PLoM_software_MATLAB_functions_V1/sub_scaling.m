function [MatRx_d,Rbeta_scale_real,Ralpha_scale_real,Ralpham1_scale_real, ...
                  Rbeta_scale_log,Ralpha_scale_log,Ralpham1_scale_log ] = ...
                                            sub_scaling(n_x,n_d,MatRxx_d,Indx_real,Indx_pos,ind_display_screen,ind_print,ind_scaling)  

   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_scaling
   %  Subject      : scaling the n_d independent realizations MatRxx_d(n_x,n_d) of XX_d in a scaled realizations MatRx_d(n_x,n_d) of X_d
   %
   %  Publications: 
   %               [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
   %                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
   %               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
   %                          American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
   %
   %  Function definition: 
   %                     Indx_real(nbreal,1) : contains the nbreal component numbers that are real (positive, negative, or zero) 
   %                                              with 0 <= nbreal <=  n_x and for which a "standard scaling" is used
   %                     Indx_pos(nbpos,1)   : contains the nbpos component numbers that are strictly positive a "specific scaling"
   %                                              with  0 <= nbpos <=  n_x  and for which the scaling is {log + "standard scaling"}
   %                     we must have n_x = nbpos + nbreal
   %
   %--- INPUTS  
   %          n_x                   : dimension of random vector XX_d (unscale) and X_d (scale)
   %          n_d                   : number of points in the training set for XX_d and X_d
   %          MatRxx_d(n_x,n_d)     : n_d realizations of XX_d
   %          Indx_real(nbreal,1)   : nbreal component numbers that are real (positive, negative, or zero) 
   %          Indx_pos(nbpos,1)     : nbpos component numbers that are strictly positive 
   %          ind_display_screen    : = 0 no display, = 1 display
   %          ind_print             : = 0 no print,   = 1 print
   %          ind_scaling           : = 0 no scaling
   %                                : = 1    scaling
   %
   %--- OUPUTS
   %          MatRx_d(n_x,n_d)               : n_d realizations of X_d
   %          Rbeta_scale_real(nbreal,1)     : loaded if nbreal >= 1 or = [] if nbreal  = 0               
   %          Ralpha_scale_real(nbreal,1)    : loaded if nbreal >= 1 or = [] if nbreal  = 0   
   %          Ralpham1_scale_real(nbreal,1)  : loaded if nbreal >= 1 or = [] if nbreal  = 0   
   %          Rbeta_scale_log(nbpos,1)       : loaded if nbpos >= 1  or = [] if nbpos   = 0                 
   %          Ralpha_scale_log(nbpos,1)      : loaded if nbpos >= 1  or = [] if nbpos   = 0  
   %          Ralpham1_scale_log(nbpos,1)    : loaded if nbpos >= 1  or = [] if nbpos   = 0  
   %
   %--- COMMENTS 
   %            (1) The standard scaling is an affine transformation that maps in [0 , 1] 
   %            (2) if all the components of XX_d are scaled with the standard scaling, 
   %                then Indx_real =(1:n_x)', Indx_pos = [], and ind_scale is not used
    
   if ind_display_screen == 1   
      disp(' ')
      disp('--- beginning Task2_Scaling')
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task2_Scaling  \n ');
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end

   TimeStartScale = tic; 
     
   %--- checking input data and parameters
   if ind_scaling ~= 0 && ind_scaling ~= 1
      error('STOP1 in sub_scaling: ind_scaling must be equal to 0 or to 1');
   end
   [nxtemp,ndtemp] = size(MatRxx_d); %  MatRxx_d(n_x,n_d) 
   if nxtemp ~= n_x || ndtemp ~= n_d
      error('STOP2 in sub_scaling: dimensions n_x and n_d are not coherent for MatRxx_d(n_x,n_d)');
   end

   nbreal = size(Indx_real,1);    %  Indx_real(nbreal,1)
   nbpos  = size(Indx_pos,1);     %  Indx_pos(nbpos,1)  
   if nbreal + nbpos ~= n_x
      error('STOP3 in sub_scaling: nbreal + nbpos not equal to n_x');
   end 
   if nbreal >= 1                                                                
      if length(Indx_real) ~= length(unique(Indx_real))
         error('STOP4 in sub_scaling: there are repetitions in Indx_real');      % There are repetitions in Indx_real
      end
      if any(Indx_real < 1) || any(Indx_real > n_x)
         error('STOP5 in sub_scaling: at least one  integer in Indx_real is not within the valid range');  % At least one  integer in Indx_real is not within the valid range.
      end
   end
   if nbpos >= 1   
      if length(Indx_pos) ~= length(unique(Indx_pos))
         error('STOP6 in sub_scaling: there are repetitions in Indx_pos');               % There are repetitions in Indx_pos
      end
       if any(Indx_pos < 1) || any(Indx_pos > n_x)
         error('STOP7 in sub_scaling: at least one  integer in Indx_pos is not within the valid range'); % At least one  integer in Indx_pos is not within the valid range.
      end
   end
   if nbreal >= 1 && nbpos >= 1                            % Check that all integers in Indx_real are different from those in Indx_pos
     if isempty(intersect(Indx_real,Indx_pos))
        combined_list = [Indx_real; Indx_pos];       % Check that the union of both lists is exactly 1:n_x without missing or repetition
        if length(combined_list) == n_x && all(sort(combined_list) == (1:n_x)')
             ind_error = 0;                                % All integers in Indx_real are different from those in Indx_pos, and the 
                                                           % union is exactly all integers from 1 to n_x without missing or repetition
        else
             ind_error =  1;                               % The union of Indx_real and Indx_pos does not contain exactly all integers 
                                                           % from 1 to n_x without missing or repetition
        end
     else
             ind_error = 1;                                % There are common integers in Indx_real and Indx_pos
     end
     if ind_error == 1
         error(['STOP8 in sub_scaling: the union of Indx_real and Indx_pos does not contain exactly all integers ' ...
                'from 1 to n_x without missing or repetition; or there are common integers ' ...
                 'in Indx_real and Indx_pos']);
     end
   end

   if nbpos >= 1
      MatRxx_d_pos =  MatRxx_d(Indx_pos,:);                             % MatRxx_d_pos(nbpos,n_d), MatRxx_d(n_x,n_d)    
      threshold    = 1e-12 * max(max(MatRxx_d_pos));                    % threshold value
      if any(MatRxx_d_pos(:) <= 0) || any(MatRxx_d_pos(:) <= threshold) % There are real numbers in MatRxx_d_pos that are not strictly 
         error(['STOP9 in sub_scaling: there are real numbers in MatRxx_d_pos that are not strictly ' ...
                'positive or not larger than the threshold']); 
      end
   end
   if nbreal >= 1
      MatRxx_d_real =  MatRxx_d(Indx_real,:);                         % MatRxx_d_real(nbreal,n_d), MatRxx_d(n_x,n_d)  
   end

   %--- Print
   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'n_x         = %9i \n ',n_x); 
      fprintf(fidlisting,'nbreal      = %9i \n ',nbreal); 
      fprintf(fidlisting,'nbpos       = %9i \n ',nbpos);  
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'ind_scaling = %1i \n ',ind_scaling);
      fprintf(fidlisting,'      \n ');  
      fclose(fidlisting); 
   end

   %--- Initialization (the unused arrays are initialized with empty arrays to allow argument transfers in functions)
   Rbeta_scale_real     = [];                            
   Ralpha_scale_real    = [];  
   Ralpham1_scale_real  = [];  
   Rbeta_scale_log      = [];     
   Ralpha_scale_log     = []; 
   Ralpham1_scale_log   = []; 

   %--- Scaling    
   
  if nbreal >= 1  
     [MatRx_d_real,Rbeta_scale_real,Ralpha_scale_real,Ralpham1_scale_real] = sub_scaling_standard(nbreal,n_d,MatRxx_d_real);  
     if ind_scaling == 0
        MatRx_d_real = MatRxx_d_real;
     end
   end
   if nbpos >= 1
      MatRxx_d_log  = log(MatRxx_d_pos);      % MatRxx_d_log(nbpos,n_d),MatRxx_d_pos(nbpos,n_d)  
      [MatRx_d_log,Rbeta_scale_log,Ralpha_scale_log,Ralpham1_scale_log] = sub_scaling_standard(nbpos,n_d,MatRxx_d_log);  
      if ind_scaling == 0
         MatRx_d_log = MatRxx_d_log;
      end
   end

   %--- Construction of MatRx_d(n_x,n_d)
   MatRx_d = zeros(n_x,n_d);
   if nbreal >= 1
      MatRx_d(Indx_real,:) = MatRx_d_real;   % MatRx_d(n_x,n_d),MatRx_d_real(nbreal,n_d),Indx_real(nbreal,1)  
   end
   if nbpos >= 1
      MatRx_d(Indx_pos,:)  = MatRx_d_log;    % MatRx_d(n_x,n_d),MatRx_d_log(nbpos,n_d),Indx_pos(nbpos,1)  
   end   

   ElapsedTimeScale = toc(TimeStartScale);   

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ----- Elapsed time for Task2_Scaling \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' Elapsed Time   =  %10.2f\n',ElapsedTimeScale);   
      fprintf(fidlisting,'      \n ');  
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task2_Scaling')
   end    
   
   return
end
      

