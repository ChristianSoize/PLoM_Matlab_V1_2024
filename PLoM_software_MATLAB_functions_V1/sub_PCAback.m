
function  [MatRx_obs] = sub_PCAback(n_x,n_d,nu,n_ar,nx_obs,MatRx_d,MatReta_ar,Indx_obs,RmuPCA,MatRVectPCA, ...
                                    ind_display_screen,ind_print)
        
   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_PCAback
   %  Subject      : back PCA; computing the n_ar realizations MatRx_obs(nx_obs,n_ar) of the scaled random observation  X_obs 
   %                 from the n_ar realizations MatReta_ar(nu,n_ar) of H_ar 
   %
   %  Publications: 
   %               [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
   %                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
   %               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
   %                          American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
   %
   %--- INPUTS
   %          n_x                   : dimension of random vector X_ar (scaled)
   %          n_d                   : number of points in the training set for XX_d and X_d  
   %          nu                    : order of the PCA reduction, which is the dimension of H_ar
   %          n_ar                  : number of realizations of H_ar and X_ar
   %          nx_obs                : number of observations extracted from X_ar  
   %          MatRx_d(n_x,n_d)      : n_d realizations of X_d (scaled)
   %          MatReta_ar(nu,n_ar)   : n_ar realizations of H_ar
   %          Indx_obs(nx_obs,1)    : nx_obs component numbers of X_ar that are observed with nx_obs <= n_x
   %          RmuPCA(nu,1)          : vector of PCA eigenvalues in descending order
   %          MatRVectPCA(n_x,nu)   : matrix of the PCA eigenvectors associated to the eigenvalues loaded in RmuPCA
   %          ind_display_screen    : = 0 no display,            = 1 display
   %          ind_print             : = 0 no print,              = 1 print
   %
   %--- OUPUTS   
   %          MatRx_obs(nx_obs,n_ar)
   
   if ind_display_screen == 1                              
      disp('--- beginning Task10_PCAback')
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task10_PCAback \n ');
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end

   TimeStartPCAback = tic; 
   
   %--- Checking input data and parameters 
   if nx_obs > n_x
      error('STOP1 in sub_PCAback: nx_obs must be less than or equal to n_x')
   end
   [nxtemp,ndtemp] = size(MatRx_d);                      %  MatRx_d(n_x,n_d) 
   if nxtemp ~= n_x || ndtemp ~= n_d
      error('STOP2 in sub_PCAback: dimensions of MatRx_d(n_x,n_d) are not correct')
   end
   [nutemp,nartemp] = size(MatReta_ar);                  %  MatReta_ar(nu,n_ar) 
   if nutemp ~= nu || nartemp ~= n_ar
      error('STOP3 in sub_PCAback: dimensions of MatReta_ar(nu,n_ar) are not correct')
   end
   nobstemp = size(Indx_obs,1);                       % Indx_obs(nx_obs,1) 
   if nobstemp ~= nx_obs 
      error('STOP4 in sub_PCAback: dimensions of Indx_obs(nx_obs,1) are not correct')
   end
   if length(Indx_obs) ~= length(unique(Indx_obs))
      error('STOP5 in sub_PCAback: there are repetitions in Indx_obs')     % There are repetitions in Indx_obs
   end
   if any(Indx_obs < 1) || any(Indx_obs > n_x)
      error('STOP6 in sub_PCAback: at least one integer in Indx_obs is not within range [1,n_x]')  
   end

   %--- Print
   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'n_x    = %9i \n ',n_x); 
      fprintf(fidlisting,'n_d    = %9i \n ',n_d);  
      fprintf(fidlisting,'nu     = %9i \n ',nu); 
      fprintf(fidlisting,'n_ar   = %9i \n ',n_ar);  
      fprintf(fidlisting,'nx_obs = %9i \n ',nx_obs);  
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting); 
   end
   
   %--- Computing MatRx_obs(nx_obs,n_ar)
   RXmean       = mean(MatRx_d,2);                                      % RXmean(n_x,1)
   MatRtemp     = MatRVectPCA*(diag(sqrt(RmuPCA)));                     % MatRtemp(n_x,nu)
   RXmeanx_obs  = RXmean(Indx_obs,1);                                   % RXmeanx_obs(nx_obs,1), Indx_obs(nx_obs,1)
   MatRtemp_obs = MatRtemp(Indx_obs,:);                                 % MatRtemp_obs(nx_obs,nu), MatRtemp(n_x,nu), Indx_obs(nx_obs,1)
   MatRx_obs    = repmat(RXmeanx_obs,1,n_ar) + MatRtemp_obs*MatReta_ar; % MatRx_obs(nx_obs,n_ar), MatRtemp_obs(nx_obs,nu), MatReta_ar(nu,n_ar)

   ElapsedTimePCAback = toc(TimeStartPCAback);   

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ----- Elapsed time for Task10_PCAback \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' Elapsed Time   =  %10.2f\n',ElapsedTimePCAback);   
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task10_PCAback')
   end    
   
   return
end
      