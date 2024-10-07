function  [nu,nnull,MatReta_d,RmuPCA,MatRVectPCA] = sub_PCA(n_x,n_d,MatRx_d,error_PCA,ind_display_screen,ind_print,ind_plot)
    
   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_PCA
   %  Subject      : PCA of the scaled random vector X_d using the n_d scaled realizations MatRx_d(n_x,n_d) of X_d 
   %
   %  Publications: 
   %               [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
   %                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
   %               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
   %                          American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
   %
   %--- INPUTS
   %          n_x                   : dimension of random vector X_d (scaled)
   %          n_d                   : number of realizations of X_d
   %          MatRx_d(n_x,n_d)      : n_d realizations of X_d
   %          error_PCA             : relative error on the mean-square norm (related to the eigenvalues of the covariance matrix of X_d)
   %                                  for the truncation of the PCA representation
   %          ind_display_screen    : = 0 no display,            = 1 display
   %          ind_print             : = 0 no print,              = 1 print
   %          ind_plot              : = 0 no plot,               = 1 plot
   %
   %--- OUPUTS   
   %          nu                    : order of the PCA reduction
   %          nnull                 : = n_x - nu dimension of the null space  
   %          MatReta_d(nu,n_d)     : n_d realizations of random vector H = (H_1,...,H_nu)  
   %          RmuPCA(nu,1)          : vector of eigenvalues in descending order
   %          MatRVectPCA(n_x,nu)   : matrix of the eigenvectors associated to the eigenvalues loaded in RmuPCA
   %
   
   if ind_display_screen == 1   
      disp(' ');
      disp('--- beginning Task3_PCA');
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task3_PCA   \n ');
      fprintf(fidlisting,'      \n ');  
      fclose(fidlisting);  
   end

   TimeStartPCA = tic; 
   numfig        = 0;   % initialization of the number of figures
   
   %--- Computating the trace of the estimated covariance matrix: traceMatRXcov = trace(MatRXcov);
   RXmean        = mean(MatRx_d,2);                                       % MatRx_d(n_x,n_d),RXmean(n_x,1)
   MatRXmean     = repmat(RXmean,1,n_d);                                  % MatRXmean(n_x,n_d)
   MatRtemp      = (MatRx_d - MatRXmean).^2;                              % MatRtemp(n_x,n_d)
   Rtemp         = sum(MatRtemp,2)/(n_d-1);                                 
   clear MatRtemp
   traceMatRXcov = sum(Rtemp);                                            % traceMatRXcov = trace(MatRXcov)
   clear Rtemp

   %---------------------------------------------------------------------------------------------------------------------------------------
   %       Case for which n_x <= n_d : construction of the estimated covariance matrix and solving the eigenvalue problem
   %---------------------------------------------------------------------------------------------------------------------------------------

   if n_x <= n_d  

      %--- Constructing the covariance matrix
      MatRXcov                  = cov(MatRx_d');                          % MatRXcov(n_x,n_x)
      MatRXcov                  = 0.5*(MatRXcov+MatRXcov');               % symmetrization 

      %--- Solving the eigenvalue problem
      [MatRVectTemp,MatRmuTemp] = eig(MatRXcov);                          % MatRVectTemp(n_x,n_x),MatRmuTemp(n_x,n_x)

      % Align the sign of each vector by ensuring the first element is positive
      for i = 1:size(MatRVectTemp,2)
          if MatRVectTemp(1,i) < 0
              MatRVectTemp(:,i) = -MatRVectTemp(:,i);
          end
      end
      
      RmuTemp                   = diag(MatRmuTemp);

      % Ordering the eigenvalues in descending order and replacing the values in RmuPCA that are less than 0 with 0
      [RmuPCA,Index] = sort(RmuTemp,'descend');                           % RmuPCA(n_x,1)
      RmuPCA(RmuPCA < 0) = 0;

      % Associate the ordering of eigenvectors with the ordering of the eigenvalues
      MatRVectPCA  = MatRVectTemp(:,Index);                               % MatRVectPCA(n_x,n_x)  

      % Find the indices where RerrPCA is less than 0 and replacing the values at those indices with RerrPCA(1) * 1e-14
      RerrPCA               = 1 - cumsum(RmuPCA,1)/traceMatRXcov;         % RerrPCA(n_x,1)
      Rneg_indices          = RerrPCA < 0;                                % Rneg_indices(n_x,1): logical array with 0 if > 0 and 1 if < 0
      RerrPCA(Rneg_indices) = RerrPCA(1) * 1e-14;
      
      %--- Print
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+');
         fprintf(fidlisting,'      \n ');  
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'RmuPCA =          \n '); 
         fprintf(fidlisting,' %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e \n ',RmuPCA');
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,' errPCA =          \n '); 
         fprintf(fidlisting,' %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e \n ',RerrPCA');
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n ');  
         fclose(fidlisting); 
      end

      %--- Plot
      if ind_plot == 1                                                 
         h = figure;                                                                                                                      
         semilogy((1:1:n_x)',RmuPCA(1:n_x,1),'-');                                                 
         title('Graph of the PCA eigenvalues in ${\rm{log}}_{10}$ scale','FontSize',16,'Interpreter','latex','FontWeight','normal');                                         
         xlabel(['$\alpha$'],'FontSize',16,'Interpreter','latex');                                                                
         ylabel(['$\mu(\alpha)$'],'FontSize',16,'Interpreter','latex');
         numfig = numfig + 1;
         saveas(h,['figure_PCA_',num2str(numfig),'_eigenvaluePCA.fig']); 
         hold off
         close(h);

         h = figure; 
         semilogy((1:1:n_x)',RerrPCA(1:n_x,1),'-'); 
         title('Graph of function $\rm{err}_{\rm{PCA}}$ in ${\rm{log}}_{10}$ scale','FontSize',16,'Interpreter','latex','FontWeight','normal');                                         
         xlabel(['$\alpha$'],'FontSize',16,'Interpreter','latex');                                                                
         ylabel(['$\rm{err}_{\rm{PCA}}(\alpha)$'],'FontSize',16,'Interpreter','latex');  
         numfig = numfig + 1;
         saveas(h,['figure_PCA_',num2str(numfig),'_errorPCA.fig']); 
         hold off
         close(h);
      end
      
      %--- Truncation of the PCA representation
      Ind = find(RerrPCA <= error_PCA);

      % Adapting the dimension of RmuPCA(nu,1) and MatRVectPCA(n_x,nu)
      if isempty(Ind) == 0                                                %    nu < n_x
         nu = n_x - size(Ind,1) + 1;
         if nu + 1 <= n_x
            RmuPCA(nu+1:n_x) = [];                                        %    RmuPCA(nu,1)
            MatRVectPCA(:,nu+1:n_x) = [];                                 %    MatRVectPCA(n_x,nu)
         end
      else                                                                %    nu = n_x
         nu = n_x;
      end  

      %--- Dimension of the null space
      nnull = n_x - nu;
   end

   %---------------------------------------------------------------------------------------------------------------------------------------
   %  Case for which n_x > n_d : the estimated covariance matrix is not constructed and eigenvalue problem is solved with a SVD on MatRx_d
   %---------------------------------------------------------------------------------------------------------------------------------------

   if n_x > n_d   

      %--- solving with a "thin SVD" without assembling the covariance matrix of X_d
      [MatRVectTemp,MatRSigma,~] = svd(MatRx_d - MatRXmean ,'econ');      % MatRVectTemp(n_x,n_d), MatRSigma(n_d,n_d) 

      % Align the sign of each vector by ensuring the first element is positive
      for i = 1:size(MatRVectTemp,2)
          if MatRVectTemp(1,i) < 0
              MatRVectTemp(:,i) = -MatRVectTemp(:,i);
          end
      end

      % Ordering the singular values in descending order 
      [RSigma,Index] = sort(diag(MatRSigma),'descend');                   % Rsigma(n_d,1)

      % Computing the eigenvalues in descending order
      RmuPCA      = (RSigma.^2)/(n_d-1);                                  % RmuPCA(n_d,1) 

      % Associate the ordering of eigenvectors with the ordering of the eigenvalues
      MatRVectPCA = MatRVectTemp(:,Index);                                % MatRVectPCA(n_x,n_d)  

      % Find the indices where RerrPCA is less than 0 and replacing the values at those indices with RerrPCA(1) * 1e-14
      RerrPCA               = 1 - cumsum(RmuPCA,1)/traceMatRXcov;         % RerrPCA(n_d,1)
      Rneg_indices          = RerrPCA < 0;                                % Rneg_indices(n_d,1): logical array with 0 if > 0 and 1 if < 0
      RerrPCA(Rneg_indices) = RerrPCA(1) * 1e-14;

      %--- Print
      if ind_print == 1
         fidlisting=fopen('listing.txt','a+');
         fprintf(fidlisting,'      \n ');  
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'RmuPCA =          \n '); 
         fprintf(fidlisting,' %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e \n ',RmuPCA');
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'errPCA =          \n '); 
         fprintf(fidlisting,' %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e \n ',RerrPCA');
         fprintf(fidlisting,'      \n '); 
         fprintf(fidlisting,'      \n ');  
         fclose(fidlisting); 
      end

      %--- Plot
      if ind_plot == 1                                                 
         h = figure;                                                                                                                      
         semilogy((1:1:n_d)',RmuPCA(1:n_d,1),'-');                                                 
         title('Graph of the PCA eigenvalues in ${\rm{log}}_{10}$ scale','FontSize',16,'Interpreter','latex','FontWeight','normal');                                         
         xlabel(['$\alpha$'],'FontSize',16,'Interpreter','latex');                                                                
         ylabel(['$\mu(\alpha)$'],'FontSize',16,'Interpreter','latex');
         numfig = numfig + 1;
         saveas(h,['figure_PCA_',num2str(numfig),'_eigenvaluePCA.fig']); 
         hold off
         close(h);

         h = figure; 
         semilogy((1:1:n_d)',RerrPCA(1:n_d,1),'-'); 
         title('Graph of function $\rm{err}_{\rm{PCA}}$ in ${\rm{log}}_{10}$ scale','FontSize',16,'Interpreter','latex','FontWeight','normal');                                         
         xlabel(['$\alpha$'],'FontSize',16,'Interpreter','latex');                                                                
         ylabel(['$\rm{err}_{\rm{PCA}}(\alpha)$'],'FontSize',16,'Interpreter','latex');  
         numfig = numfig + 1;
         saveas(h,['figure_PCA_',num2str(numfig),'_errorPCA.fig']); 
         hold off
         close(h);
      end
      
      %--- Truncation of the PCA representation
      Ind = find(RerrPCA <= error_PCA);   
      if isempty(Ind) == 0                                                % nu < n_d
         nu = n_d - size(Ind,1) + 1;
         if nu + 1 <= n_d
            RmuPCA(nu+1:n_d)        = [];                                 % RmuPCA(nu,1)
            MatRVectPCA(:,nu+1:n_d) = [];                                 % MatRVectPCA(n_x,nu)
         end
      else
         nu = n_d;
      end

      %--- Dimension of the null space
      nnull = n_d - nu;
   end  
   
   %---------------------------------------------------------------------------------------------------------------------------------------
   %             Computing  MatReta_d(nu,n_d) : n_d realizations of random vector H = (H_1,...,H_nu)  
   %---------------------------------------------------------------------------------------------------------------------------------------
                                                                          %--- Construction of  the samples of RH = (H_1,...,H_nu)
                                                                          %    MatReta_d(nu,n_d)
   Rcoef     = 1./sqrt(RmuPCA);
   MatRcoef  = diag(Rcoef);
   MatReta_d = MatRcoef*MatRVectPCA'*(MatRx_d - MatRXmean);
   
   %---------------------------------------------------------------------------------------------------------------------------------------
   %             Print PCA results  
   %---------------------------------------------------------------------------------------------------------------------------------------
   
   % Computing the L2 error and the second-order moments of X_d      
   MatRX_nu  = MatRXmean + MatRVectPCA*(diag(sqrt(RmuPCA)))*MatReta_d;       % MatRX_nu(n_x,n_d)
   error_nu  = norm(MatRx_d - MatRX_nu,'fro')/norm(MatRx_d,'fro'); 
   clear MatRX_nu
                                                                            
   fidlisting=fopen('listing.txt','a+');  
   fprintf(fidlisting,'      \n '); 
   fprintf(fidlisting,'      \n ');                     
   fprintf(fidlisting,'error_PCA                    = %14.7e \n ',error_PCA); 
   fprintf(fidlisting,'      \n ');  
   fprintf(fidlisting,'Number n_d of samples of X_d = %4i \n ',n_d); 
   fprintf(fidlisting,'Dimension n_x of X_d         = %4i \n ',n_x); 
   fprintf(fidlisting,'Dimension nu  of H           = %4i \n ',nu); 
   fprintf(fidlisting,'Null-space dimension         = %4i \n ',nnull); 
   fprintf(fidlisting,'      \n ');  
   fprintf(fidlisting,'L2 error error_nu            = %14.7e \n ',error_nu); 
   fprintf(fidlisting,'      \n '); 
   fclose(fidlisting); 

   if ind_plot == 1
      fidlisting=fopen('listing.txt','a+');  
      fprintf(fidlisting,'      \n ');                     
      fprintf(fidlisting,'RmuPCA =          \n '); 
      fprintf(fidlisting,' %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e \n ',RmuPCA');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting); 
   end

   ElapsedTimePCA = toc(TimeStartPCA);   

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ----- Elapsed time for Task3_PCA \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' Elapsed Time   =  %10.2f\n',ElapsedTimePCA);   
      fprintf(fidlisting,'      \n ');  
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task3_PCA');
      disp(' ');
   end    
   
   return
end
      