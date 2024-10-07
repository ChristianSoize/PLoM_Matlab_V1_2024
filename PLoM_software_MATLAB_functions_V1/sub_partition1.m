function [ngroup,Igroup,MatIgroup,SAVERANDendPARTITION] = sub_partition1(nu,n_d,nref,MatReta_d,RINDEPref, ...
                                     SAVERANDstartPARTITION,ind_display_screen,ind_print,ind_plot,MatRHplot,MatRcoupleRHplot,ind_parallel)  

   %---------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 24 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_partition1
   %  Subject      : optimal partition in indepedent groups of random vector H from its n_d independent realizations in MatReta_d(nu,n_d)
   %
   %  Publications: 
   %               [1] C. Soize, Optimal partition in terms of independent random vectors of any non-Gaussian vector defined by 
   %                      a set of realizations,SIAM-ASA Journal on Uncertainty Quantification, 
   %                      doi: 10.1137/16M1062223, 5(1), 176-211 (2017).                 
   %               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
   %                      Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).
   %
   %  Function definition: Decomposition of
   %                       H    = (H_1,...,H_r,...,H_nu)    in ngroup subsets (groups) H^1,...,H^j,...,H^ngroup
   %                       H    = (H^1,...,H^j,...,H^ngroup) 
   %                       H^j  = (H_rj1,...,Hrjmj)         with j = 1,...,ngroup
   %                                                        with n1 + ... + nngroup = nu
   %
   %--- INPUTS
   %          nu                    : dimension of random vector H = (H_1, ... H_nu)
   %          n_d                   : number of points in the training set for H
   %          nref                  : first dimension of matrix RINDEPref(nref,1)
   %          MatReta_d(nu,n_d)     : n_d realizations of H
   %          RINDEPref(nref,1)     : contains the values of the mutual information for exploring the dependence of two components of H
   %                                  exemple: RINDEPref =(0.001:0.001:0.016)';  
   %          SAVERANDstartPARTITION: state of the random generator at the end of the PCA step
   %          ind_display_screen    : = 0 no display,            = 1 display
   %          ind_print             : = 0 no print,              = 1 print
   %          ind_plot              : = 0 no plot,               = 1 plot
   %          ind_parallel          : = 0 no parallel computing, = 1 parallel computing
   %          MatRHplot(1,:)        : list of components H_j of H for which the pdf are estimated and plotted 
   %                                  exemple 1: MatRHplot = [1 2 5]; plot the 3 pdfs of components 1,2, and 5
   %                                  exemple 2: MatRHplot = [];      no plot
   %                                 
   %          MatRcoupleRHplot(:,2) : list of pairs H_j - H_j' of components of H for which the joint pdf are estimated and plotted
   %                                  exemple 1: MatRcoupleRHplot = [1 2; 1 4; 8 9];  plot the 3 joint pdfs of pairs (1,2), (1,4), and (8,9) 
   %                                  exemple 2: MatRcoupleRHplot = [];               no plot
   %
   %--- OUPUTS
   %          ngroup                   : number of constructed independent groups  
   %          Igroup(ngroup,1)         : vector Igroup(ngroup,1), mj = Igroup(j),  mj is the number of components of Y^j = (H_jr1,... ,H_jrmj)  
   %          MatIgroup(ngroup,mmax)   : MatIgroup1(j,r) = rj, in which rj is the component of H in group j such that Y^j_r = H_jrj 
   %                                     with mmax = max_j mj for j = 1, ... , ngroup
   %          SAVERANDendPARTITION     : state of the random generator at the end of the function
   %
   %--- COMMENTS about the internal variables
   %
   %         nedge: number of edges in the graph
   %         MatPrint(nedge,5) such that MatPrint(edge,:) = [edge r1 r2 INDEPr1r2 INDEPGaussRef]  
   %         INDEPGaussRef: Numerical criterion for testing the independence of two normalized Gaussian random variables HGauss_r1 and 
   %                        HGauss_r2 by using the MUTUAL INFORMATION criterion. The random generator for randn is reinitialized 
   %                        for avoiding some variations on the statistical fluctuations in the computation of INDEPGaussRef with respect
   %                        to the different cases analyzed with iexec_TEST. Note that the independent realizations used for constructing 
   %                        INDEPGaussRef can be dependent of all the other random quantities constructed before this one.

   if ind_display_screen == 1                              
      disp('--- beginning Task4_Partition')
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task4_Partition \n ');
      fprintf(fidlisting,'      \n ');  
      fclose(fidlisting);  
   end

   TimeStartPartition = tic; 

   %--- initializing the random generator at the value of the end of the PCA step 
   rng(SAVERANDstartPARTITION);    

   %--- changing the name of nu,n_d, and MatReta for the partition functions                                           
   NKL      = nu;                                                                                          
   nr       = n_d;  
   MatRHexp = MatReta_d;    % MatRHexp(NKL,nr), MatReta_d(nu,n_d)

   %--- initialization of the number of generated figures (plot)
   numfig   = 0;
   
   %--- checking the input parameters
   [nu_temp, n_d_temp] = size(MatReta_d);                   % MatReta_d(nu,n_d)  
   if nu_temp ~= nu || n_d_temp ~= n_d
     error('STOP1 in sub_partition1_main')
   end
   [nref_temp,dim_temp] = size(RINDEPref);                  % RINDEPref(nref,1)
   if nref_temp ~= nref || dim_temp ~= 1
     error('STOP2 in sub_partition1_main')
   end
   [dim_temp,nRHplot_temp] = size(MatRHplot);               % MatRHplot(1,:)  
   if dim_temp ~= 1
     error('STOP3 in sub_partition1_main')
   end
   if max(MatRHplot)  > NKL || nRHplot_temp > NKL
     error('STOP4 in sub_partition1_main')
   end
   [ncoupleRHplot_temp,dim_temp] = size(MatRcoupleRHplot);  % MatRcoupleRHplot(:,2)
   if dim_temp ~= 2
     error('STOP5 in sub_partition1_main')
   end
   if max(max(MatRcoupleRHplot)) > NKL || ncoupleRHplot_temp > NKL^2
     error('STOP6 in sub_partition1_main')
   end
  
   %==================================================================================================================================  
   %            construction of the Gauss reference for independent group
   %==================================================================================================================================  

   if ind_display_screen == 1                              
      disp('--- beginning the construction of the Gauss reference for independent group')
   end

   MatRHexpGauss = randn(NKL,nr);   % (NKL,nr) matrix of the nr independent realizations of HGauss = (HGauss_1,...,HGauss_NKL)
                                   
   %--- Constructing the optimal value INDEPopt such that the rate tau(INDEPref) is maximum
   [INDEPGaussmax,meanINDEPGauss,stdINDEPGauss,gust,meanMaxINDEPGauss,stdMaxINDEPGauss,numfig] = ...
                                        sub_partition2_constructing_GAUSS_reference(NKL,nr,MatRHexpGauss,ind_plot,ind_parallel,numfig); 
   if ind_display_screen == 1   
      disp('--- end of the construction of the Gauss reference for independent group') 
   end

   %==================================================================================================================================  
   %           optimization loop for independent group
   %==================================================================================================================================  

   if ind_display_screen == 1   
      disp('--- beginning the optimization loop for independent group')  
   end

   %--- sequential computing
   if ind_parallel == 0
      Rtau = zeros(nref,1);
      for iref = 1:nref       
          % [iref nref]
          INDEPref = RINDEPref(iref); 

          % Construction of groups corresponding to the value INDEPref
          [ngroup,Igroup,~,MatIgroup] = sub_partition3_find_groups1(NKL,nr,MatRHexp,INDEPref,ind_parallel); 
      
          % Rate of independence of the constructed partition     
          [tau] = sub_partition5_checking_independence_constructed_groups1(NKL,nr,ngroup,Igroup,MatIgroup,MatRHexp,MatRHexpGauss);  
          Rtau(iref) = tau;
      end
   end

   %--- parallel computing
   if ind_parallel == 1
      Rtau = zeros(nref,1);
      parfor iref = 1:nref
          % [iref nref]
          INDEPref = RINDEPref(iref); 

          % Construction of groups corresponding to the value INDEPref
          [ngroup,Igroup,~,MatIgroup] = sub_partition3_find_groups1(NKL,nr,MatRHexp,INDEPref,ind_parallel); 

          % Rate of independence of the constructed partition     
          [tau] = sub_partition5_checking_independence_constructed_groups1(NKL,nr,ngroup,Igroup,MatIgroup,MatRHexp,MatRHexpGauss);  
          Rtau(iref) = tau;
      end
   end

   if ind_display_screen == 1    
      disp('--- end of the optimization loop for independent group')
   end
   
   if ind_plot == 1
      h = figure; 
      plot(RINDEPref,Rtau,'-ob')
           title({['Graph of the rate $\tau (i_{\rm{ref}})$ of mutual information'] ; ...
                  ['for the partition obtained with the level $i_{\rm{ref}}$']}, ...
                  'FontWeight','Normal','FontSize',16,'Interpreter','latex');
           xlabel(['$i_{\rm{ref}}$'],'FontSize',16,'Interpreter','latex');                                                                 
           ylabel(['$\tau (i_{\rm{ref}})$'],'FontSize',16,'Interpreter','latex');  
           numfig = numfig + 1;
           saveas(h,['figure_PARTITION_',num2str(numfig),'_tau.fig']) 
      close(h);
   end
   
   %==================================================================================================================================  
   %           construction of the independent groups
   %==================================================================================================================================  

   if ind_display_screen == 1    
      disp('--- beginning the construction of the independent groups')
   end
   
   %--- Calculation of irefopt, tauopt, and INDEPopt 
   [tauoptMin,irefoptMin] = min(Rtau); 
   [tauoptMax,irefoptMax] = max(Rtau); 

   %--- General case: all the values of tau are positive or equal to zero and the optimal value corresponds to the maximum value of tau
   if tauoptMin >=0 || ((tauoptMin < 0 && tauoptMax > 0) && (irefoptMin < irefoptMax )) 
      irefopt  = irefoptMax;                                                            
      tauopt   = tauoptMax;
      INDEPopt = RINDEPref(irefopt);                 
   end

   %--- Particular case: there exits tau < 0; in this case all the components of H are independent and are Gaussian 
   if tauoptMax < 0 || ( (tauoptMin < 0 && tauoptMax > 0) && (irefoptMin > irefoptMax )) 
      irefopt  = irefoptMin;                                                                            
      tauopt   = tauoptMin;
      INDEPopt = RINDEPref(irefopt);
   end
                                   
   %--- Constructing the groups corresponding to the optimal value INDEPopt of INDEPref
   [ngroup,Igroup,~,MatIgroup,nedge,nindep,npair,MatPrintEdge,MatPrintIndep,MatPrintPair,RplotPair] =  ...
                                                                sub_partition4_find_groups2(NKL,nr,MatRHexp,INDEPopt,ind_parallel); 
   if ind_display_screen == 1   
      disp('--- end of the construction of the independent groups')
   end

   %==================================================================================================================================  
   %           print, plot, and checking
   %==================================================================================================================================                                     

   Indic = 0;                       % if Indic = 0 at the end of the loop, then Rtau(iref) < or = tauopt for all iref
   for iref=1:nref
       if Rtau(iref) > tauopt
          Indic = 1;
       end
   end

   %--- Print and plot the groups
   if Indic == 0 && tauopt < 0     % then H_1,...,H_nu are mutually independent and Gaussian
      ngroup    = NKL; 
      Igroup    = ones(ngroup,1);
      MatIgroup = (1:1:ngroup)';
      [numfig]  = sub_partition7_print_plot_groups1(INDEPopt,ngroup,Igroup,MatIgroup,npair,RplotPair,ind_print,ind_plot,numfig); 

   else                            % then H_1,...,H_nu are mutually dependent and not Gaussian       

      [numfig] = sub_partition8_print_plot_groups2(INDEPopt,ngroup,Igroup,MatIgroup,nedge,MatPrintEdge,nindep,MatPrintIndep,npair, ...
                                                   MatPrintPair,RplotPair,ind_print,ind_plot,numfig); 
   end

   %--- Checking the independence of the constructed groups
   [INDEPGausscheck,INDEPcheck,tauopt] = sub_partition6_checking_independence_constructed_groups2(NKL,nr,ngroup,Igroup, ...
                                                                                                  MatIgroup,MatRHexp,MatRHexpGauss);  
   %--- Print
   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'--------------- SUMMARIZING THE OPTIMAL PARTITION CONSTRUCTED IN INDEPENDENT GROUPS');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'Optimal value INDEPopt of INDEPref used for constructing the optimal partition = %8.5i \n ',INDEPopt);
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'Optimal value tauopt of tau corresponding to the construction of the optimal partition = %8.5i \n ',tauopt); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'INDEPGaussmax     = %8.6f \n ',INDEPGaussmax); 
      fprintf(fidlisting,'meanINDEPGauss    = %8.6f \n ',meanINDEPGauss);
      fprintf(fidlisting,'stdINDEPGauss     = %8.6f \n ',stdINDEPGauss);
      fprintf(fidlisting,'gust              = %8.6f \n ',gust);
      fprintf(fidlisting,'meanMaxINDEPGauss = meanINDEPGauss + gust*stdINDEPGauss = %8.6f \n ',meanMaxINDEPGauss); 
      fprintf(fidlisting,'stdMaxINDEPGauss  = %8.6f \n ',stdMaxINDEPGauss);
      fprintf(fidlisting,'number of groups  = %7i \n ',ngroup);  
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'maximum number of nodes in the largest group                    = %7i \n ',max(Igroup));  
      fprintf(fidlisting,'mutual information for the identified decomposition INDEPcheck  = %8.6f \n ',INDEPcheck);  
      fprintf(fidlisting,'mutual information for a normalized Gaussian vector having the identified decomposition INDEPGausscheck = %8.6f \n ',INDEPGausscheck);
      fprintf(fidlisting,'the groups are independent if INDEPcheck < = INDEPGausscheck');    
      fprintf(fidlisting,'      \n ');  
      fclose(fidlisting); 
   end
                                              
   %--- Plot pdf of H_j and joint pdf of H_j -  H_j' for the training dataset
   if ind_plot == 1        
      
      % Estimation and plot of the pdf of component H_j
      nRHplot = size(MatRHplot,2);  % MatRHplot(1,nRHplot)
      if nRHplot >= 1         
         npoint = 100;                                               
         for jj = 1:nRHplot
              j = MatRHplot(1,jj);
              if j <= NKL       % if j > NKL, component does not exist and plot is skipped
                 h = figure;                                                                                                                      
                 [Rpdfj,Retaj] = ksdensity(MatRHexp(j,:),'npoints',npoint);
                 Rgaussj = normpdf(Retaj,0,1);   
                 plot(Retaj,Rpdfj,'-k',Retaj,Rgaussj,'--k')
                 title({['Graph of the training pdf (solid black) and its'] ; ...
                        ['Gaussian approximation (dashed black) for $H_{', num2str(j), '}$']}, ...
                        'FontWeight', 'Normal', 'FontSize', 16, 'Interpreter', 'latex');
                 xlabel(['$\eta_{',num2str(j),'}$'],'FontSize',16,'Interpreter','latex');                                                                
                 ylabel(['$p_{H_{',num2str(j),'}}(\eta_{',num2str(j),'})$'],'FontSize',16,'Interpreter','latex');  
                 numfig = numfig + 1;
                 saveas(h,['figure_PARTITION_',num2str(numfig),'_pdf_H',num2str(j),'.fig']);
                 close(h);
              end
         end
         clear Rpdfj Retaj Rgaussj
      end

      % Estimation and plot of the joint pdf of components H_j et H_j'
      ncoupleRHplot = size(MatRcoupleRHplot,1);
      if ncoupleRHplot >= 1
         npoint   = 100;
         npointp1 = npoint+1;     
         RHexpmin = zeros(NKL,1);               % lower and upper bounds of the support of each marginal pdf of H_j estimated with ksdensity
         RHexpmax = zeros(NKL,1);
         for j =1:NKL
             Rbid     = MatRHexp(j,:)';                    % Rbid(nr,1)
             [~,R] = ksdensity(Rbid, 'npoints',npoint); % ksdensity is only used for computing the bounds and the point sampling of each axis
             minR     = min(R);
             maxR     = max(R);
             s        = min(maxR,-minR);
             RHexpmin(j) = -1.5*s;                       
             RHexpmax(j) =  1.5*s; 
         end 
         clear Rbid R    
         
         for icouple=1:ncoupleRHplot 
             j1 = MatRcoupleRHplot(icouple,1);
             j2 = MatRcoupleRHplot(icouple,2);   
             if j1 <= NKL && j2 <= NKL                    % if j1 > NKL or j2 > NKL, component does not exist and plot is skipped
                D1 =(RHexpmax(j1)-RHexpmin(j1))/npoint;
                D2 =(RHexpmax(j2)-RHexpmin(j2))/npoint;   % MatR1(npointp1,npointp1),MatR2(npointp1,npointp1)
                [MatR1,MatR2] = meshgrid(RHexpmin(j1):D1:RHexpmax(j1),RHexpmin(j2):D2:RHexpmax(j2)); 
                MatR3H        = zeros(npointp1,npointp1);
                for mm =1:npointp1       
                    MatRR =[MatR1(mm,:)                   % MatR1(npointp1,npointp1),MatR2(npointp1,npointp1),MatRHexp(NKL,nr)
                            MatR2(mm,:)];                 % MatRR(2,npointp1)
      
                    MatRRHData = [ MatRHexp(j1,:)         % MatRRHData(2,nr)
                                   MatRHexp(j2,:)];                
                   [RpdfH] = sub_partition10_ksdensity_mult(NKL,nr,2,npointp1,MatRRHData,MatRR);       % RpdfH(npointp1,1) 
                    MatR3H(mm,:) = RpdfH';
                end            
                h = figure;   
                axis('square')
                % contourf(MatR1,MatR2,MatR3H)  
                surf(MatR1,MatR2,MatR3H,'facecolor','interp','edgecolor','none');
                % colormap(bone)
                colorbar;
                colormap(jet);
                view(2);
                title(['Training joint pdf of $H_{',num2str(j1),'}$ and $H_{',num2str(j2),'}$'],'FontWeight','Normal','FontSize',16,'Interpreter','latex'); 
                xlabel(['$\eta_{',num2str(j1),'}$'],'FontSize',16,'Interpreter','latex');                                                                
                ylabel(['$\eta_{',num2str(j2),'}$'],'FontSize',16,'Interpreter','latex');
                numfig = numfig + 1;        
                saveas(h,['figure_PARTITION_',num2str(numfig),'_H',num2str(j1),'_H',num2str(j2),'.fig']);  
                close(h);
             end
         end
      end           
   end                                %--- end ind_plot 

   SAVERANDendPARTITION = rng;
   ElapsedTimePartition = toc(TimeStartPartition);   

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'-------   Elapsed time for Task4_Partition \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'Elapsed Time   =  %10.2f\n',ElapsedTimePartition);   
      fprintf(fidlisting,'      \n ');   
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task4_Partition')
   end     
   
   return
end
      

