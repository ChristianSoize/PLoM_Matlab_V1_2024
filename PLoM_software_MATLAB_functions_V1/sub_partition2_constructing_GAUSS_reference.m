
function [INDEPGaussmax,meanINDEPGauss,stdINDEPGauss,gust,meanMaxINDEPGauss,stdMaxINDEPGauss,numfig] = ...
                                             sub_partition2_constructing_GAUSS_reference(NKL,nr,MatRHexpGauss,ind_plot,ind_parallel,numfig)
   %
   % Copyright C. Soize 24 May 2024 
   %
   %---INPUT 
   %         NKL                   : dimension of random vector H
   %         nr                    : number of independent realizations of random vector H
   %         MatRHexpGauss(NKL,nr) : nr independent realizations of the random vector HGauss = (HGauss_1,...,HGauss_NKL)
   %         ind_plot              : = 0 no plot, = 1 plot
   %         ind_parallel          : = 0 no parallel computing, = 1 parallel computing
   %         numfig                : number of generated figures before executing this function
   %
   %---OUTPUT  
   %         INDEPGaussmax      : maximum observer on all the realizations of INDEPGauss 
   %         meanINDEPGauss     : mean value of INDEPGauss
   %         stdINDEPGauss      : std of INDEPGauss
   %         gust               : gust factor
   %         meanMaxINDEPGauss  : mean value of MaxINDEPGauss 
   %         stdMaxINDEPGauss   : std of MaxINDEPGauss 
   %         numfig             : number of generated figures after executing this function
   %
   %--- INTERNAL PARAMETERS
   %         r1 and r2          : indices to estimate the statistical independence criterion of HGauss_r1 with HGauss_r2
   %         INDEPGauss         : random variable whose realizations are INDEPr1r2Gauss with respect to 1 <= r1 < r2 <= NKL 
   %         MaxINDEPGauss      : random variable corresponding to the extreme values of INDEPGauss
   %
   %--- COMMENTS
   %            (1) The independent Gaussian solution is constructed to have the numerical reference of the
   %                independence criterion with the same number of realizations: nr
   %            (2) For each pair icouple = (r1, r2), INDEPr1r2Gauss = Sr1Gauss + Sr2Gauss - Sr1r2Gauss
   %                where Sr1Gauss is the entropy of HGauss_r1, Sr2Gauss is the entropy of HGauss_r2, and
   %                Sr1r2Gauss is the joint entropy of HGauss_r1 and HGauss_r2.
   %
   %--- METHOD 
   %
   %    For testing the independence of two normalized Gaussian random variables HGauss_r1 and HGauss_r2, the numerical criterion is
   %    based on the MUTUAL INFORMATION criterion: 
   %
   %    INDEPr1r2Gauss = Sr1Gauss + Sr2Gauss - Sr1r2Gauss > = 0
   %    
   %    in which Sr1Gauss, Sr2Gauss, and Sr1r2Gauss are the entropy of  HGauss_r1, HGauss_r2, and (HGauss_r1,HGauss_r2).
   %    As the Gaussian random vector (HGauss_r1,HGauss_r2) is normalized, we should have INDEPr1r2Gauss  = 0. However as the entropy are 
   %    estimated with a finite numer of realizations, we have INDEPr1r2Gauss > 0 (and not = 0).
   %    This positive value could thus used as a numerical reference for testing the independence of the non-Gaussian normalized random 
   %    variables H_r1 and H_r2. However as the entropy are estimated with a finite of realizations, we have INDEPr1r2Gauss > 0 (and not = 0). 
   %    This positive value is thus used as a numerical reference for testing the independence of the non-Gaussian normalized random 
   %    variables H_r1 and H_r2
   %  
   %    Let Z be the positive-valued random variable for which the np = NKL*(NKL-1)/2 realizations are z_p = RINDEPGauss(p) = INDEPr1r2Gauss 
   %    with 1 <= r1 < r2 <= NKL.  

   np = NKL*(NKL-1)/2;
   RINDEPGauss = zeros(np,1);              % RINDEPGauss(np,1)
   Indr1 = zeros(np,1);
   Indr2 = zeros(np,1);
   
   p = 0;
   for r1 = 1:NKL-1
       for r2 = r1+1:NKL 
           p = p + 1;
           Indr1(p) = r1;
           Indr2(p) = r2;
       end
   end 
   
   %--- Sequential computation
   if ind_parallel == 0
      for p = 1:np
          r1 = Indr1(p);
          r2 = Indr2(p);
          [INDEPr1r2Gauss] = sub_partition11_testINDEPr1r2GAUSS(NKL,nr,MatRHexpGauss,r1,r2); 
          RINDEPGauss(p)   = INDEPr1r2Gauss;
      end  
   end

   %--- Parallel computation
   if ind_parallel == 1
      parfor p = 1:np
          r1 = Indr1(p);
          r2 = Indr2(p);
          [INDEPr1r2Gauss] = sub_partition11_testINDEPr1r2GAUSS(NKL,nr,MatRHexpGauss,r1,r2); 
          RINDEPGauss(p)   = INDEPr1r2Gauss;
      end  
   end
   
   meanINDEPGauss = mean(RINDEPGauss);                    % empirical mean value of Z
   stdINDEPGauss  = std(RINDEPGauss);                     % empirical standard deviation Z
   INDEPGaussmax  = max(RINDEPGauss);                     % maximum on the realizations z_p for p=1,...,np
   RINDEPGauss0   = RINDEPGauss - meanINDEPGauss;         % centering 
   nup = 0;
   for p = 2:np                                           % number of upcrossings by 0
       if RINDEPGauss0(p)- RINDEPGauss0(p-1) > 0 && RINDEPGauss0(p)*RINDEPGauss0(p-1) < 0    % there is one upcrossing by zero
          nup = nup + 1;
       end
   end
   cons = sqrt(2*log(nup));
   gust = cons + 0.577/cons;
   meanMaxINDEPGauss = meanINDEPGauss + gust*stdINDEPGauss;  
   stdMaxINDEPGauss  = stdINDEPGauss*(pi/sqrt(6))/cons;    
   
   if ind_plot == 1
      h = figure;                                            %--- plot the trajectory
      plot(RINDEPGauss,'-b')
            title({['Graph of $i^\nu(G_{r_1},G_{r_2})$ as a function of the pair $(r_1,r_2)$']}, ...
                     'FontWeight','Normal','FontSize',16,'Interpreter','latex');                                       
            xlabel(['number of the pair $(r_1,r_2)$'],'FontSize',16,'Interpreter','latex');                                                                     
            ylabel(['$i^\nu(G_{r_1},G_{r_2})$'],'FontSize',16,'Interpreter','latex');                                                           
            numfig = numfig+1;                                                         
            saveas(h,['figure_PARTITION_',num2str(numfig),'_i-Gauss.fig'])
       close(h);                                                     
                                                             %--- plot the pdf in assuming the independence of earch pair (r1,r2)
       h = figure; 
       npoint = 1000;
       [Rpdf,Rp]  = ksdensity(RINDEPGauss,'npoints',npoint,'support','positive');     
              plot(Rp,Rpdf,'-b')
              title({['Graph of the pdf of $Z^\nu$ whose realizations are '] ; ...
                     ['$z^\nu_p = i^\nu(G_{r_1},G_{r_2})$ with $p = (r_1,r_2)$ (blue solid)']}, ...
                     'FontWeight','Normal','FontSize',16,'Interpreter','latex')                                        
              xlabel(['$z$'],'FontSize',16,'Interpreter','latex')                                                                
              ylabel(['$p_{Z^\nu}(z)$'],'FontSize',16,'Interpreter','latex') 
              numfig = numfig+1;  
              saveas(h,['figure_PARTITION_',num2str(numfig),'_pdf_i-Gauss.fig'])
       close(h);
   end
   return   
end
