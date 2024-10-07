function sub_solverDirect_plot_Hd_Har(n_d,n_ar,MatReta_d,MatReta_ar,nbplotHsamples,nbplotHClouds,nbplotHpdf,nbplotHpdf2D, ...
                                      MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D,numfig)

   %------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 24 May 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_solverDirect
   %  Subject      : solver PLoM for direct predictions with or without the constraints of normalization fo H_ar
   %                 computation of n_ar lerned realizations MatReta_ar(nu,n_ar) of H_ar
   %
   %--- INPUTS    
   %          n_d                               : number of points in the training set for H   
   %          n_ar                              : number of realizations of H_ar such that n_ar  = nbMC x n_d
   %          MatReta_d(nu,n_d)                 : n_d realizations of H   
   %          MatReta_ar(nu,n_ar)               : n_ar realizations of H_ar 
   %          nbplotHsamples                    : number of the components numbers of H_ar for which the plot of the realizations are made  
   %          nbplotHClouds                     : number of the 3 components numbers of H_ar for which the plot of the clouds are made   
   %          nbplotHpdf                        : number of the components numbers of H_d and H_ar for which the plot of the pdfs are made  
   %          nbplotHpdf2D                      : number of the 2 components numbers of H_d and H_ar for which the plot of joint pdfs are made
   %          MatRplotHsamples(1,nbplotHsamples): contains the components numbers of H_ar for which the plot of the realizations are made
   %          MatRplotHClouds(nbplotHClouds,3)  : contains the 3 components numbers of H_ar for which the plot of the clouds are made
   %          MatRplotHpdf(1,nbplotHpdf)        : contains the components numbers of H_d and H_ar for which the plot of the pdfs are made 
   %          MatRplotHpdf2D(nbplotHpdf2D,2)    : contains the 2 components numbers of H_d and H_ar for which the plot of joint pdfs are made 
   %
   %--- INTERNAL PARAMETERS
   %          nu                                : dimension of random vector H = (H_1, ... H_nu)
   %          nbMC                              : number of realizations of (nu,n_d)-valued random matrix [H_ar]  

   %--- Plot the 2D graphs ell --> H_{ar,ih}^ell    
   if nbplotHsamples >= 1 
      for iplot = 1:nbplotHsamples 
          ih = MatRplotHsamples(1,iplot);                    % MatRplotHsamples(1,nbplotHsamples)
          MatRih_ar = MatReta_ar(ih,:);                      % MatRih_ar(1,n_ar)
          h = figure; 
          plot((1:1:n_ar), MatRih_ar(1,:),'b-')
          title({['$H_{\rm{ar},',num2str(ih),'}$ with  $n_{\rm{ar}} = $', num2str(n_ar)]}, ...
                   'FontWeight','Normal','FontSize',16,'Interpreter','latex')
          xlabel(' $\ell$','FontSize',16,'Interpreter','latex') 
          ylabel(['$\eta^\ell_{\rm{ar},',num2str(ih),'}$'],'FontSize',16,'Interpreter','latex') 
          numfig = numfig + 1;
          saveas(h,['figure_SolverDirect_',num2str(numfig),'_eta_ar_',num2str(ih),'.fig']);  
          close(h);
      end
   end                                                
   
   %--- Plot the 3D clouds ell --> (H_{d,ih}^ell,H_{d,jh}^ell,H_{d,kh}^ell)  and  ell --> (H_{ar,ih}^ell,H_{ar,jh}^ell,H_{ar,kh}^ell) 
   if nbplotHClouds >= 1  
      for iplot = 1:nbplotHClouds
          ih = MatRplotHClouds(iplot,1);
          jh = MatRplotHClouds(iplot,2);
          kh = MatRplotHClouds(iplot,3); 
          MINih = min(MatReta_ar(ih,:));
          MAXih = max(MatReta_ar(ih,:));
          MINjh = min(MatReta_ar(jh,:));
          MAXjh = max(MatReta_ar(jh,:));
          MINkh = min(MatReta_ar(kh,:));
          MAXkh = max(MatReta_ar(kh,:));
          h=figure; 
          axes1 = axes;
          view(axes1,[54 25]);
          hold(axes1,'on');
          xlim(axes1,[MINih-0.5 MAXih+0.5]);
          ylim(axes1,[MINjh-0.5 MAXjh+0.5]);
          zlim(axes1,[MINkh-0.5 MAXkh+0.5]);
          set(axes1,'FontSize',16);
          plot3(MatReta_d(ih,:),MatReta_d(jh,:),MatReta_d(kh,:),'MarkerSize',2,'Marker','hexagram','LineStyle','none','Color',[0 0 1]);
          hold on
          plot3(MatReta_ar(ih,:),MatReta_ar(jh,:),MatReta_ar(kh,:),'MarkerSize',2,'Marker','hexagram','LineStyle','none','Color',[1 0 0]); 
          grid on;
          title({[' clouds $H_d$ with $n_d = $',num2str(n_d), ' and $H_{\rm{ar}}$ with $n_{\rm{ar}} = $' , num2str(n_ar)]}, ...
                 'FontWeight','Normal','FontSize',16,'Interpreter','latex');      
          xlabel(['$H_{',num2str(ih),'}$'],'FontSize',16,'Interpreter','latex')                                                                 
          ylabel(['$H_{',num2str(jh),'}$'],'FontSize',16,'Interpreter','latex') 
          zlabel(['$H_{',num2str(kh),'}$'],'FontSize',16,'Interpreter','latex')
          numfig = numfig+1;
          saveas(h,['figure_SolverDirect_',num2str(numfig),'_eta_clouds_',num2str(ih),'_',num2str(jh),'_',num2str(kh),'.fig']);  
          hold off
          close(h);
      end
   end

   %--- Generation and plot the pdf of H_{d,ih} and H_{ar,ih}   
   if nbplotHpdf >= 1                                    
      npoint = 200;                                         % number of points for the pdf plot using ksdensity                                
      for iplot = 1:nbplotHpdf
          ih              = MatRplotHpdf(1,iplot);          % MatRplotHpdf(1,nbplotHpdf)  
          [Rpdf_d,Rh_d]   = ksdensity(MatReta_d(ih,:),'npoints',npoint);
          [Rpdf_ar,Rh_ar] = ksdensity(MatReta_ar(ih,:),'npoints',npoint);  
          MIN = min(min(Rh_d),min(Rh_ar));
          MAX = max(max(Rh_d),max(Rh_ar));
          h = figure; 
          axes1 = axes;
          hold(axes1,'on');
          xlim(axes1,[MIN-1 MAX+1]);
          plot(Rh_d,Rpdf_d,'k-')
          plot(Rh_ar,Rpdf_ar,'LineStyle','-','LineWidth',1,'Color',[0 0 1]);  
          title({['$p_{H_{{\rm d},',num2str(ih),'}}$ (black thin) with $n_d = $',num2str(n_d)] ; ...
                 ['$p_{H_{\rm{ar},',num2str(ih),'}}$ (blue thick) with $n_{\rm{ar}} = $',num2str(n_ar)]}, ...
                 'FontWeight','Normal','FontSize',16,'Interpreter','latex');
          xlabel(['$\eta_{',num2str(ih),'}$'],'FontSize',16,'Interpreter','latex')              
          ylabel(['$p_{H_',num2str(ih),'}(\eta_{',num2str(ih),'})$'],'FontSize',16,'Interpreter','latex') 
          numfig = numfig + 1;
          saveas(h,['figure_SolverDirect_',num2str(numfig),'_pdf_H_d_H_ar',num2str(ih),'.fig']);  
          hold off
          close(h);
      end
   end

   %--- Generation and plot the joint pdf of (H_{d,ih},H_{d,jh}) 
   if nbplotHpdf2D >= 1
      npoint = 100;      
      for iplot = 1:nbplotHpdf2D
          ih =  MatRplotHpdf2D(iplot,1);                                     % MatRplotHpdf2D(nbplotHpdf2D,2)  
          jh =  MatRplotHpdf2D(iplot,2);
          MINih = min(MatReta_d(ih,:));
          MAXih = max(MatReta_d(ih,:));
          MINjh = min(MatReta_d(jh,:));
          MAXjh = max(MatReta_d(jh,:));
          coeff     = 0.2;
          deltaih   = MAXih - MINih;
          deltajh   = MAXjh - MINjh;
          MINih = MINih - coeff*deltaih;
          MAXih = MAXih + coeff*deltaih;
          MINjh = MINjh - coeff*deltajh;
          MAXjh = MAXjh + coeff*deltajh;

          % Compute the joint probability density function using mvksdensity
          [MatRx, MatRy] = meshgrid(linspace(MINih,MAXih,npoint), linspace(MINjh,MAXjh,npoint));
          MatRetaT = [MatReta_d(ih,:)', MatReta_d(jh,:)'];              % MatRetaT(n_d,2)
          MatRpts = [MatRx(:) MatRy(:)];                                % MatRpts(npoint*npoint,2),MatRx(npoint,npoint),MatRy(npoint,npoint)
          Rpdf = mvksdensity(MatRetaT,MatRpts);                         % Rpdf(npoint*npoint,1)
          
          % Reshape the computed PDF values for contour plot
          MatRpdf = reshape(Rpdf,npoint,npoint);                        % MatRpdf(npoint,npoint), Rpdf(npoint*npoint,1)
          
          % Plot the contours of the joint PDF
          h = figure;  
          surf(MatRx,MatRy,MatRpdf,'facecolor','interp','edgecolor','none');
          xlim([MINih MAXih]);
          ylim([MINjh MAXjh]);
          colorbar;
          colormap(jet);
          view(2);
          xlabel(['$\eta_{',num2str(ih),'}$'],'FontSize',16,'Interpreter','latex')   
          ylabel(['$\eta_{',num2str(jh),'}$'],'FontSize',16,'Interpreter','latex')   
          title({['Joint pdf of $H_{\rm{d},',num2str(ih),'}$ with $H_{\rm{d},',num2str(jh),'}$ for $n_d = $',num2str(n_d)]}, ...
                  'FontWeight','Normal','FontSize',16,'Interpreter','latex');
          numfig = numfig + 1;
          saveas(h,['figure_SolverDirect_',num2str(numfig),'_joint_pdf_Hd',num2str(ih),'_Hd',num2str(jh),'.fig']); 
          close(h);
      end
   end

    %--- Generation and plot the joint pdf of (H_{ar,ih},H_{ar,jh}) 
   if nbplotHpdf2D  > 0
      npoint = 100;      
      for iplot = 1:nbplotHpdf2D
          ih =  MatRplotHpdf2D(iplot,1);                                     % MatRplotHpdf2D(nbplotHpdf2D,2)  
          jh =  MatRplotHpdf2D(iplot,2);
          MINih = min(MatReta_ar(ih,:));
          MAXih = max(MatReta_ar(ih,:));
          MINjh = min(MatReta_ar(jh,:));
          MAXjh = max(MatReta_ar(jh,:));
          coeff     = 0.2;
          deltaih   = MAXih - MINih;
          deltajh   = MAXjh - MINjh;
          MINih = MINih - coeff*deltaih;
          MAXih = MAXih + coeff*deltaih;
          MINjh = MINjh - coeff*deltajh;
          MAXjh = MAXjh + coeff*deltajh;

          % Compute the joint probability density function using mvksdensity
          [MatRx, MatRy] = meshgrid(linspace(MINih,MAXih,npoint), linspace(MINjh,MAXjh,npoint));
          MatRetaT = [MatReta_ar(ih,:)', MatReta_ar(jh,:)'];            % MatRetaT(n_ar,2)
          MatRpts = [MatRx(:) MatRy(:)];                                % MatRpts(npoint*npoint,2),MatRx(npoint,npoint),MatRy(npoint,npoint)
          Rpdf = mvksdensity(MatRetaT,MatRpts);                         % Rpdf(npoint*npoint,1)
          
          % Reshape the computed PDF values for contour plot
          MatRpdf = reshape(Rpdf,npoint,npoint);                        % MatRpdf(npoint,npoint), Rpdf(npoint*npoint,1)
          
          % Plot the contours of the joint PDF
          h = figure;  
          surf(MatRx,MatRy,MatRpdf,'facecolor','interp','edgecolor','none');
          xlim([MINih MAXih]);
          ylim([MINjh MAXjh]);
          colorbar;
          colormap(jet);
          view(2);
          xlabel(['$\eta_{',num2str(ih),'}$'],'FontSize',16,'Interpreter','latex')   
          ylabel(['$\eta_{',num2str(jh),'}$'],'FontSize',16,'Interpreter','latex')   
          title({['Joint pdf of $H_{\rm{ar},',num2str(ih),'}$ with $H_{\rm{ar},',num2str(jh),'}$ for $n_{\rm{ar}} = $',num2str(n_ar)]}, ...
                  'FontWeight','Normal','FontSize',16,'Interpreter','latex');
          numfig = numfig + 1;
          saveas(h,['figure_SolverDirect_',num2str(numfig),'_joint_pdf_Har',num2str(ih),'_Har',num2str(jh),'.fig']); 
          close(h);
      end
   end
   return
end



   
 

 