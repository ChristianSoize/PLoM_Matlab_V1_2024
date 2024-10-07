function sub_polynomial_chaosZWiener_plot_Har_HPCE(n_ar,nar_PCE,MatReta_ar,MatReta_PCE,nbplotHClouds,nbplotHpdf,nbplotHpdf2D, ...
                                      MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D,numfig)
        
   %------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 12 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_polynomial_chaosZWiener_plot_Har_HPCE
   %  Subject      : solver PLoM for direct predictions with or without the constraints of normalization fo H_ar
   %                 computation of n_ar lerned realizations MatReta_ar(nu,n_ar) of H_ar
   %
   %--- INPUTS    
   %          n_ar                              : number of realizations of H_ar such that n_ar  = nbMC x n_d
   %          nar_PCE                           : number of realizations of H_PCE such that nar_PCE  = nbMC_PCE x n_d
   %          MatReta_ar(nu,n_ar)               : n_ar realizations of H_ar   
   %          MatReta_PCE(nu,nar_PCE)           : nar_PCE realizations of H_PCE 
   %          nbplotHClouds                     : number of the 3 components numbers of H_ar for which the plot of the clouds are made   
   %          nbplotHpdf                        : number of the components numbers of H_ar for which the plot of the pdfs are made  
   %          nbplotHpdf2D                      : number of the 2 components numbers of H_ar for which the plot of joint pdfs are made
   %          MatRplotHClouds(nbplotHClouds,3)  : contains the 3 components numbers of H_ar for which the plot of the clouds are made
   %          MatRplotHpdf(1,nbplotHpdf)        : contains the components numbers of H_ar for which the plot of the pdfs are made 
   %          MatRplotHpdf2D(nbplotHpdf2D,2)    : contains the 2 components numbers of H_ar for which the plot of joint pdfs are made 
   %
   %--- INTERNAL PARAMETERS
   %          nu                                : dimension of random vector H = (H_1, ... H_nu)
   %          nbMC                              : number of realizations of (nu,n_d)-valued random matrix [H_ar]  
   
   %--- Plot the 3D clouds ell --> (H_{ar,ih}^ell,H_{ar,jh}^ell,H_{ar,kh}^ell)  and ell --> (H_{PCE,ih}^ell,H_{PCE,jh}^ell,H_{PCE,kh}^ell)  
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

          % plot clouds of H_ar
          h=figure; 
          axes1 = axes;
          view(axes1,[54 25]);
          hold(axes1,'on');
          xlim(axes1,[MINih-0.5 MAXih+0.5]);
          ylim(axes1,[MINjh-0.5 MAXjh+0.5]);
          zlim(axes1,[MINkh-0.5 MAXkh+0.5]);
          set(axes1,'FontSize',16);
          plot3(MatReta_ar(ih,:),MatReta_ar(jh,:),MatReta_ar(kh,:),'MarkerSize',2,'Marker','hexagram','LineStyle','none','Color',[0 0 1]);         
          title({[' clouds of $H_{\rm{ar}}$ with $n_{\rm{ar}} = $' , num2str(n_ar)]}, ...
                 'FontWeight','Normal','FontSize',16,'Interpreter','latex');      
          xlabel(['$H_{',num2str(ih),'}$'],'FontSize',16,'Interpreter','latex');                                                                 
          ylabel(['$H_{',num2str(jh),'}$'],'FontSize',16,'Interpreter','latex'); 
          zlabel(['$H_{',num2str(kh),'}$'],'FontSize',16,'Interpreter','latex');
          numfig = numfig+1;
          saveas(h,['figure_PolynomialChaosZWiener_',num2str(numfig),'_clouds_Har_',num2str(ih),'_',num2str(jh),'_',num2str(kh),'.fig']);  
          hold off
          close(h);

          % plot clouds of H_PCE
          h=figure; 
          axes1 = axes;
          view(axes1,[54 25]);
          hold(axes1,'on');
          xlim(axes1,[MINih-0.5 MAXih+0.5]);
          ylim(axes1,[MINjh-0.5 MAXjh+0.5]);
          zlim(axes1,[MINkh-0.5 MAXkh+0.5]);
          set(axes1,'FontSize',16);
          plot3(MatReta_PCE(ih,:),MatReta_PCE(jh,:),MatReta_PCE(kh,:),'MarkerSize',2,'Marker','hexagram','LineStyle','none','Color',[1 0 0]); 
          title({[' clouds of $H_{\rm{PCE}}$ with $n_{\rm{ar,PCE}} = $' , num2str(nar_PCE)]}, ...
                 'FontWeight','Normal','FontSize',16,'Interpreter','latex');      
          xlabel(['$H_{',num2str(ih),'}$'],'FontSize',16,'Interpreter','latex');                                                                 
          ylabel(['$H_{',num2str(jh),'}$'],'FontSize',16,'Interpreter','latex'); 
          zlabel(['$H_{',num2str(kh),'}$'],'FontSize',16,'Interpreter','latex');
          numfig = numfig+1;
          saveas(h,['figure_PolynomialChaosZWiener_',num2str(numfig),'_clouds_HPCE_',num2str(ih),'_',num2str(jh),'_',num2str(kh),'.fig']);  
          hold off
          close(h);
      end
   end

   %--- Generation and plot the pdf of H_{ar,ih} and H_{PCE,ih}   
   if nbplotHpdf >= 1                                    
      npoint = 200;                                                          % number of points for the pdf plot using ksdensity                                
      for iplot = 1:nbplotHpdf
          ih                = MatRplotHpdf(1,iplot);                         % MatRplotHpdf(1,nbplotHpdf)  
          [Rpdf_ar,Rh_ar]   = ksdensity(MatReta_ar(ih,:),'npoints',npoint);
          [Rpdf_PCE,Rh_PCE] = ksdensity(MatReta_PCE(ih,:),'npoints',npoint);  
          MIN = min(min(Rh_ar),min(Rh_PCE));
          MAX = max(max(Rh_ar),max(Rh_PCE));
          h = figure; 
          axes1 = axes;
          hold(axes1,'on');
          xlim(axes1,[MIN-1 MAX+1]);
          plot(Rh_ar,Rpdf_ar,'k-');
          plot(Rh_PCE,Rpdf_PCE,'LineStyle','-','LineWidth',1,'Color',[0 0 1]);  
          title({['$p_{H_{{\rm ar},',num2str(ih),'}}$ (black thin) with $n_{\rm{ar}} = $',num2str(n_ar)] ; ...
                 ['$p_{H_{\rm{PCE},',num2str(ih),'}}$ (blue thick) with $n_{\rm{ar,PCE}} = $',num2str(nar_PCE)]}, ...
                 'FontWeight','Normal','FontSize',16,'Interpreter','latex');
          xlabel(['$\eta_{',num2str(ih),'}$'],'FontSize',16,'Interpreter','latex');              
          ylabel(['$p_{H_',num2str(ih),'}(\eta_{',num2str(ih),'})$'],'FontSize',16,'Interpreter','latex'); 
          numfig = numfig + 1;
          saveas(h,['figure_PolynomialChaosZWiener_',num2str(numfig),'_pdf_Har_HPCE',num2str(ih),'.fig']);  
          hold off
          close(h);
      end
   end

   %--- Generation and plot the joint pdf of (H_{ar,ih},H_{ar,ih}) 
   if nbplotHpdf2D >= 1
      npoint = 100;      
      for iplot = 1:nbplotHpdf2D
          ih =  MatRplotHpdf2D(iplot,1);                                     % MatRplotHpdf2D(nbplotHpdf2D,2)  
          jh =  MatRplotHpdf2D(iplot,2);
          MINih = min(MatReta_ar(ih,:));
          MAXih = max(MatReta_ar(ih,:));
          MINjh = min(MatReta_ar(jh,:));
          MAXjh = max(MatReta_ar(jh,:));

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
          xlabel(['$\eta_{',num2str(ih),'}$'],'FontSize',16,'Interpreter','latex');   
          ylabel(['$\eta_{',num2str(jh),'}$'],'FontSize',16,'Interpreter','latex');   
          title({['Joint pdf of $H_{\rm{ar},',num2str(ih),'}$ with $H_{\rm{ar},',num2str(jh),'}$ for $n_{\rm{ar}} = $',num2str(n_ar)]}, ...
                  'FontWeight','Normal','FontSize',16,'Interpreter','latex');
          numfig = numfig + 1;
          saveas(h,['figure_PolynomialChaosZWiener_',num2str(numfig),'_joint_pdf_Har',num2str(ih),'_Har',num2str(jh),'.fig']); 
          close(h);
      end
   end

    %--- Generation and plot the joint pdf of (H_{PCE,ih},H_{PCE,ih}) 
   if nbplotHpdf2D  > 0
      npoint = 100;      
      for iplot = 1:nbplotHpdf2D
          ih =  MatRplotHpdf2D(iplot,1);                                     % MatRplotHpdf2D(nbplotHpdf2D,2)  
          jh =  MatRplotHpdf2D(iplot,2);
          MINih = min(MatReta_PCE(ih,:));
          MAXih = max(MatReta_PCE(ih,:));
          MINjh = min(MatReta_PCE(jh,:));
          MAXjh = max(MatReta_PCE(jh,:));

          % Compute the joint probability density function using mvksdensity
          [MatRx, MatRy] = meshgrid(linspace(MINih,MAXih,npoint), linspace(MINjh,MAXjh,npoint));
          MatRetaT = [MatReta_PCE(ih,:)', MatReta_PCE(jh,:)'];          % MatRetaT(nar_PCE,2)
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
          xlabel(['$\eta_{',num2str(ih),'}$'],'FontSize',16,'Interpreter','latex');   
          ylabel(['$\eta_{',num2str(jh),'}$'],'FontSize',16,'Interpreter','latex') ;  
          title({['Joint pdf of $H_{\rm{PCE},',num2str(ih),'}$ with $H_{\rm{PCE},',num2str(jh),'}$ for $n_{\rm{ar,PCE}} = $',num2str(nar_PCE)]}, ...
                  'FontWeight','Normal','FontSize',16,'Interpreter','latex');
          numfig = numfig + 1;
          saveas(h,['figure_PolynomialChaosZWiener_',num2str(numfig),'_joint_pdf_HPCE',num2str(ih),'_HPCE',num2str(jh),'.fig']); 
          close(h);
      end
   end
   return
end



   
 

 