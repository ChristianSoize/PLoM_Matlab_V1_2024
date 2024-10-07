function sub_plot_Xd_Xar(n_x,n_q,n_w,n_d,n_ar,nu,MatRxx_d,MatRx_d,MatReta_ar,RmuPCA,MatRVectPCA,Indx_real,Indx_pos,nx_obs, ...
                         Indx_obs,ind_scaling,Rbeta_scale_real,Ralpha_scale_real,Rbeta_scale_log,Ralpha_scale_log, ...
                         MatRplotSamples,MatRplotClouds,MatRplotPDF,MatRplotPDF2D,ind_display_screen,ind_print)

   %------------------------------------------------------------------------------------------------------------------------------------------
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel,  une 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_plot_Xd_Xar
   %  Subject      : Plot a subset of MatRqq_obs(nq_obs,:) and MatRww_obs(nw_obs,:) for 
   %                 computation of n_ar lerned realizations MatReta_ar(nu,n_ar) of H_ar
   %
   %--- INPUTS 
   %
   %     n_x                         : dimension of random vectors XX_ar  (unscaled) and X_ar (scaled)  
   %     n_q                         : dimension of random vector QQ (unscaled quantitity of interest)  1 <= n_q    
   %     n_w                         : dimension of random vector WW (unscaled control variable) with 1 <= n_w  
   %     n_d                         : number of points in the training set for XX_d and X_d  
   %     n_ar                        : number of points in the learning set for H_ar, X_obs, and XX_obs
   %     nu                          : order of the PCA reduction, which is the dimension of H_ar   
   %     MatRxx_d(n_x,n_d)           : n_d realizations of XX_d (unscaled)
   %     MatRx_d(n_x,n_d)            : n_d realizations of X_d (scaled)
   %     MatReta_ar(nu,n_ar)         : n_ar realizations of H_ar 
   %     RmuPCA(nu,1)                : vector of PCA eigenvalues in descending order
   %     MatRVectPCA(n_x,nu)         : matrix of the PCA eigenvectors associated to the eigenvalues loaded in RmuPCA   %
   %     Indx_real(nbreal,1)         : nbreal component numbers of XX_ar that are real (positive, negative, or zero) 
   %     Indx_pos(nbpos,1)           : nbpos component numbers of XX_ar that are strictly positive 
   %     nx_obs                      : dimension of random vectors XX_obs (unscaled) and X_obs (scaled) (extracted from X_ar)  
   %     Indx_obs(nx_obs,1)          : nx_obs component numbers of X_ar and XX_ar that are observed with nx_obs <= n_x
   %     ind_scaling                 : = 0 no scaling
   %                                 : = 1    scaling
   %     Rbeta_scale_real(nbreal,1)  : loaded if nbreal >= 1 or = [] if nbreal  = 0               
   %     Ralpha_scale_real(nbreal,1) : loaded if nbreal >= 1 or = [] if nbreal  = 0    
   %     Rbeta_scale_log(nbpos,1)    : loaded if nbpos >= 1  or = [] if nbpos = 0                 
   %     Ralpha_scale_log(nbpos,1)   : loaded if nbpos >= 1  or = [] if nbpos = 0   

   %---- EXAMPLE OF DATA FOR EXPLAINING THE PLOT-DATA STRUCTURE
   %     component numbers of qq     = 1:100
   %     component numbers of ww     = 1:20
   %     Indq_obs(nq_obs,1)          = [2 4 6 8 80 98]', nq_obs = 6 
   %     Indw_obs(nw_obs,1)          = [1 3 8 15 17]'  , nw_obs = 5
   %     nx_obs                      = 6 + 5 = 11
   %     Indx_obs = [Indq_obs                     
   %                 n_q + Indw_obs] =  [2 4 6 8 80 98  101 103 108 115 117]'   

   %--- Plot data (illustration in examples using the above data)
   %
   %     MatRplotSamples(1,nbplotSamples): contains the components numbers of XX considered in XX_obs (plot the realizations)
   %                                       example 1: MatRplotSamples = [4 8 80 108];    plot, nbplotSamples = 4
   %                                       example 2: MatRplotSamples = [];              no plot, nbplotSamples = 0
   %     MatRplotClouds(nbplotClouds,3): contains the 3 components numbers of XX considered in XX_obs (plot the clouds)
   %                                       example 1: MatRplotClouds  = [101 108  8      plot for the 3 components 101, 108, 8 
   %                                                                      6  80 98 ];    plot for the 3 components   6, 80, 98
   %                                       example 2: MatRplotHClouds = [];              no plot, nbplotClouds = 0
   %     MatRplotPDF(1,nbplotPDF)      : contains the components numbers of XX_obs for which the plot of the PDFs are made 
   %                                       example 1: MatRplotPDF = [6 80 103 115];      plot for the nbplotPDF = 4 components 6, 80, 103, 115
   %                                       example 2: MatRplotPDF = [];                  no plot, nbplotPDF = 0
   %     MatRplotPDF2D(nbplotPDF2D,2)  : contains the 2 components numbers of XX_obs for which the plot of joint PDFs are made 
   %                                       example 1: MatRplotPDF2D = [2 103             plot for the 2 components 2 and 103 
   %                                                                   8  80];           plot for the 2 components 8 and 80
   %                                       example 2: MatRplotPDF2D = [];                no plot, nbplotHpdf2D = 0
   %
   %     ind_display_screen            : = 0 no display, = 1 display
   %     ind_print                     : = 0 no print,   = 1 print
   %
   %--- INTERNAL PARAMETERS
   %          nu                                : dimension of random vector H = (H_1, ... H_nu)
   %          nbMC                              : number of realizations of (nu,n_d)-valued random matrix [H_ar]  
   
   if ind_display_screen == 1                              
      disp('--- beginning Task12_PlotXdXar')
   end

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ------ Task12_PlotXdXar \n ');
      fprintf(fidlisting,'      \n ');   
      fclose(fidlisting);  
   end

   TimeStartPlotXdXar = tic; 
   numfig = 0;

   %--- Checking parameters and data
   if n_x <= 0 
       error('STOP1 in sub_plot_Xd_Xar: n_x <= 0');
   end  
   if n_q <= 0 || n_w < 0
       error('STOP2 in sub_plot_Xd_Xar: n_q <= 0 or n_w < 0');
   end
   nxtemp = n_q + n_w;                                                 % dimension of random vector XX = (QQ,WW)
   if nxtemp ~= n_x 
       error('STOP3 in sub_plot_Xd_Xar: n_x not equal to n_q + n_w');
   end
   if n_d <= 0 
       error('STOP4 in sub_plot_Xd_Xar: n_d <= 0');
   end
   if n_ar <= 0 
       error('STOP5 in sub_plot_Xd_Xar: n_ar <= 0');
   end
   if nu <= 0 || nu >= n_d
       error('STOP6 in sub_plot_Xd_Xar: nu <= 0 or nu >= n_d');
   end
   [n1temp,n2temp] = size(MatRxx_d);                 %  MatRxx_d(n_x,n_d) 
   if n1temp ~= n_x || n2temp ~= n_d
      error('STOP7 in sub_plot_Xd_Xar: dimension error in matrix MatRxx_d(n_x,n_d)');
   end
   [n1temp,n2temp] = size(MatRx_d);                  %  MatRx_d(n_x,n_d) 
   if n1temp ~= n_x || n2temp ~= n_d
      error('STOP8 in sub_plot_Xd_Xar: dimension error in matrix MatRx_d(n_x,n_d)');
   end
   [n1temp,n2temp] = size(MatReta_ar);               %  MatReta_ar(nu,n_ar) 
   if n1temp ~= nu || n2temp ~= n_ar
      error('STOP9 in sub_plot_Xd_Xar: dimension error in matrix MatReta_ar(nu,n_ar)');
   end
   [n1temp,n2temp] = size(RmuPCA);                   %  RmuPCA(nu,1) 
   if n1temp ~= nu || n2temp ~= 1
      error('STOP10 in sub_plot_Xd_Xar: dimension error in matrix RmuPCA(nu,1)');
   end
   [n1temp,n2temp] = size(MatRVectPCA);                   %  MatRVectPCA(n_x,nu)
   if n1temp ~= n_x || n2temp ~= nu
      error('STOP11 in sub_plot_Xd_Xar: dimension error in matrix MatRVectPCA(n_x,nu)');
   end

   nbreal = size(Indx_real,1);                           % Indx_real(nbreal,1) 
   if nbreal >= 1
      [n1temp,n2temp] = size(Indx_real);                  
      if n1temp ~= nbreal || n2temp ~= 1
         error('STOP12 in sub_plot_Xd_Xar: dimension error in matrix Indx_real(nbreal,1)');
      end
   end

   nbpos = size(Indx_pos,1);                             % Indx_pos(nbpos,1)
   if nbpos >= 1
      [n1temp,n2temp] = size(Indx_pos);                  
      if n1temp ~= nbpos || n2temp ~= 1
         error('STOP13 in sub_plot_Xd_Xar: dimension error in matrix Indx_pos(nbpos,1)');
      end
   end

   nxtemp = nbreal + nbpos;
   if nxtemp ~= n_x 
       error('STOP14 in sub_plot_Xd_Xar: n_x not equal to nreal + nbpos');
   end

   if nx_obs <= 0 
       error('STOP15 in sub_plot_Xd_Xar: nx_obs <= 0');
   end
   [n1temp,n2temp] = size(Indx_obs);                      % Indx_obs(nx_obs,1)                
   if n1temp ~= nx_obs || n2temp ~= 1
      error('STOP16 in sub_plot_Xd_Xar: dimension error in matrix Indx_obs(nx_obs,1)');
   end
   if ind_scaling ~= 0 && ind_scaling ~= 1
      error('STOP17 in sub_plot_Xd_Xar: ind_scaling must be equal to 0 or to 1');
   end
   if nbreal >= 1 
      [n1temp,n2temp] = size(Rbeta_scale_real);                   % Rbeta_scale_real(nbreal,1)              
      if n1temp ~= nbreal || n2temp ~= 1
         error('STOP18 in sub_plot_Xd_Xar: dimension error in matrix Rbeta_scale_real(nbreal,1) ');
      end
      [n1temp,n2temp] = size(Ralpha_scale_real);                   % Ralpha_scale_real(nbreal,1)              
      if n1temp ~= nbreal || n2temp ~= 1
         error('STOP19 in sub_plot_Xd_Xar: dimension error in matrix Ralpha_scale_real(nbreal,1) ');
      end                    
   end
   if nbpos >= 1 
      [n1temp,n2temp] = size(Rbeta_scale_log);                     % Rbeta_scale_log(nbpos,1)              
      if n1temp ~= nbpos || n2temp ~= 1
         error('STOP20 in sub_plot_Xd_Xar: dimension error in matrix Rbeta_scale_log(nbpos,1) ');
      end
      [n1temp,n2temp] = size(Ralpha_scale_log);                    % Ralpha_scale_log(nbpos,1)              
      if n1temp ~= nbpos || n2temp ~= 1
         error('STOP21 in sub_plot_Xd_Xar: dimension error in matrix Ralpha_scale_log(nbpos,1) ');
      end                    
   end

   nbplotSamples = size(MatRplotSamples,2);   % MatRplotSamples(1,nbplotSamples)
   nbplotClouds  = size(MatRplotClouds,1);    % MatRplotClouds(nbplotHClouds,3)
   nbplotPDF     = size(MatRplotPDF,2);       % MatRplotPDF(1,nbplotPDF)
   nbplotPDF2D   = size(MatRplotPDF2D,1);     % MatRplotPDF2D(nbplotPDF2D,2)

   if nbplotSamples >= 1                      % MatRplotSamples(1,nbplotSamples)
      n1temp = size(MatRplotSamples,1);
      if n1temp ~= 1 
         error('STOP22 in sub_plot_Xd_Xar: the first dimension of MatRplotSamples must be equal to 1') 
      end
      if any(MatRplotSamples(1,:) < 1) || any(MatRplotSamples(1,:) > n_x)   % at least one integer is not within the valid range
         error('STOP23 in sub_plot_Xd_Xar: at least one integer in MatRplotSamples(1,:) is not within range [1,n_x]') 
      end     
      isContained = all(ismember(MatRplotSamples(1,:),Indx_obs)); % Check if each integer in MatRplotSamples(1,:) belongs to Indx_obs
      if ~isContained
         error('STOP24 in sub_plot_Xd_Xar: one or more integers in MatRplotSamples(1,:) do not belong to Indx_obs ');
      end
   end
   if nbplotClouds >= 1                            % MatRplotClouds(nbplotClouds,3)
      n2temp = size(MatRplotClouds,2);
      if n2temp ~= 3
         error('STOP25 in sub_plot_Xd_Xar: the second dimension of MatRplotClouds must be equal to 3') 
      end
      if any(MatRplotClouds(:) < 1) || any(MatRplotClouds(:) > n_x)   % At least one integer is not within the valid range
         error('STOP26 in sub_plot_Xd_Xar: at least one integer of MatRplotClouds is not within range [1,n_x]')         
      end
      isContained = all(ismember(MatRplotClouds(:),Indx_obs)); % Check if each integer in MatRplotClouds belongs to Indx_obs
      if ~isContained
         error('STOP27 in sub_plot_Xd_Xar: one or more integers in MatRplotClouds do not belong to Indx_obs ');
      end
   end
   if nbplotPDF >= 1                               % MatRplotPDF(1,nbplotPDF)
      n1temp = size(MatRplotPDF,1);
      if n1temp ~= 1 
         error('STOP28 in sub_plot_Xd_Xar: the first dimension of MatRplotPDF must be equal to 1') 
      end
      if any(MatRplotPDF(1,:) < 1) || any(MatRplotPDF(1,:) > n_x) % at least one integer  is not within the valid range
         error('STOP29 in sub_plot_Xd_Xar: at least one integer in MatRplotPDF is not within range [1,n_x]')            
      end
      isContained = all(ismember(MatRplotPDF(1,:),Indx_obs)); % Check if each integer in MatRplotPDF(1,:) belongs to Indx_obs
      if ~isContained
         error('STOP30 in sub_plot_Xd_Xar: One or more integers in MatRplotPDF(1,:) do not belong to Indx_obs ');
      end
   end
   if nbplotPDF2D >= 1                             % MatRplotPDF2D(nbplotPDF2D,2)
      n2temp = size(MatRplotPDF2D,2);
      if n2temp ~= 2
         error('STOP31 in sub_plot_Xd_Xar: the second dimension of MatRplotPDF2D must be equal to 2') 
      end
      if any(MatRplotPDF2D(:) < 1) || any(MatRplotPDF2D(:) > n_x)  % at least one integer is not within the valid range
         error('STOP32 in sub_plot_Xd_Xar: at least one integer at least one integer in MatRplotPDF2D is not within range [1,n_x]')       
      end
      isContained = all(ismember(MatRplotPDF2D(:),Indx_obs)); % Check if each integer in MatRplotPDF2D belongs to Indx_obs
      if ~isContained
         error('STOP33 in sub_plot_Xd_Xar: one or more integers in MMatRplotPDF2D do not belong to Indx_obs ');
      end
   end

   %--- PCA back: MatRx_obs(nx_obs,n_ar)
   [MatRx_obs] = sub_PCAback(n_x,n_d,nu,n_ar,nx_obs,MatRx_d,MatReta_ar,Indx_obs,RmuPCA,MatRVectPCA, ...
                             ind_display_screen,ind_print);
   
   %--- Scaling back: MatRxx_obs(nx_obs,n_ar)
   [MatRxx_obs] = sub_scalingBack(nx_obs,n_x,n_ar,MatRx_obs,Indx_real,Indx_pos,Indx_obs,Rbeta_scale_real,Ralpha_scale_real, ...
                                  Rbeta_scale_log,Ralpha_scale_log,ind_display_screen,ind_print,ind_scaling); 

   %--- Plot the 2D graphs ell --> XX_{ar,ih}^ell    
   if nbplotSamples >= 1 
      for iplot = 1:nbplotSamples 
          ih        = MatRplotSamples(1,iplot);              % MatRplotSamples(1,nbplotSamples)
          MatRih_ar = MatRxx_obs(Indx_obs == ih,:);          % MatRih_ar(1,n_ar),Indx_obs(nx_obs,1),MatRxx_obs(nx_obs,n_ar)
          ihq = 0;
          ihw = 0;
          if ih <= n_q
             ihq = ih;
          end
          if ih > n_q
             ihw = ih - n_q;
          end
          h = figure; 
          plot((1:1:n_ar), MatRih_ar(1,:),'b-')
          if ihq >= 1
             title({['$Q_{\rm{ar},',num2str(ihq),'}$ with  $n_{\rm{ar}} = $', num2str(n_ar)]}, ...
                      'FontWeight','Normal','FontSize',16,'Interpreter','latex')
             xlabel(' $\ell$','FontSize',16,'Interpreter','latex') 
             ylabel(['$q^\ell_{\rm{ar},',num2str(ihq),'}$'],'FontSize',16,'Interpreter','latex') 
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_qar',num2str(ihq),'.fig']);  
             close(h);
          end
          if ihw >= 1
             title({['$W_{\rm{ar},',num2str(ihw),'}$ with  $n_{\rm{ar}} = $', num2str(n_ar)]}, ...
                      'FontWeight','Normal','FontSize',16,'Interpreter','latex')
             xlabel(' $\ell$','FontSize',16,'Interpreter','latex') 
             ylabel(['$w^\ell_{\rm{ar},',num2str(ihw),'}$'],'FontSize',16,'Interpreter','latex') 
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_war',num2str(ihw),'.fig']);  
             close(h);
          end
      end
   end                                                
   
   %--- Plot the 3D clouds ell --> XX_{d,ih}^ell,XX_{d,jh}^ell,XX_{d,kh}^ell)  and  ell --> (XX_{ar,ih}^ell,XX_{ar,jh}^ell,XX_{ar,kh}^ell) 
   if nbplotClouds >= 1  
      for iplot = 1:nbplotClouds
          ih        = MatRplotClouds(iplot,1);
          jh        = MatRplotClouds(iplot,2);
          kh        = MatRplotClouds(iplot,3); 
          MatRih_ar = MatRxx_obs(Indx_obs == ih,:);     % MatRih_ar(1,n_ar),Indx_obs(nx_obs,1),MatRxx_obs(nx_obs,n_ar)
          MatRjh_ar = MatRxx_obs(Indx_obs == jh,:);              
          MatRkh_ar = MatRxx_obs(Indx_obs == kh,:);              
          MINih     = min(MatRih_ar(1,:));
          MAXih     = max(MatRih_ar(1,:));
          MINjh     = min(MatRjh_ar(1,:));
          MAXjh     = max(MatRjh_ar(1,:));
          MINkh     = min(MatRkh_ar(1,:));
          MAXkh     = max(MatRkh_ar(1,:));
          if MINih >= 0; MINih = 0.6*MINih; end
          if MINih < 0;  MINih = 1.4*MINih; end
          if MAXih >= 0; MAXih = 1.4*MAXih; end
          if MAXih < 0;  MAXih = 0.6*MAXih; end
          if MINjh >= 0; MINjh = 0.6*MINjh; end
          if MINjh < 0;  MINjh = 1.4*MINjh; end
          if MAXjh >= 0; MAXjh = 1.4*MAXjh; end
          if MAXjh < 0;  MAXjh = 0.6*MAXjh; end
          if MINkh >= 0; MINkh = 0.6*MINkh; end
          if MINkh < 0;  MINkh = 1.4*MINkh; end
          if MAXkh >= 0; MAXkh = 1.4*MAXkh; end
          if MAXkh < 0;  MAXkh = 0.6*MAXkh; end
          MatRih_d  = MatRxx_d(ih,:);                 % MatRih_d(1,n_d),MatRxx_d(n_x,n_d)
          MatRjh_d  = MatRxx_d(jh,:);                 % MatRjh_d(1,n_d),MatRxx_d(n_x,n_d)
          MatRkh_d  = MatRxx_d(kh,:);                 % MatRkh_d(1,n_d),MatRxx_d(n_x,n_d)
          ihq = 0;
          jhq = 0;
          khq = 0;
          ihw = 0;
          jhw = 0;
          khw = 0;
          if ih <= n_q
             ihq = ih;
          end
          if ih > n_q
             ihw = ih - n_q;
          end 
          if jh <= n_q
             jhq = jh;
          end
          if jh > n_q
             jhw = jh - n_q;
          end 
          if kh <= n_q
             khq = kh;
          end
          if kh > n_q
             khw = kh - n_q;
          end 
          h=figure; 
          axes1 = axes;
          view(axes1,[54 25]);
          hold(axes1,'on');
          xlim(axes1,[MINih MAXih]);
          ylim(axes1,[MINjh MAXjh]);
          zlim(axes1,[MINkh MAXkh]);
          set(axes1,'FontSize',16);
          plot3(MatRih_d(1,:),MatRjh_d(1,:),MatRkh_d(1,:),'MarkerSize',2,'Marker','hexagram','LineStyle','none','Color',[0 0 1]);
          hold on
          plot3(MatRih_ar(1,:),MatRjh_ar(1,:),MatRkh_ar(1,:),'MarkerSize',2,'Marker','hexagram','LineStyle','none','Color',[1 0 0]); 
          grid on;
          title({[' clouds $X_d$ with $n_d = $',num2str(n_d), ' and $X_{\rm{ar}}$ with $n_{\rm{ar}} = $' , num2str(n_ar)]}, ...
                    'FontWeight','Normal','FontSize',16,'Interpreter','latex');  
          if ihq >= 1 && jhq >= 1 && khq >= 1
             xlabel(['$q_{',num2str(ihq),'}$'],'FontSize',16,'Interpreter','latex')                                                                 
             ylabel(['$q_{',num2str(jhq),'}$'],'FontSize',16,'Interpreter','latex') 
             zlabel(['$q_{',num2str(khq),'}$'],'FontSize',16,'Interpreter','latex')             
             numfig = numfig+1;
             saveas(h,['figure_PlotX',num2str(numfig),'_clouds_Q',num2str(ihq),'_Q',num2str(jhq),'_Q',num2str(khq),'.fig']);  
             hold off
             close(h);
          end
          if ihq >= 1 && jhq >= 1 && khw >= 1
             xlabel(['$q_{',num2str(ihq),'}$'],'FontSize',16,'Interpreter','latex')                                                                 
             ylabel(['$q_{',num2str(jhq),'}$'],'FontSize',16,'Interpreter','latex') 
             zlabel(['$w_{',num2str(khw),'}$'],'FontSize',16,'Interpreter','latex')             
             numfig = numfig+1;
             saveas(h,['figure_PlotX',num2str(numfig),'_clouds_Q',num2str(ihq),'_Q',num2str(jhq),'_W',num2str(khw),'.fig']);  
             hold off
             close(h);
          end
          if ihq >= 1 && jhw >= 1 && khq >= 1
             xlabel(['$q_{',num2str(ihq),'}$'],'FontSize',16,'Interpreter','latex')                                                                 
             ylabel(['$w_{',num2str(jhw),'}$'],'FontSize',16,'Interpreter','latex') 
             zlabel(['$q_{',num2str(khq),'}$'],'FontSize',16,'Interpreter','latex')             
             numfig = numfig+1;
             saveas(h,['figure_PlotX',num2str(numfig),'_clouds_Q',num2str(ihq),'_W',num2str(jhw),'_Q',num2str(khq),'.fig']);  
             hold off
             close(h);
          end
          if ihw >= 1 && jhq >= 1 && khq >= 1
             xlabel(['$w_{',num2str(ihw),'}$'],'FontSize',16,'Interpreter','latex')                                                                 
             ylabel(['$q_{',num2str(jhq),'}$'],'FontSize',16,'Interpreter','latex') 
             zlabel(['$q_{',num2str(khq),'}$'],'FontSize',16,'Interpreter','latex')             
             numfig = numfig+1;
             saveas(h,['figure_PlotX',num2str(numfig),'_clouds_W',num2str(ihw),'_Q',num2str(jhq),'_Q',num2str(khq),'.fig']);  
             hold off
             close(h);
          end
          if ihq >= 1 && jhw >= 1 && khw >= 1
             xlabel(['$q_{',num2str(ihq),'}$'],'FontSize',16,'Interpreter','latex')                                                                 
             ylabel(['$w_{',num2str(jhw),'}$'],'FontSize',16,'Interpreter','latex') 
             zlabel(['$w_{',num2str(khw),'}$'],'FontSize',16,'Interpreter','latex')             
             numfig = numfig+1;
             saveas(h,['figure_PlotX',num2str(numfig),'_clouds_Q',num2str(ihq),'_W',num2str(jhw),'_W',num2str(khw),'.fig']);  
             hold off
             close(h);
          end
          if ihw >= 1 && jhq >= 1 && khw >= 1
             xlabel(['$w_{',num2str(ihw),'}$'],'FontSize',16,'Interpreter','latex')                                                                 
             ylabel(['$q_{',num2str(jhq),'}$'],'FontSize',16,'Interpreter','latex') 
             zlabel(['$w_{',num2str(khw),'}$'],'FontSize',16,'Interpreter','latex')             
             numfig = numfig+1;
             saveas(h,['figure_PlotX',num2str(numfig),'_clouds_W',num2str(ihw),'_Q',num2str(jhq),'_W',num2str(khw),'.fig']);  
             hold off
             close(h);
          end
          if ihw >= 1 && jhw >= 1 && khq >= 1
             xlabel(['$w_{',num2str(ihw),'}$'],'FontSize',16,'Interpreter','latex')                                                                 
             ylabel(['$w_{',num2str(jhw),'}$'],'FontSize',16,'Interpreter','latex') 
             zlabel(['$q_{',num2str(khq),'}$'],'FontSize',16,'Interpreter','latex')             
             numfig = numfig+1;
             saveas(h,['figure_PlotX',num2str(numfig),'_clouds_W',num2str(ihw),'_W',num2str(jhw),'_Q',num2str(khq),'.fig']);  
             hold off
             close(h);
          end
          if ihw >= 1 && jhw >= 1 && khw >= 1
             xlabel(['$w_{',num2str(ihw),'}$'],'FontSize',16,'Interpreter','latex')                                                                 
             ylabel(['$w_{',num2str(jhw),'}$'],'FontSize',16,'Interpreter','latex') 
             zlabel(['$w_{',num2str(khw),'}$'],'FontSize',16,'Interpreter','latex')             
             numfig = numfig+1;
             saveas(h,['figure_PlotX',num2str(numfig),'_clouds_W',num2str(ihw),'_W',num2str(jhw),'_W',num2str(khw),'.fig']);  
             hold off
             close(h);
          end
      end
   end

   %--- Generation and plot the pdf of XX_{d,ih} and XX_{ar,ih}   
   if nbplotPDF >= 1                                    
      npoint = 200;                                          % number of points for the pdf plot using ksdensity                                
      for iplot = 1:nbplotPDF
          ih        = MatRplotPDF(1,iplot);                  % MatRplotPDF(1,nbplotPDF)
          MatRih_d  = MatRxx_d(ih,:);                        %  MatRih_d(1,n_d),MatRxx_d(n_x,n_d)
          MatRih_ar = MatRxx_obs(Indx_obs == ih,:);          % MatRih_ar(1,n_ar),Indx_obs(nx_obs,1),MatRxx_obs(nx_obs,n_ar)
          ihq = 0;
          ihw = 0;
          if ih <= n_q
             ihq = ih;
          end
          if ih > n_q
             ihw = ih - n_q;
          end         
          [Rpdf_d,Rh_d]   = ksdensity(MatRih_d(1,:),'npoints',npoint);
          [Rpdf_ar,Rh_ar] = ksdensity(MatRih_ar(1,:),'npoints',npoint);  
          MIN = min(min(Rh_d),min(Rh_ar));
          MAX = max(max(Rh_d),max(Rh_ar));
          h = figure; 
          axes1 = axes;
          hold(axes1,'on');
          xlim(axes1,[MIN-1 MAX+1]);
          plot(Rh_d,Rpdf_d,'k-')
          plot(Rh_ar,Rpdf_ar,'LineStyle','-','LineWidth',1,'Color',[0 0 1]);  
          if ihq >= 1
             title({['$p_{Q_{{\rm d},',num2str(ihq),'}}$ (black thin) with $n_d = $',num2str(n_d)] ; ...
                    ['$p_{Q_{\rm{ar},',num2str(ihq),'}}$ (blue thick) with $n_{\rm{ar}} = $',num2str(n_ar)]}, ...
                    'FontWeight','Normal','FontSize',16,'Interpreter','latex');
             xlabel(['$q_{',num2str(ihq),'}$'],'FontSize',16,'Interpreter','latex')              
             ylabel(['$p_{Q_{',num2str(ihq),'}}(q_{',num2str(ihq),'})$'],'FontSize',16,'Interpreter','latex') 
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_pdf_Qd_Qar',num2str(ihq),'.fig']);  
             hold off
             close(h);
          end
          if ihw >= 1
             title({['$p_{W_{{\rm d},',num2str(ihw),'}}$ (black thin) with $n_d = $',num2str(n_d)] ; ...
                    ['$p_{W_{\rm{ar},',num2str(ihw),'}}$ (blue thick) with $n_{\rm{ar}} = $',num2str(n_ar)]}, ...
                    'FontWeight','Normal','FontSize',16,'Interpreter','latex');
             xlabel(['$w_{',num2str(ihw),'}$'],'FontSize',16,'Interpreter','latex')              
             ylabel(['$p_{W_{',num2str(ihw),'}}(w_{',num2str(ihw),'})$'],'FontSize',16,'Interpreter','latex') 
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_pdf_Wd_War',num2str(ihw),'.fig']);  
             hold off
             close(h);
          end
      end
   end

   %--- Generation and plot the joint pdf of (XX_{d,ih},XX_{d,jh}) 
   if nbplotPDF2D >= 1
      npoint = 100;      
      for iplot = 1:nbplotPDF2D
          ih        = MatRplotPDF2D(iplot,1);                % MatRplotPDF2D(nbplotPDF2D,2) 
          jh        = MatRplotPDF2D(iplot,2);  
          MatRih_d  = MatRxx_d(ih,:);                        %  MatRih_d(1,n_d),MatRxx_d(n_x,n_d)
          MatRjh_d  = MatRxx_d(jh,:);                        %  MatRjh_d(1,n_d),MatRxx_d(n_x,n_d)
          MINih     = min(MatRih_d(1,:));
          MAXih     = max(MatRih_d(1,:));
          MINjh     = min(MatRjh_d(1,:));
          MAXjh     = max(MatRjh_d(1,:));
          coeff     = 0.2;
          deltaih   = MAXih - MINih;
          deltajh   = MAXjh - MINjh;
          MINih = MINih - coeff*deltaih;
          MAXih = MAXih + coeff*deltaih;
          MINjh = MINjh - coeff*deltajh;
          MAXjh = MAXjh + coeff*deltajh;
          ihq = 0;
          jhq = 0;
          ihw = 0;
          jhw = 0;
          if ih <= n_q
             ihq = ih;
          end
          if ih > n_q
             ihw = ih - n_q;
          end 
          if jh <= n_q
             jhq = jh;
          end
          if jh > n_q
             jhw = jh - n_q;
          end 

          % Compute the joint probability density function using mvksdensity
          [MatRx, MatRy] = meshgrid(linspace(MINih,MAXih,npoint), linspace(MINjh,MAXjh,npoint));
          MatRijT = [MatRih_d(1,:)', MatRjh_d(1,:)'];                   % MatRijT(n_d,2)
          MatRpts = [MatRx(:) MatRy(:)];                                % MatRpts(npoint*npoint,2),MatRx(npoint,npoint),MatRy(npoint,npoint)
          Rpdf = mvksdensity(MatRijT,MatRpts);                          % Rpdf(npoint*npoint,1)
          
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
          if ihq >= 1 && jhq >= 1
             xlabel(['$q_{',num2str(ihq),'}$'],'FontSize',16,'Interpreter','latex')   
             ylabel(['$q_{',num2str(jhq),'}$'],'FontSize',16,'Interpreter','latex')   
             title({['Joint pdf of $Q_{\rm{d},',num2str(ihq),'}$ with $Q_{\rm{d},',num2str(jhq),'}$ for $n_d = $',num2str(n_d)]}, ...
                     'FontWeight','Normal','FontSize',16,'Interpreter','latex');
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_joint_pdf_Qd',num2str(ihq),'_Qd',num2str(jhq),'.fig']); 
             close(h);
          end
          if ihq >= 1 && jhw >= 1
             xlabel(['$q_{',num2str(ihq),'}$'],'FontSize',16,'Interpreter','latex')   
             ylabel(['$w_{',num2str(jhw),'}$'],'FontSize',16,'Interpreter','latex')   
             title({['Joint pdf of $Q_{\rm{d},',num2str(ihq),'}$ with $W_{\rm{d},',num2str(jhw),'}$ for $n_d = $',num2str(n_d)]}, ...
                     'FontWeight','Normal','FontSize',16,'Interpreter','latex');
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_joint_pdf_Qd',num2str(ihq),'_Wd',num2str(jhw),'.fig']); 
             close(h);
          end
          if ihw >= 1 && jhq >= 1
             xlabel(['$w_{',num2str(ihw),'}$'],'FontSize',16,'Interpreter','latex')   
             ylabel(['$q_{',num2str(jhq),'}$'],'FontSize',16,'Interpreter','latex')   
             title({['Joint pdf of $W_{\rm{d},',num2str(ihw),'}$ with $Q_{\rm{d},',num2str(jhq),'}$ for $n_d = $',num2str(n_d)]}, ...
                     'FontWeight','Normal','FontSize',16,'Interpreter','latex');
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_joint_pdf_Wd',num2str(ihw),'_Qd',num2str(jhq),'.fig']); 
             close(h);
          end
          if ihw >= 1 && jhw >= 1
             xlabel(['$w_{',num2str(ihw),'}$'],'FontSize',16,'Interpreter','latex')   
             ylabel(['$w_{',num2str(jhw),'}$'],'FontSize',16,'Interpreter','latex')   
             title({['Joint pdf of $W_{\rm{d},',num2str(ihw),'}$ with $W_{\rm{d},',num2str(jhw),'}$ for $n_d = $',num2str(n_d)]}, ...
                     'FontWeight','Normal','FontSize',16,'Interpreter','latex');
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_joint_pdf_Wd',num2str(ihw),'_d',num2str(jhw),'.fig']); 
             close(h);
          end
      end
   end

   %--- Generation and plot the joint pdf of (XX_{ar,ih},XX_{ar,ih}) 
   if nbplotPDF2D >= 1
      npoint = 100;      
      for iplot = 1:nbplotPDF2D
          ih        = MatRplotPDF2D(iplot,1);                % MatRplotPDF2D(nbplotPDF2D,2) 
          jh        = MatRplotPDF2D(iplot,2);  
          MatRih_ar = MatRxx_obs(Indx_obs == ih,:);          % MatRih_ar(1,n_ar),Indx_obs(nx_obs,1),MatRxx_obs(nx_obs,n_ar)
          MatRjh_ar = MatRxx_obs(Indx_obs == jh,:);          % MatRjh_ar(1,n_ar),MatRxx_obs(nx_obs,n_ar)
          MINih     = min(MatRih_ar(1,:));
          MAXih     = max(MatRih_ar(1,:));
          MINjh     = min(MatRjh_ar(1,:));
          MAXjh     = max(MatRjh_ar(1,:));
          coeff     = 0.2;
          deltaih   = MAXih - MINih;
          deltajh   = MAXjh - MINjh;
          MINih = MINih - coeff*deltaih;
          MAXih = MAXih + coeff*deltaih;
          MINjh = MINjh - coeff*deltajh;
          MAXjh = MAXjh + coeff*deltajh;
          ihq = 0;
          jhq = 0;
          ihw = 0;
          jhw = 0;
          if ih <= n_q
             ihq = ih;
          end
          if ih > n_q
             ihw = ih - n_q;
          end 
          if jh <= n_q
             jhq = jh;
          end
          if jh > n_q
             jhw = jh - n_q;
          end 

          % Compute the joint probability density function using mvksdensity
          [MatRx, MatRy] = meshgrid(linspace(MINih,MAXih,npoint), linspace(MINjh,MAXjh,npoint));
          MatRijT = [MatRih_ar(1,:)', MatRjh_ar(1,:)'];                 % MatRijT(n_ar,2)
          MatRpts = [MatRx(:) MatRy(:)];                                % MatRpts(npoint*npoint,2),MatRx(npoint,npoint),MatRy(npoint,npoint)
          Rpdf = mvksdensity(MatRijT,MatRpts);                          % Rpdf(npoint*npoint,1)
          
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
          if ihq >= 1 && jhq >= 1
             xlabel(['$q_{',num2str(ihq),'}$'],'FontSize',16,'Interpreter','latex')   
             ylabel(['$q_{',num2str(jhq),'}$'],'FontSize',16,'Interpreter','latex')   
             title({['Joint pdf of $Q_{\rm{ar},',num2str(ihq),'}$ with $Q_{\rm{ar},',num2str(jhq),'}$ for $n_{\rm{ar}} = $',num2str(n_ar)]}, ...
                     'FontWeight','Normal','FontSize',16,'Interpreter','latex');
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_joint_pdf_Qar',num2str(ihq),'_Qar',num2str(jhq),'.fig']); 
             close(h);
          end
          if ihq >= 1 && jhw >= 1
             xlabel(['$q_{',num2str(ihq),'}$'],'FontSize',16,'Interpreter','latex')   
             ylabel(['$w_{',num2str(jhw),'}$'],'FontSize',16,'Interpreter','latex')   
             title({['Joint pdf of $Q_{\rm{ar},',num2str(ihq),'}$ with $W_{\rm{ar},',num2str(jhw),'}$ for $n_{\rm{ar}} = $',num2str(n_ar)]}, ...
                     'FontWeight','Normal','FontSize',16,'Interpreter','latex');
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_joint_pdf_Qar',num2str(ihq),'_War',num2str(jhw),'.fig']); 
             close(h);
          end
          if ihw >= 1 && jhq >= 1
             xlabel(['$w_{',num2str(ihw),'}$'],'FontSize',16,'Interpreter','latex')   
             ylabel(['$q_{',num2str(jhq),'}$'],'FontSize',16,'Interpreter','latex')   
             title({['Joint pdf of $W_{\rm{ar},',num2str(ihw),'}$ with $Q_{\rm{ar},',num2str(jhq),'}$ for $n_{\rm{ar}} = $',num2str(n_ar)]}, ...
                     'FontWeight','Normal','FontSize',16,'Interpreter','latex');
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_joint_pdf_War',num2str(ihw),'_Qar',num2str(jhq),'.fig']); 
             close(h);
          end
          if ihw >= 1 && jhw >= 1
             xlabel(['$w_{',num2str(ihw),'}$'],'FontSize',16,'Interpreter','latex')   
             ylabel(['$w_{',num2str(jhw),'}$'],'FontSize',16,'Interpreter','latex')   
             title({['Joint pdf of $W_{\rm{ar},',num2str(ihw),'}$ with $W_{\rm{ar},',num2str(jhw),'}$ for $n_{\rm{ar}} = $',num2str(n_ar)]}, ...
                     'FontWeight','Normal','FontSize',16,'Interpreter','latex');
             numfig = numfig + 1;
             saveas(h,['figure_PlotX',num2str(numfig),'_joint_pdf_War',num2str(ihw),'_War',num2str(jhw),'.fig']); 
             close(h);
          end
      end
   end

   ElapsedTimePlotXdXar = toc(TimeStartPlotXdXar);   

   if ind_print == 1
      fidlisting=fopen('listing.txt','a+');
      fprintf(fidlisting,'      \n ');                                                                
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' ----- Elapsed time for Task12_PlotXdXar \n ');
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' Elapsed Time   =  %10.2f\n',ElapsedTimePlotXdXar);   
      fprintf(fidlisting,'      \n '); 
      fclose(fidlisting);  
   end
   if ind_display_screen == 1   
      disp('--- end Task12_PlotXdXar')
   end    
   return
end


   
 

 