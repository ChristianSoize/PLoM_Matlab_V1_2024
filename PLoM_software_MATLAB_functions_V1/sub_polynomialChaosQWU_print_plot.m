function sub_polynomialChaosQWU_print_plot(nq_obs,nar_PCE,Indq_obs,MatRqq_ar0,MatRqq_PolChaos_ar0, ...
                                           ind_display_screen,ind_print,ind_plot,ind_type)
                                        
    %----------------------------------------------------------------------------------------------------------------------------------------
    %
    %  Copyright: Christian Soize, Universite Gustave Eiffel, 24 June 2024
    %
    %  Software     : Probabilistic Learning on Manifolds (PLoM) 
    %  Function name: sub_polynomialChaosQWU_print_plot
    %  Subject      : print and plot 

    %--- INPUT
    %        nq_obs                               : dimension of Qobs^0
    %        nar_PCE                              : number of realizations used for the PCE construction 
    %        Indq_obs(nq_obs,1)                   : obervation number for Q
    %        MatRqq_ar0(nq_obs,nar_PCE)           : nar_PCE learned realizations for Qobs
    %        MatRqq_PolChaos_ar0(nq_obs,nar_PCE)) : polynomial chaos representation
    %        ind_display_screen                   : = 0 no display, = 1 display
    %        ind_print                            : = 0 no print,   = 1 print
    %        ind_plot                             : = 0 no plot,    = 1 plot
    %        ind_type                             : = 1 Polynomial-chaos representation for MatRww_ar0(nw_obs,nar_PCE) 
    %                                             : = 2 Polynomial-chaos validation for MatRww_o(nw_obs,n_o)
    
    if ind_print == 1
       if ind_type == 1 
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,'Polynomial-chaos representation for MatRww_ar0(nw_obs,nar_PCE) --------------------------------------  \n ');
          fprintf(fidlisting,'      \n ');                                                                                                                                                         
          fprintf(fidlisting,'      \n '); 
          fclose(fidlisting);  
       end
       if ind_type == 2 
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,'Polynomial-chaos validation for MatRww_o(nw_obs,n_o) ------------------------------------------------  \n ');
          fprintf(fidlisting,'      \n ');                                                                                                                                                         
          fprintf(fidlisting,'      \n '); 
          fclose(fidlisting);  
       end
       if ind_type == 3 
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,'Polynomial-chaos realization ------------------------------------------------------------------------  \n ');
          fprintf(fidlisting,'      \n ');                                                                                                                                                         
          fprintf(fidlisting,'      \n '); 
          fclose(fidlisting);  
       end
    end

    %--- second-order moment
    MatRmom2QQ_ar0          = (MatRqq_ar0*(MatRqq_ar0'))/(nar_PCE-1);                    % MatRmom2QQ_ar0(nq_obs,nq_obs)
    MatRmom2QQ_PolChaos_ar0 = MatRqq_PolChaos_ar0*(MatRqq_PolChaos_ar0')/(nar_PCE - 1);  % MatRmom2QQ_PolChaos_ar0(nq_obs,nq_obs) 
   
    %--- mean value
    MatRmean = [ mean(MatRqq_ar0,2)  mean(MatRqq_PolChaos_ar0,2)];
    if ind_display_screen == 1
       disp('mean value: QQ_ar0 QQ_PolChaos_ar0')
       disp(MatRmean)  
    end
   
    %--- standard deviation       
    MatRstd = [ std(MatRqq_ar0,0,2)  std(MatRqq_PolChaos_ar0,0,2)]; 
    if ind_display_screen == 1
       disp('standard deviation:  QQ_ar0 QQ_PolChaos_ar0')
       disp(MatRstd)
    end

    % %--- skewness  (power 3)      
    % MatRskew = [ (skewness(MatRqq_ar0'))'  (skewness(MatRqq_PolChaos_ar0'))']; 
    % if ind_display_screen == 1
    %    disp('skewness:  QQ_ar0 QQ_PolChaos_ar0')
    %    disp(MatRskew)
    % end
    % 
    % %--- kurtosis (power 4)      
    % MatRkurto = [ (kurtosis(MatRqq_ar0'))'  (kurtosis(MatRqq_PolChaos_ar0'))']; 
    % if ind_display_screen == 1
    %    disp('kurtosis:  QQ_ar0 QQ_PolChaos_ar0')
    %    disp(MatRkurto)
    % end
   
    %---- print
    if ind_print == 1
       fidlisting=fopen('listing.txt','a+');
       fprintf(fidlisting,'      \n ');  
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,'Second-order moment matrix of MatRqq_ar0 \n '); 
       for i = 1:nq_obs          
           Rprint = MatRmom2QQ_ar0(i,:);  
           fprintf(fidlisting,' %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e \n ',Rprint); 
       end          
       fprintf(fidlisting,'      \n ');  
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,'Second-order moment matrix of MatRqq_PolChaos_ar0 \n '); 
       for i = 1:nq_obs          
           Rprint = MatRmom2QQ_PolChaos_ar0(i,:);  
           fprintf(fidlisting,' %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e \n ',Rprint); 
       end           
       fprintf(fidlisting,'      \n ');  
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,'--- Mean value: \n ');
       fprintf(fidlisting,'                  mean(QQ_ar0)   mean(QQ_PolChaos_ar0) \n ');
       fprintf(fidlisting,'      \n '); 
       for i = 1:nq_obs   
           fprintf(fidlisting,'             %14.7f %14.7f  \n ',MatRmean(i,:)); 
       end 
       fprintf(fidlisting,'      \n ');  
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,'--- Standard deviation: \n ');
       fprintf(fidlisting,'                  std(QQ_ar0)    std(QQ_PolChaos_ar0) \n ');
       fprintf(fidlisting,'      \n '); 
       for i = 1:nq_obs   
           fprintf(fidlisting,'             %14.7f %14.7f  \n ',MatRstd(i,:)); 
       end          
       % fprintf(fidlisting,'      \n ');  
       % fprintf(fidlisting,'      \n '); 
       % fprintf(fidlisting,'--- Skewness: \n ');
       % fprintf(fidlisting,'                 skew(QQ_ar0)   skew(QQ_PolChaos_ar0) \n ');
       % fprintf(fidlisting,'      \n '); 
       % for i = 1:nq_obs   
       %     fprintf(fidlisting,'             %14.7f %14.7f  \n ',MatRskew(i,:)); 
       % end 
       % fprintf(fidlisting,'      \n ');  
       % fprintf(fidlisting,'      \n '); 
       % fprintf(fidlisting,'--- Kurtosis: \n ');
       % fprintf(fidlisting,'                kurto(QQ_ar0)  kurto(QQ_PolChaos_ar0) \n ');
       % fprintf(fidlisting,'      \n '); 
       % for i = 1:nq_obs   
       %     fprintf(fidlisting,'             %14.7f %14.7f  \n ',MatRkurto(i,:)); 
       % end 
       fprintf(fidlisting,'      \n ');  
       fprintf(fidlisting,'      \n '); 
       fclose(fidlisting);        
    end
    
    if ind_plot == 1
       %--- plot histogram MatRqq_ar0(nq_obs,nar_PCE)
       for k = 1:nq_obs   
           kobs = Indq_obs(k,1);                    % Indq_obs(nq_obs,1)
           hold off
           h = figure;
           histogram(MatRqq_ar0(k,:));
           title(['Learning'],'FontSize',16,'Interpreter','latex','FontWeight','normal')  
           xlabel('$q$','FontSize',16,'Interpreter','latex')                                                                
           ylabel(['${\rm{histogram}}_{Q_{',num2str(kobs),'}}(q)$'],'FontSize',16,'Interpreter','latex')
           saveas(h,['figure_histogram_Q',num2str(kobs),'_learning.fig']); 
           close(h)
       end     
   
       %--- plot histogram MatRqq_PolChaos_ar0(nq_obs,nar_PCE)
       for k = 1:nq_obs   
           kobs = Indq_obs(k,1);                    % Indq_obs(nq_obs,1)
           hold off
           h = figure;
           histogram(MatRqq_PolChaos_ar0(k,:));
           if ind_type == 1
              title(['Polynomial chaos representation'],'FontSize',16,'Interpreter','latex','FontWeight','normal')  
              xlabel('$q$','FontSize',16,'Interpreter','latex')                                                                
              ylabel(['${\rm{histogram}}_{Q_{',num2str(kobs),'}}(q)$'],'FontSize',16,'Interpreter','latex')
              saveas(h,['figure_histogram_Q',num2str(kobs),'_PCE_representation.fig']); 
              close(h)
           end
           if ind_type == 2
              title(['Polynomial chaos validation'],'FontSize',16,'Interpreter','latex','FontWeight','normal')  
              xlabel('$q$','FontSize',16,'Interpreter','latex')                                                                
              ylabel(['${\rm{histogram}}_{Q_{',num2str(kobs),'}}(q)$'],'FontSize',16,'Interpreter','latex')
              saveas(h,['figure_histogram_Q',num2str(kobs),'_PCE_validation.fig']); 
              close(h)
           end
           if ind_type == 3
              title(['Polynomial chaos realization'],'FontSize',16,'Interpreter','latex','FontWeight','normal')  
              xlabel('$q$','FontSize',16,'Interpreter','latex')                                                                
              ylabel(['${\rm{histogram}}_{Q_{',num2str(kobs),'}}(q)$'],'FontSize',16,'Interpreter','latex')
              saveas(h,['figure_histogram_Q',num2str(kobs),'_PCE_realization.fig']); 
              close(h)
           end
       end     
       close all
    end
    return
end
   
 









         