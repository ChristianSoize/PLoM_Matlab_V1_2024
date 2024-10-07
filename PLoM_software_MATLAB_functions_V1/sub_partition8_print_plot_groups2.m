
function [numfig] = sub_partition8_print_plot_groups2(INDEPopt,ngroup,Igroup,MatIgroup,nedge,MatPrintEdge,nindep,MatPrintIndep,npair, ...
                                                      MatPrintPair,RplotPair,ind_print,ind_plot,numfig)
   %
   % Copyright C. Soize 24 May 2024 
   %
   %--- INPUTS
   %           INDEPopt               : optimal value of INDEPref
   %           ngroup                 : number of groups that are constructed
   %           Igroup(ngroup,1)       : such that Igroup(j): number mj of the components of  Y^j = (H_r1,... ,H_rmj)             
   %           MatIgroup(ngroup,mmax) : such that MatIgroup1(j,r) = rj : indice rj de H dans le groupe j tel que Y^j_r = H_rj  
   %                                    with mmax = max_j Igroup(j)
   %           nedge                  : number of pairs (r1,r2) for which H_r1 and H_r2 are dependent (number of edges in the graph)
   %           MatPrintEdge(nedge,5)  : such that MatPrintEdge(edge,:) = [edge  r1 r2 INDEPr1r2 INDEPopt]
   %           nindep                 : number of pairs (r1,r2) for which H_r1 and H_r2 are independent
   %           MatPrintIndep(nindep,5): such that MatPrintIndep(indep,:) = [indep r1 r2 INDEPr1r2 INDEPopt]
   %           npair                  : dimension of RplotPair
   %           RplotPair(npair,1)     : such that RplotPair(pair,1)  = INDEPr1r2 with pair=(r1,r2)
   %           npair                  : total number of pairs (r1,r2) = npairmax = NKL(NKL-1)/2
   %           MatPrintPair(npair,5)  : such that MatPrintPair(pair,:)    = [pair  r1 r2 INDEPr1r2 INDEPopt]
   %           RplotPair(npair,1)     : such that RplotPair(pair,1)       = INDEPr1r2 with pair=(r1,r2)
   %           ind_print              : = 0 no print, = 1 print
   %           ind_plot               : = 0 no plot,  = 1 plot
   %           numfig                 : number of generated figures before executing this function
   %--- OUTPUT  
   %           numfig                 : number of generated figures after executing this function

   %--- Print
   if ind_print == 1           
      fidlisting=fopen('listing.txt','a+'); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'--------------- OPTIMAL PARTITION IN GROUPS OF INDEPENDENT RANDOM VECTORS ----------');
      fprintf(fidlisting,'      \n ');    
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'Optimal value INDEPopt of INDEPref used for constructing the optimal partition = %8.5i \n ',INDEPopt); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'ngroup  = %7i \n ',ngroup);   
      for j = 1:ngroup 
           fprintf(fidlisting,'      \n '); 
           PPrint =[j MatIgroup(j,1:Igroup(j))];
           fprintf(fidlisting,' %7i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i \n ', PPrint);
           clear PPrint
      end
      fclose(fidlisting); 
                                           
      fidlisting=fopen('listing.txt','a+'); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'nedge  = %7i \n ',nedge); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' edge     r1    r2 INDEPr1r2 INDEPopt  \n ');
      fprintf(fidlisting,'      \n ');
      for edge = 1:nedge 
          Print = MatPrintEdge(edge,:);                                       
          fprintf(fidlisting,' %4i %6i %6i %8.5f %8.5f \n ', Print);         
      end
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n ');    
      fclose(fidlisting); 
     
      fidlisting=fopen('listing.txt','a+'); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'nindep  = %7i \n ',nindep); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,' indep     r1    r2 INDEPr1r2 INDEPopt \n ');
      fprintf(fidlisting,'      \n ');
      for indep = 1:nindep 
          Print = MatPrintIndep(indep,:);                                        
          fprintf(fidlisting,' %4i %6i %6i %8.5f %8.5f \n ', Print);         
      end
      fprintf(fidlisting,'      \n '); 
      fprintf(fidlisting,'      \n ');    
      fclose(fidlisting); 
   end     
   
   %--- Plot
   if ind_plot == 1
      if nedge >= 2
         h = figure; 
         plot((1:1:nedge)',MatPrintEdge(:,4),'ob',(1:1:nedge)',MatPrintEdge(:,5),'-k')
         title({['Graph of $i^\nu(H^\nu_{r_1},H^\nu_{r_2})$ for which $H^\nu_{r_1}$ and $H^\nu_{r_2}$ are'] ; ...
                ['dependent and value of $i^{\rm{opt}}_{\rm{ref}}$ (solid line)']}, ...
                'FontWeight','Normal','FontSize',16,'Interpreter','latex');                                        
         xlabel(['pair $p=(r_1,r_2)$'],'FontSize',16,'Interpreter','latex');                                                                 
         ylabel(['$i^\nu(H^\nu_{r_1},H^\nu_{r_2})$'],'FontSize',16,'Interpreter','latex'); 
         numfig = numfig + 1;
         saveas(h,['figure_PARTITION_',num2str(numfig),'_i-H-iopt1.fig'])
         close(h);
      end
                                                                                        
      if nindep >= 2
         h = figure; 
         plot((1:1:nindep)',MatPrintIndep(:,4),'ob',(1:1:nindep)',MatPrintIndep(:,5),'-k')
         title({['Graph of $i^\nu(H^\nu_{r_1},H^\nu_{r_2})$ for which $H^\nu_{r_1}$ and $H^\nu_{r_2}$ are'] ; ...
                ['independent and value of $i^{\rm{opt}}_{\rm{ref}}$ (solid line)']}, ...
                'FontWeight','Normal','FontSize',16,'Interpreter','latex');                                        
         xlabel(['pair $p=(r_1,r_2)$'],'FontSize',16,'Interpreter','latex');                                                                 
         ylabel(['$i^\nu(H^\nu_{r_1},H^\nu_{r_2})$'],'FontSize',16,'Interpreter','latex');
         numfig = numfig + 1;
         saveas(h,['figure_PARTITION_',num2str(numfig),'_i-H-iopt2.fig'])
         close(h);
      end
                                                                                        
      if npair >= 2
         h = figure; 
         plot((1:1:npair)',MatPrintPair(:,4),'ob',(1:1:npair)',MatPrintPair(:,5),'-k')
         title({['Graph of $i^\nu(H^\nu_{r_1},H^\nu_{r_2})$ for which $H^\nu_{r_1}$ and $H^\nu_{r_2}$ are'] ; ...
                ['dependent or independent and value of $i^{\rm{opt}}_{\rm{ref}}$ (solid line)']}, ...
                'FontWeight','Normal','FontSize',16,'Interpreter','latex');                                        
         xlabel(['pair $p=(r_1,r_2)$'],'FontSize',16,'Interpreter','latex');                                                                 
         ylabel(['$i^\nu(H^\nu_{r_1},H^\nu_{r_2})$'],'FontSize',16,'Interpreter','latex'); 
         numfig = numfig + 1;
         saveas(h,['figure_PARTITION_',num2str(numfig),'_i-H-iopt3.fig'])
         close(h);
         
         h = figure; 
         plot((1:1:npair)',RplotPair(:,1),'ob')
         title({['Values of $i^\nu(H^\nu_{r_1},H^\nu_{r_2})$ as a function of the pair $p = (r_1,r_2)$']}, ...
                 'FontWeight','Normal','FontSize',16,'Interpreter','latex');                                        
         xlabel(['pair $p = (r_1,r_2)$'],'FontSize',16,'Interpreter','latex') ;                                                                
         ylabel(['$i^\nu(H^\nu_{r_1},H^\nu_{r_2})$'],'FontSize',16,'Interpreter','latex');                                                         
         numfig = numfig + 1;                                                         
         saveas(h,['figure_PARTITION_',num2str(numfig),'_i-H-pair.fig'])
         close(h);
      end
   end
return 
end
