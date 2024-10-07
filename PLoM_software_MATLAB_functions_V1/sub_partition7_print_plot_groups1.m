
function [numfig] = sub_partition7_print_plot_groups1(INDEPopt,ngroup,Igroup,MatIgroup,npair,RplotPair,ind_print,ind_plot,numfig)
   %
   % Copyright C. Soize 24 May 2024 
   %
   %--- INPUTS
   %           INDEPopt               : optimal value of INDEPref
   %           ngroup                 : number of groups that are constructed
   %           Igroup(ngroup,1)       : such that Igroup(j): number mj of the components of  Y^j = (H_r1,... ,H_rmj)             
   %           MatIgroup(ngroup,mmax) : such that MatIgroup1(j,r) = rj : indice rj de H dans le groupe j tel que Y^j_r = H_rj  
   %                                    with mmax = max_j Igroup(j)
   %           npair                  : dimension of RplotPair
   %           RplotPair(npair,1)     : such that RplotPair(pair,1)  = INDEPr1r2 with pair=(r1,r2)
   %           ind_print              : = 0 no print, = 1 print
   %           ind_plot               : = 0 no plot,  = 1 plot
   %           numfig                 : number of generated figures before executing this function
   %--- OUTPUT  
   %           numfig                 : number of generated figures after executing this function

   %--- print
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
   end
   
   %--- plot
   if ind_plot == 1                                                                                    
      if npair >= 2   
         h = figure; 
         plot((1:1:npair)',RplotPair(:,1),'ob')
         title({['Values of $i^\nu(H^\nu_{r_1},H^\nu_{r_2})$ as a function of the pair $p = (r_1,r_2)$']}, ...
                  'FontWeight','Normal','FontSize',16,'Interpreter','latex')                                        
         xlabel(['pair $p = (r_1,r_2)$'],'FontSize',16,'Interpreter','latex')                                                                 
         ylabel(['$i^\nu(H^\nu_{r_1},H^\nu_{r_2})$'],'FontSize',16,'Interpreter','latex')                                                         
         numfig = numfig+1;                                                         
         saveas(h,['figure_PARTITION_',num2str(numfig),'_INDGROUP.fig'])
         close(h);
      end
   end
   return 
end
