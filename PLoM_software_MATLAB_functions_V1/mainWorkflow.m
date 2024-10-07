function mainWorkflow(mainWorkflow_Data_WorkFlow,ind_workflow)

    %----------------------------------------------------------------------------------------------------------------------------------------
    %
    %  Copyright: Christian Soize, Universite Gustave Eiffel, 05 October 2024
    %
    %  Software     : Probabilistic Learning on Manifolds (PLoM) 
    %  Function name: mainWorkflow
    %  Subject      : managing the work flow and data 
    %
    %  Publications: [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    %                       Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).
    %                [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    %                       American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
    %                [3] C. Soize, R. Ghanem, Physics-constrained non-Gaussian probabilistic learning on manifolds, 
    %                       International Journal for Numerical Methods in Engineering, doi: 10.1002/nme.6202, 121 (1), 110-145 (2020).
    %                [4] C. Soize, R. Ghanem, Probabilistic learning on manifolds constrained by nonlinear partial differential equations 
    %                       from small datasets, Computer Methods in Applied Mechanics and Engineering, doi:10.1016/j.cma.2021.113777, 
    %                       380, 113777 (2021).
    %                [5] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
    %                       Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).
    %                [6] C. Soize, Probabilistic learning inference of boundary value problem with uncertainties based on Kullback-Leibler 
    %                       divergence under implicit constraints, Computer Methods in Applied Mechanics and Engineering,
    %                       doi:10.1016/j.cma.2022.115078, 395, 115078 (2022). 
    %                [7] C. Soize, Probabilistic learning constrained by realizations using a weak formulation of Fourier transform of 
    %                       probability measures, Computational Statistics, doi:10.1007/s00180-022-01300-w, 38(4),1879–1925 (2023).
    %                [8] C. Soize, R. Ghanem, Probabilistic-learning-based stochastic surrogate model from small incomplete datasets 
    %                       for nonlinear dynamical systems,Computer Methods in Applied Mechanics and Engineering, 
    %                       doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024).
    %                [9] C. Soize, R. Ghanem, Transient anisotropic kernel for probabilistic learning on manifolds, 
    %                       Computer Methods in Applied Mechanics and Engineering, pp.1-44 (2024).
    %               [10] C. Soize, R. Ghanem, Physical systems with random uncertainties: Chaos representation with arbitrary probability 
    %                       measure, SIAM Journal on Scientific Computing, doi:10.1137/S1064827503424505, 26}2), 395-410 (2004).
    %               [11] C. Soize, C. Desceliers, Computational aspects for constructing realizations of polynomial chaos in high 
    %                       dimension}, SIAM Journal On Scientific Computing, doi:10.1137/100787830, 32(5), 2820-2831 (2010).
    %               [13] C. Soize, Q-D. To, Polynomial-chaos-based conditional statistics for probabilistic learning with heterogeneous
    %                       data applied to atomic collisions of Helium on graphite substrate, Journal of Computational Physics,
    %                       doi:10.1016/j.jcp.2023.112582, 496, 112582, pp.1-20 (2024).
    
    %--------------------------------------------------------------------------------------------------------------------------------------
    %                  WorkFlow1                 |                  WorkFlow2                 |                  WorkFlow3 |              |  
    %        SolverDirect_WithoutPartition       |          SolverDirect_WithPartition        |        SolverInverse_WithoutPartition     |     
    %--------------------------------------------|--------------------------------------------|-------------------------------------------|
    %  Step1_Pre_processing                      |  Step1_Pre_processing                      |   Step1_Pre_processing                    |
    %        Task1_DataStructureCheck            |        Task1_DataStructureCheck            |         Task1_DataStructureCheck          |
    %        Task2_Scaling                       |        Task2_Scaling                       |         Task2_Scaling                     |
    %        Task3_PCA                           |        Task3_PCA                           |         Task3_PCA                         |
    %                                            |        Task4_Partition                     |         Task8_ProjectionTarget            |
    %--------------------------------------------|--------------------------------------------|-------------------------------------------|
    %  Step2_Processing                          |  Step2_Processing                          |   Step2_Processing                        |
    %        Task5_ProjectionBasisNoPartition    |                                            |         Task5_ProjectionBasisNoPartition  |
    %        Task6_SolverDirect                  |        Task7_SolverDirectPartition         |         Task9_SolverInverse               |
    %--------------------------------------------|--------------------------------------------|-------------------------------------------|
    %  Step3_Post_processing                     |  Step3_Post_processing                     |   Step3_Post_processing                   |
    %        Task10_PCAback                      |        Task10_PCAback                      |         Task10_PCAback                    |
    %        Task11_ScalingBack                  |        Task11_ScalingBack                  |         Task11_ScalingBack                |
    %        Task12_PlotXdXar                    |        Task12_PlotXdXar                    |         Task12_PlotXdXar                  |
    %--------------------------------------------|--------------------------------------------|-------------------------------------------|
    %  Step4_Conditional_statistics_processing   |  Step4_Conditional_statistics_processing   |   Step4_Conditional_statistics_processing |
    %        Task13_ConditionalStatistics        |        Task13_ConditionalStatistics        |         Task13_ConditionalStatistics      |
    %        Task14_PolynomialChaosZwiener       |                                            |         Task14_PolynomialChaosZwiener     |
    %        Task15_PolynomialChaosQWU           |        Task15_PolynomialChaosQWU           |         Task15_PolynomialChaosQWU         |
    %--------------------------------------------|--------------------------------------------|-------------------------------------------|
    
    %
    %--- IMORTANT
    %    For the supervised analysis the information structure is 
    %    XX = (QQ,WW) with dimension n_x = n_q + n_w, 
    %
   
    %--- WARNING: defining the workflow types, which must not modify by the user
    nbworkflow = 3;
    cellRworkflow= { ...                                            % cell array cellRworkflow(nbworkflow,1)
                     'WorkFlow1_SolverDirect_WithoutPartition', ...
                     'WorkFlow2_SolverDirect_WithPartition', ...
                     'WorkFlow3_SolverInverse_WithoutPartition', ...  
                   }';
   
    %--- WARNING: defining all the existing tasks for all the workflows, which must not modify by the user
    nbtask = 15;
    cellRtask = { ...                                               % cell array cellRtask(nbtask,1)
                  'Task1_DataStructureCheck', ...
                  'Task2_Scaling', ...
                  'Task3_PCA', ...       
                  'Task4_Partition', ...
                  'Task5_ProjectionBasisNoPartition', ...
                  'Task6_SolverDirect', ...
                  'Task7_SolverDirectPartition', ...
                  'Task8_ProjectionTarget', ...
                  'Task9_SolverInverse', ...
                  'Task10_PCAback', ...
                  'Task11_ScalingBack', ...
                  'Task12_PlotXdXar', ...
                  'Task13_ConditionalStatistics', ...
                  'Task14_PolynomialChaosZwiener', ...
                  'Task15_PolynomialChaosQWU' ...
                }';
   
    %--- WARNING: initialization of the execution of the tasks, which must not modify by the user
    exec_task1 = 0; exec_task2 = 0; exec_task3 = 0; exec_task4 = 0; exec_task5 = 0; exec_task6 = 0; exec_task7 = 0;
    exec_task8 = 0; exec_task9 = 0; exec_task10 = 0; exec_task11 = 0; exec_task12 = 0; exec_task13 = 0; exec_task14 = 0;
    exec_task15 = 0;
   
    %--- WARNING: defining the steps, which must not modify by the user 
    nbstep = 4;
    cellRstep= { ...                                               % cell array cellRstep(nbstep,1)
                 'Step1_Pre_processing', ...  
                 'Step2_Processing', ... 
                 'Step3_Post_processing', ...
                 'Step4_Conditional_statistics_processing', ...
               }'; 
   
    %--- Initialization sequence must not be modified by the user --------------------------------------------------------------------
    ind_scaling=0;error_PCA=0;n_q=0;Indq_real=[];Indq_pos=[];n_w=0;Indw_real=[];Indw_pos=[];Indq_obs=[];Indw_obs=[];n_d=0;MatRxx_d=[];
    mDP=0;ind_generator=0;ind_basis_type=0;ind_file_type=0;epsilonDIFFmin=0;step_epsilonDIFF=0;iterlimit_epsilonDIFF=0;comp_ref=0;
    nbMC=0;icorrectif=0;f0_ref=0;ind_f0=0;coeffDeltar=0;M0transient=0;ind_constraints=0;ind_coupling=0;iter_limit=0;epsc=0;minVarH=0;
    maxVarH=0;alpha_relax1=0;iter_relax2=0;alpha_relax2=0;MatRplotHsamples=[];MatRplotHClouds=[];MatRplotHpdf=[];MatRplotHpdf2D=[];
    ind_Kullback=0;ind_Entropy=0;ind_MutualInfo=0;MatRplotSamples=[];MatRplotClouds=[];MatRplotPDF=[];MatRplotPDF2D=[];
    ind_mean=0;ind_mean_som=0;ind_pdf=0;ind_confregion=0;nbParam=0;nbw0_obs=0;MatRww0_obs=[];Ind_Qcomp=[];
    nbpoint_pdf=0;pc_confregion=0;ind_PCE_ident=0;ind_PCE_compt=0;nbMC_PCE=0;Rmu=[];RM=[];mu_PCE=0;M_PCE=0;
    Ng=0;Ndeg=0;ng=0;ndeg=0;MaxIter=0;RINDEPref=[];MatRHplot=[];MatRcoupleRHplot=[]; ind_type_targ=0;Indq_targ_real=[];Indq_targ_pos=[];
    Indw_targ_real=[];Indw_targ_pos=[];N_r=0;MatRxx_targ_real=[];MatRxx_targ_pos=[];Rmeanxx_targ=[];MatRcovxx_targ=[];eps_inv=0;
    ind_SavefileStep3=0;ind_SavefileStep4=0;
    %----------------------------------------------------------------------------------------------------------------------------------
    
    %------------------------------------------------------------------------------------------------------------------------------------
    %          DATA FOR "WorkFlow1_SolverDirect_WithoutPartition" (ind_workflow = 1) 
    %                   "WorkFlow2_SolverDirect_WithPartition" (ind_workflow = 2)
    %                   "WorkFlow3_SolverInverse_WithoutPartition" (ind_workflow = 3)
    %          Data are given inside function mainWorkflow_Data_WorkFlow1, 2 or 3
    %------------------------------------------------------------------------------------------------------------------------------------
       
    [ind_step1,ind_step2,ind_step3,ind_step4,ind_step4_task13,ind_step4_task14,ind_step4_task15,ind_SavefileStep3,ind_SavefileStep4] = ...
                                                                             mainWorkflow_Data_WorkFlow();
    
   
    %===================================================================================================================
    %                                                  CHECKING DATA
    %===================================================================================================================
   
    % Save the current values of the control parameters  on a temporary file of the job
    save('FileTemporary.mat','ind_step1','ind_step2','ind_step3','ind_step4','ind_step4_task13','ind_step4_task14','ind_step4_task15', ...
          'ind_SavefileStep3','ind_SavefileStep4');
   
    % The two following "if sequence" must not be modified by the user
    if ind_workflow == 1 || ind_workflow == 2
       ind_exec_solver  = 1;
       Indq_targ_real   = [];
       Indq_targ_pos    = [];
       Indw_targ_real   = [];
       Indw_targ_pos    = [];
       ind_type_targ    = 0;
       N_r              = 0;
       MatRxx_targ_real = [];
       MatRxx_targ_pos  = [];
       Rmeanxx_targ     = [];
       MatRcovxx_targ   = [];
    end
    if ind_workflow == 3
       ind_exec_solver = 2;
    end
    
    %--- checking the consistency of parameters
    if  ind_workflow ~= 1 &&  ind_workflow ~= 2 && ind_workflow ~= 3
        error('STOP1 in mainWorkflow: ind_workflow must be equal to 1, 2 or 3');  
    end
    if  ind_step1 ~= 0 &&  ind_step1 ~= 1
        error('STOP2 in mainWorkflow: ind_step1 must be equal to 0 or 1');  
    end
    if  ind_step2 ~= 0 &&  ind_step2 ~= 1
        error('STOP3 in mainWorkflow: ind_step2 must be equal to 0 or 1');  
    end
    if  ind_step3 ~= 0 &&  ind_step3 ~= 1
        error('STOP4 in mainWorkflow: ind_step3 must be equal to 0 or 1');  
    end
    if  ind_step4 ~= 0 &&  ind_step4 ~= 1
        error('STOP5 in mainWorkflow: ind_step4 must be equal to 0 or 1');  
    end
    if  ind_step4_task13 ~= 0 &&  ind_step4_task13 ~= 1
        error('STOP6 in mainWorkflow: ind_step4_task13 must be equal to 0 or 1');  
    end
    if  ind_step4_task14 ~= 0 &&  ind_step4_task14 ~= 1
        error('STOP7 in mainWorkflow: ind_step4_task14 must be equal to 0 or 1');  
    end
    if  ind_step4_task15 ~= 0 &&  ind_step4_task15 ~= 1
        error('STOP8 in mainWorkflow: ind_step4_task15 must be equal to 0 or 1');  
    end
    if  ind_SavefileStep3 ~= 0 &&  ind_SavefileStep3 ~= 1
        error('STOP9 in mainWorkflow: ind_SavefileStep3 must be equal to 0 or 1');  
    end 
    if  ind_SavefileStep4 ~= 0 &&  ind_SavefileStep4 ~= 1
        error('STOP10 in mainWorkflow: ind_SavefileStep4 must be equal to 0 or 1');  
    end 
   
    %--- print
    fidlisting=fopen('listing.txt','a+');
    fprintf(fidlisting,'      \n '); 
    fprintf(fidlisting,' ---------------- %s ---------------- \n', cellRworkflow{ind_workflow,1}); 
    fprintf(fidlisting,'      \n ');   
    fprintf(fidlisting,'  %s                    = %1i \n', cellRstep{1,1},ind_step1);       % the spaces are added for getting alignment
    fprintf(fidlisting,'   %s                        = %1i \n', cellRstep{2,1},ind_step2); 
    fprintf(fidlisting,'   %s                   = %1i \n', cellRstep{3,1},ind_step3); 
    fprintf(fidlisting,'   %s = %1i \n', cellRstep{4,1},ind_step4); 
    fclose(fidlisting); 
   
    %------------------------------------------------------------------------------------------------------------------------------------
    %                                                 Step1_Pre_processing
    %------------------------------------------------------------------------------------------------------------------------------------
    
    if ind_step1 == 1
       rng('default')    
   
       %--- Execution of the tasks of Step2
       if ind_workflow  == 1 || ind_workflow  == 2 || ind_workflow  == 3
   
          if ind_workflow  == 1
             % Load FileDataWorkFlow1Step1.mat
             fileName = 'FileDataWorkFlow1Step1.mat';
             if exist(fileName, 'file') == 2
                load(fileName, ...
                      'ind_display_screen','ind_print','ind_plot','ind_parallel','ind_scaling','error_PCA','n_q','Indq_real', ...
                      'Indq_pos','n_w','Indw_real','Indw_pos','Indq_obs','Indw_obs','n_d','MatRxx_d');
             else
                 error('STOP11 in mainWorkflow: File FileDataWorkFlow1Step1.mat does not exist');
             end
          end
   
          if ind_workflow  == 2
             % Load FileDataWorkFlow1Step1.mat
             fileName = 'FileDataWorkFlow2Step1.mat';
             if exist(fileName, 'file') == 2
                load(fileName, ...
                  'ind_display_screen','ind_print','ind_plot','ind_parallel','ind_scaling','error_PCA','n_q','Indq_real', ...
                  'Indq_pos','n_w','Indw_real','Indw_pos','Indq_obs','Indw_obs','n_d','MatRxx_d','RINDEPref','MatRHplot','MatRcoupleRHplot');
             else
                 error('STOP12 in mainWorkflow: File FileDataWorkFlow2Step1.mat does not exist');
             end
          end
   
          if ind_workflow  == 3
             % Load FileDataWorkFlow1Step1.mat
             fileName = 'FileDataWorkFlow3Step1.mat';
             if exist(fileName, 'file') == 2
                load(fileName, ...
                 'ind_display_screen','ind_print','ind_plot','ind_parallel','ind_scaling','error_PCA','n_q','Indq_real', ...
                 'Indq_pos','n_w','Indw_real','Indw_pos','Indq_obs','Indw_obs','n_d','MatRxx_d','ind_type_targ','Indq_targ_real', ...
                 'Indq_targ_pos','Indw_targ_real','Indw_targ_pos','N_r','MatRxx_targ_real','MatRxx_targ_pos','Rmeanxx_targ','MatRcovxx_targ');
             else
                 error('STOP13 in mainWorkflow: File FileDataWorkFlow3Step1.mat does not exist');
             end
          end
   
          if ind_display_screen == 1                              
             disp(' ================================ Step1 Pre_processing ================================ ')
          end
          if ind_print == 1
             fidlisting=fopen('listing.txt','a+');
             fprintf(fidlisting,'      \n '); 
             fprintf(fidlisting,' ================================ Step1 Pre_processing ================================ \n ');
             fprintf(fidlisting,'      \n ');  
             fclose(fidlisting);  
          end
   
          % Task1_DataStructureCheck
          [n_x,Indx_real,Indx_pos,nx_obs,Indx_obs,Indx_targ_real,Indx_targ_pos,nx_targ,Indx_targ] =  ...
                         sub_data_structure_and_check(n_q,Indq_real,Indq_pos,n_w,Indw_real,Indw_pos,n_d,MatRxx_d,Indq_obs,Indw_obs, ...
                                                      ind_display_screen,ind_print,ind_exec_solver,Indq_targ_real,Indq_targ_pos, ...
                                                      Indw_targ_real,Indw_targ_pos,ind_type_targ,N_r,MatRxx_targ_real,MatRxx_targ_pos, ...
                                                      Rmeanxx_targ,MatRcovxx_targ);
          exec_task1 = 1; 
   
          % Task2_Scaling
          [MatRx_d,Rbeta_scale_real,Ralpha_scale_real,Ralpham1_scale_real, ...
                    Rbeta_scale_log,Ralpha_scale_log,Ralpham1_scale_log ] = ...
                                      sub_scaling(n_x,n_d,MatRxx_d,Indx_real,Indx_pos,ind_display_screen,ind_print,ind_scaling); 
          exec_task2 = 1;   

          % Task3_PCA
          [nu,nnull,MatReta_d,RmuPCA,MatRVectPCA] = sub_PCA(n_x,n_d,MatRx_d,error_PCA,ind_display_screen,ind_print,ind_plot);
          exec_task3 = 1;
          SAVERANDendSTEP1 = rng;
       end
   
       if ind_workflow  == 2 
          % Task4_Partition
          SAVERANDstartPARTITION = rng; 
          nref                   = size(RINDEPref,1); 
          [ngroup,Igroup,MatIgroup,SAVERANDendPARTITION] = sub_partition1(nu,n_d,nref,MatReta_d,RINDEPref, ...
                                SAVERANDstartPARTITION,ind_display_screen,ind_print,ind_plot,MatRHplot,MatRcoupleRHplot,ind_parallel);
          exec_task4 = 1;
          SAVERANDendSTEP1 = SAVERANDendPARTITION;
       end
   
       if ind_workflow == 3 
          % Task8_ProjectionTarget
          [Rb_targ1,coNr,coNr2,MatReta_targ,Rb_targ2,Rb_targ3] = sub_projection_target(n_x,n_d,MatRx_d,ind_exec_solver,ind_scaling, ...
                   ind_type_targ,Indx_targ_real,Indx_targ_pos,nx_targ,Indx_targ,N_r,MatRxx_targ_real,MatRxx_targ_pos,Rmeanxx_targ, ...
                   MatRcovxx_targ,nu,RmuPCA,MatRVectPCA,ind_display_screen,ind_print,ind_parallel,Rbeta_scale_real,Ralpham1_scale_real, ...
                   Rbeta_scale_log,Ralpham1_scale_log);
          exec_task8 = 1;
          SAVERANDendSTEP1 = rng;
       end
       
       %--- SavefileStep1.mat
       fileName = 'SavefileStep1.mat';  
       if exist(fileName, 'file') == 2
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,'  Warning: The file "%s" already exists and will be overwritten\n', fileName);
          fprintf(fidlisting,'      \n '); 
          fclose(fidlisting); 
          disp(['Warning: The file "', fileName, '" already exists and will be overwritten']);
       end
       save(fileName, '-v7.3');
       fidlisting=fopen('listing.txt','a+');
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,'  The file "%s" has been saved\n', fileName);
       fprintf(fidlisting,'      \n '); 
       fclose(fidlisting); 
       disp(['The file "', fileName, '" has been saved']);
    end
   
    %------------------------------------------------------------------------------------------------------------------------------------
    %                                                 Step2_Processing
    %------------------------------------------------------------------------------------------------------------------------------------
    
    if ind_step2 == 1
    
       %--- Load SavefileStep1.mat
       fileName = 'SavefileStep1.mat';
       if isfile(fileName)
           fidlisting=fopen('listing.txt','a+');
           fprintf(fidlisting,'      \n '); 
           fprintf(fidlisting,'  The file "%s" exists\n', fileName);
           fprintf(fidlisting,'      \n '); 
           fclose(fidlisting);   
           disp(['The file "',fileName,'" exists']);  
           
           % Load filename
           load(fileName);
           
           % Restore the current values of the control parameters 'ind_step1','ind_step2','ind_step3','ind_step4','ind_step4_task13',
           %         'ind_step4_task14','ind_step4_task15','ind_SavefileStep3','ind_SavefileStep4'
           load('FileTemporary')
   
           fidlisting=fopen('listing.txt','a+');
           fprintf(fidlisting,'      \n '); 
           fprintf(fidlisting,'  The file "%s" has been loaded\n', fileName);
           fprintf(fidlisting,'      \n '); 
           fclose(fidlisting);
           disp(['The file "',fileName,'" has been loaded']);
       else
           error('STOP14 in mainWorkflow: in Step 2 the file "%s" does not exist.\n', fileName);
       end
   
       if ind_workflow  == 1
          % Load FileDataWorkFlow1Step2.mat
          fileName = 'FileDataWorkFlow1Step2.mat';
          if exist(fileName, 'file') == 2
              load(fileName, ...
                  'ind_display_screen','ind_print','ind_plot','ind_parallel','mDP','ind_generator','ind_basis_type','ind_file_type', ...
                  'epsilonDIFFmin','step_epsilonDIFF','iterlimit_epsilonDIFF','comp_ref','nbMC','icorrectif','f0_ref','ind_f0', ...
                  'coeffDeltar','M0transient','ind_constraints','ind_coupling','iter_limit','epsc','minVarH','maxVarH','alpha_relax1', ...
                  'iter_relax2','alpha_relax2','MatRplotHsamples','MatRplotHClouds','MatRplotHpdf','MatRplotHpdf2D','ind_Kullback', ...
                  'ind_Entropy','ind_MutualInfo');
          else
              error('STOP15 in mainWorkflow: File FileDataWorkFlow1Step2.mat does not exist');
          end
       end
   
       if ind_workflow  == 2
          % Load FileDataWorkFlow2Step2.mat
          fileName = 'FileDataWorkFlow2Step2.mat';
          if exist(fileName, 'file') == 2
              load(fileName, ...
               'ind_display_screen','ind_print','ind_plot','ind_parallel','nbMC','ind_generator','icorrectif','f0_ref','ind_f0', ...
               'coeffDeltar','M0transient','epsilonDIFFmin','step_epsilonDIFF','iterlimit_epsilonDIFF','comp_ref','ind_constraints', ...
               'ind_coupling','iter_limit','epsc','minVarH','maxVarH','alpha_relax1','iter_relax2','alpha_relax2', ...
               'MatRplotHsamples','MatRplotHClouds','MatRplotHpdf','MatRplotHpdf2D','ind_Kullback','ind_Entropy','ind_MutualInfo');
          else
              error('STOP16 in mainWorkflow: File FileDataWorkFlow2Step2.mat does not exist');
          end
       end
       
       if ind_workflow  == 3
          % Load FileDataWorkFlow3Step2.mat
          fileName = 'FileDataWorkFlow3Step2.mat';
          if exist(fileName, 'file') == 2
              load(fileName, ...
               'ind_display_screen','ind_print','ind_plot','ind_parallel','mDP','ind_generator','ind_basis_type','ind_file_type', ...
               'epsilonDIFFmin','step_epsilonDIFF','iterlimit_epsilonDIFF','comp_ref','nbMC','icorrectif','f0_ref','ind_f0', ...
               'coeffDeltar','M0transient','eps_inv','ind_coupling','iter_limit','epsc','alpha_relax1', ...
               'iter_relax2','alpha_relax2','MatRplotHsamples','MatRplotHClouds','MatRplotHpdf','MatRplotHpdf2D','ind_Kullback', ...
               'ind_Entropy','ind_MutualInfo');
          else
              error('STOP17 in mainWorkflow: File FileDataWorkFlow3Step2.mat does not exist');
          end
       end
   
       if ind_display_screen == 1                              
          disp(' ================================ Step2 Processing ================================ ')
       end
       if ind_print == 1
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,' ================================ Step2 Processing ================================ \n ');
          fprintf(fidlisting,'      \n ');  
          fclose(fidlisting);  
       end
   
       %--- Checking that all the required tasks in Step 1 have correctly been executed before executing the tasks of Step2
       if exec_task1 ~= 1 
          error('STOP18 in mainWorkflow: Step 2 cannot be executed because Task1_DataStructureCheck was not correctly executed');
       end
       if exec_task2 ~= 1 
          error('STOP19 in mainWorkflow: Step 2 cannot be executed because Task2_Scaling was not correctly executed');
       end
       if exec_task3 ~= 1 
          error('STOP20 in mainWorkflow: Step 2 cannot be executed because Task3_PCA was not correctly executed');
       end
       if ind_workflow  == 2 
          if exec_task4 ~= 1 
             error('STOP21 in mainWorkflow: Step 2 cannot be executed because Task4_Partition was not correctly executed');
          end
       end 
       if ind_workflow == 3 
          if exec_task8 ~= 1 
             error('STOP22 in mainWorkflow: Step 2 cannot be executed because Task8_ProjectionTarget was not correctly executed');
          end
       end 
       if ind_workflow == 3 
          if exec_task4 ~= 0 
             error('STOP23 in mainWorkflow: Step 2 cannot be executed because Inverse Analysis cannot be done with partition');
          end
       end 
   
       rng(SAVERANDendSTEP1); 
   
       %--- Execution of the tasks of Step2
       if ind_workflow  == 1  || ind_workflow  == 3
   
          % Task5_ProjectionBasisNoPartition
          if ind_generator == 0           
              nbmDMAP = n_d; 
          end
          if ind_generator == 1   
             nbmDMAP = nu + 1;
          end
          [MatRg,MatRa] = sub_projection_basis_NoPartition(nu,n_d,MatReta_d,ind_generator,mDP,nbmDMAP,ind_basis_type, ...
                                  ind_file_type,ind_display_screen,ind_print,ind_plot,ind_parallel,epsilonDIFFmin,step_epsilonDIFF, ...
                                  iterlimit_epsilonDIFF,comp_ref) ; 
          exec_task5 = 1;
       end
       if ind_workflow  == 1 
          % Task6_SolverDirect
          SAVERANDstartDirect = rng;
   
          [n_ar,MatReta_ar,ArrayZ_ar,ArrayWienner,SAVERANDendDirect,d2mopt_ar,divKL,iHd,iHar,entropy_Hd,entropy_Har] = ...
                       sub_solverDirect(nu,n_d,nbMC,MatReta_d,ind_generator,icorrectif,f0_ref,ind_f0,coeffDeltar,M0transient, ...
                                nbmDMAP,MatRg,MatRa,ind_constraints,ind_coupling,iter_limit,epsc,minVarH,maxVarH,alpha_relax1, ...
                                iter_relax2,alpha_relax2,SAVERANDstartDirect,ind_display_screen,ind_print,ind_plot,ind_parallel, ...
                                MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D,ind_Kullback,ind_Entropy,ind_MutualInfo); 
          exec_task6 = 1;
          SAVERANDendSTEP2 = SAVERANDendDirect;
       end
       if ind_workflow  == 2 
          % Task7_SolverDirectPartition
          SAVERANDstartDirectPartition = SAVERANDendSTEP1;
   
          [n_ar,MatReta_ar,SAVERANDendDirectPartition,d2mopt_ar,divKL,iHd,iHar,entropy_Hd,entropy_Har] = ...
             sub_solverDirectPartition(nu,n_d,nbMC,MatReta_d,ind_generator,icorrectif,f0_ref,ind_f0,coeffDeltar,M0transient, ...
                              epsilonDIFFmin,step_epsilonDIFF,iterlimit_epsilonDIFF,comp_ref, ... 
                              ind_constraints,ind_coupling,iter_limit,epsc,minVarH,maxVarH,alpha_relax1,iter_relax2, ...
                              alpha_relax2,ngroup,Igroup,MatIgroup,SAVERANDstartDirectPartition,ind_display_screen,ind_print, ...
                              ind_plot,ind_parallel,MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D, ...
                              ind_Kullback,ind_Entropy,ind_MutualInfo); 
          exec_task7 = 1;
          SAVERANDendSTEP2 = SAVERANDendDirectPartition;
       end
       if ind_workflow  == 3
          % Task9_SolverInverse
          SAVERANDstartInverse = SAVERANDendSTEP1;
          
          [n_ar,MatReta_ar,ArrayZ_ar,ArrayWienner,SAVERANDendInverse,d2mopt_ar,divKL,iHd,iHar,entropy_Hd,entropy_Har] = ...
             sub_solverInverse(nu,n_d,nbMC,MatReta_d,ind_generator,icorrectif,f0_ref,ind_f0,coeffDeltar,M0transient, ...
                              nbmDMAP,MatRg,MatRa,ind_type_targ,N_r,Rb_targ1,coNr,coNr2,MatReta_targ,eps_inv,Rb_targ2,Rb_targ3, ...
                              ind_coupling,iter_limit,epsc,alpha_relax1,iter_relax2,alpha_relax2,SAVERANDstartInverse, ...
                              ind_display_screen,ind_print,ind_plot,ind_parallel,MatRplotHsamples,MatRplotHClouds, ...
                              MatRplotHpdf,MatRplotHpdf2D,ind_Kullback,ind_Entropy,ind_MutualInfo); 
          exec_task9 = 1;
          SAVERANDendSTEP2 = SAVERANDendInverse;
       end
       
       %--- SavefileStep2.mat
       fileName = 'SavefileStep2.mat';  
       if exist(fileName, 'file') == 2
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,'  Warning: The file "%s" already exists and will be overwritten\n', fileName);
          fprintf(fidlisting,'      \n '); 
          fclose(fidlisting); 
          disp(['Warning: The file "', fileName, '" already exists and will be overwritten']);
       end
       save(fileName, '-v7.3');
       fidlisting=fopen('listing.txt','a+');
       fprintf(fidlisting,'      \n '); 
       fprintf(fidlisting,' The file "%s" has been saved\n', fileName);
       fprintf(fidlisting,'      \n '); 
       fclose(fidlisting); 
       disp(['The file "', fileName, '" has been saved']);
    end 
   
    %------------------------------------------------------------------------------------------------------------------------------------
    %                                                 Step3_Post_processing
    %------------------------------------------------------------------------------------------------------------------------------------
    
    if ind_step3 == 1
   
       %--- Load SavefileStep2.mat
       fileName = 'SavefileStep2.mat';
       if isfile(fileName)
           fidlisting=fopen('listing.txt','a+');
           fprintf(fidlisting,'      \n '); 
           fprintf(fidlisting,'  The file "%s" exists\n', fileName);
           fprintf(fidlisting,'      \n '); 
           fclose(fidlisting);   
           disp(['The file "',fileName,'" exists']);
   
           % Load filename
           load(fileName);
           
           % Restore the current values of the control parameters 'ind_step1','ind_step2','ind_step3','ind_step4','ind_step4_task13',
           %         'ind_step4_task14','ind_step4_task15','ind_SavefileStep3','ind_SavefileStep4'
           load('FileTemporary')
           
           fidlisting=fopen('listing.txt','a+');
           fprintf(fidlisting,'      \n '); 
           fprintf(fidlisting,'  The file "%s" has been loaded\n', fileName);
           fprintf(fidlisting,'      \n '); 
           fclose(fidlisting);
           disp(['The file "',fileName,'" has been loaded']);
       else
           error('STOP24 in mainWorkflow: in Step 3 the file "%s" does not exist\n', fileName);
       end
   
       % Load FileDataWorkFlowStep3.mat for ind_workflow = 1, 2, or 3 
       fileName = 'FileDataWorkFlowStep3.mat';
       if exist(fileName, 'file') == 2
           load(fileName, ...
             'ind_display_screen','ind_print','ind_plot','ind_parallel','MatRplotSamples','MatRplotClouds','MatRplotPDF','MatRplotPDF2D');
       else
           error('STOP25 in mainWorkflow: File FileDataWorkFlowStep3.mat does not exist');
       end 
   
       if ind_display_screen == 1                              
          disp(' ================================ Step3 Post_processing ================================ ')
       end
       if ind_print == 1
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,' ================================ Step3 Post_processing ================================ \n ');
          fprintf(fidlisting,'      \n ');  
          fclose(fidlisting);  
       end
   
       %--- Checking that all the required tasks in Step 2 have correctly been executed before executing the tasks of Step3
       if ind_workflow == 1 || ind_workflow == 3
          if exec_task5 ~= 1 
             error('STOP26 in mainWorkflow: Step 3 cannot be executed because Task5_ProjectionBasisNoPartition was not correctly executed');
          end
       end
       if ind_workflow  == 1 
          if exec_task6 ~= 1 
             error('STOP27 in mainWorkflow: Step 3 cannot be executed because Task6_SolverDirect was not correctly executed');
          end
       end 
       if ind_workflow  == 2 
          if exec_task7 ~= 1 
             error('STOP28 in mainWorkflow: Step 3 cannot be executed because Task7_SolverDirectPartition was not correctly executed');
          end
       end 
       if ind_workflow == 3 
          if exec_task9 ~= 1 
             error('STOP29 in mainWorkflow: Step 3 cannot be executed because Task9_SolverInverse was not correctly executed');
          end
       end 
       SAVERANDstartSTEP3 = SAVERANDendSTEP2;
       rng(SAVERANDstartSTEP3); 
   
       %--- Execution of the tasks of Step3
   
       if ind_workflow  == 1 || ind_workflow  == 2 || ind_workflow  == 3 
   
          % Task10_PCAback
          [MatRx_obs] = sub_PCAback(n_x,n_d,nu,n_ar,nx_obs,MatRx_d,MatReta_ar,Indx_obs,RmuPCA,MatRVectPCA, ...
                                    ind_display_screen,ind_print);
          exec_task10 = 1;
   
          % Task11_ScalingBack
          [MatRxx_obs] =  sub_scalingBack(nx_obs,n_x,n_ar,MatRx_obs,Indx_real,Indx_pos,Indx_obs,Rbeta_scale_real,Ralpha_scale_real, ...
                                        Rbeta_scale_log,Ralpha_scale_log,ind_display_screen,ind_print,ind_scaling);
          exec_task11 = 1;
   
          % Task12_PlotXdXar
          sub_plot_Xd_Xar(n_x,n_q,n_w,n_d,n_ar,nu,MatRxx_d,MatRx_d,MatReta_ar,RmuPCA,MatRVectPCA,Indx_real,Indx_pos,nx_obs, ...
                         Indx_obs,ind_scaling,Rbeta_scale_real,Ralpha_scale_real,Rbeta_scale_log,Ralpha_scale_log, ...
                         MatRplotSamples,MatRplotClouds,MatRplotPDF,MatRplotPDF2D,ind_display_screen,ind_print);
          exec_task12 = 1;
       end
   
       SAVERANDendSTEP3 = rng;
   
       %--- SavefileStep3.mat
       fileName = 'SavefileStep3.mat'; 
       if ind_SavefileStep3 == 1
          if exist(fileName, 'file') == 2
             fidlisting=fopen('listing.txt','a+');
             fprintf(fidlisting,'      \n '); 
             fprintf(fidlisting,'  Warning: The file "%s" already exists and will be overwritten\n', fileName);
             fprintf(fidlisting,'      \n '); 
             fclose(fidlisting); 
             disp(['Warning: The file "', fileName, '" already exists and will be overwritten']);
          end
          save(fileName, '-v7.3');
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,'  The file "%s" has been saved\n', fileName);
          fprintf(fidlisting,'      \n '); 
          fclose(fidlisting); 
          disp(['The file "', fileName, '" has been saved']);
       else
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,'  The file "%s" has not been saved\n', fileName);
          fprintf(fidlisting,'      \n '); 
          fclose(fidlisting); 
          disp(['The file "', fileName, '" has notbeen saved']);
       end
    end 
   
    %------------------------------------------------------------------------------------------------------------------------------------
    %                                                 Step4_Conditional_statistics_processing
    %------------------------------------------------------------------------------------------------------------------------------------
    
    if ind_step4 == 1
   
       %--- Load SavefileStep2.mat
       fileName = 'SavefileStep2.mat';
       if isfile(fileName)
           fidlisting=fopen('listing.txt','a+');
           fprintf(fidlisting,'      \n '); 
           fprintf(fidlisting,'  The file "%s" exists\n', fileName);
           fprintf(fidlisting,'      \n '); 
           fclose(fidlisting);   
           disp(['The file "',fileName,'" exists']);
   
           % Load filename
           load(fileName);
           
           % Restore the current values of the control parameters 'ind_step1','ind_step2','ind_step3','ind_step4','ind_step4_task13',
           %         'ind_step4_task14','ind_step4_task15','ind_SavefileStep3','ind_SavefileStep4'
           load('FileTemporary')
   
           fidlisting=fopen('listing.txt','a+');
           fprintf(fidlisting,'      \n '); 
           fprintf(fidlisting,'  The file "%s" has been loaded\n', fileName);
           fprintf(fidlisting,'      \n '); 
           fclose(fidlisting);
           disp(['The file "',fileName,'" has been loaded']);
       else
           error('STOP30 in mainWorkflow: in Step 3 the file "%s" does not exist\n', fileName);
       end
   
       % Load FileDataWorkFlowStep4.mat_task13, task14, or task15 for ind_workflow = 1, 2, or 3 
       if ind_step4_task13 == 1
          fileName = 'FileDataWorkFlowStep4_task13.mat';
          if exist(fileName, 'file') == 2
             load(fileName, ...
                  'ind_display_screen','ind_print','ind_plot','ind_parallel','ind_mean', ...
                  'ind_mean_som','ind_pdf','ind_confregion','nbParam','nbw0_obs','MatRww0_obs','Ind_Qcomp','nbpoint_pdf','pc_confregion');
          else
             error('STOP31 in mainWorkflow: File FileDataWorkFlowStep4_task13.mat does not exist');
          end     
       end
   
       if ind_step4_task14 == 1
          fileName = 'FileDataWorkFlowStep4_task14.mat';
          if exist(fileName, 'file') == 2
             load(fileName, ...
               'ind_display_screen','ind_print','ind_plot','ind_parallel', ...
               'ind_PCE_ident','ind_PCE_compt','nbMC_PCE','Rmu','RM','mu_PCE','M_PCE', ...
               'MatRplotHsamples','MatRplotHClouds','MatRplotHpdf','MatRplotHpdf2D');
          else
             error('STOP32 in mainWorkflow: File FileDataWorkFlowStep4_task14.mat does not exist');
          end     
       end
   
       if ind_step4_task15 == 1
          fileName = 'FileDataWorkFlowStep4_task15.mat';
          if exist(fileName, 'file') == 2
             load(fileName, ...
               'ind_display_screen','ind_print','ind_plot','ind_parallel', ...
               'nbMC_PCE','Ng','Ndeg','ng','ndeg','MaxIter');
          else
             error('STOP33 in mainWorkflow: File FileDataWorkFlowStep4_task15.mat does not exist');
          end      
       end
       
       if ind_display_screen == 1                              
         disp(' ============================ Step4 Conditional statistics processing ============================ ')
       end
       if ind_print == 1
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,' ============================ Step4 Conditional statistics processing ============================ \n ');
          fprintf(fidlisting,'      \n ');  
          fclose(fidlisting);  
       end
   
       %--- Checking that all the required tasks in Step 2 have correctly been executed before executing the tasks of Step4
       if ind_workflow  == 1 
          if exec_task6 ~= 1 
             error('STOP34 in mainWorkflow: Step 4 cannot be executed because Task6_SolverDirect was not correctly executed');
          end
       end 
       if ind_workflow  == 2 
          if exec_task7 ~= 1 
             error('STOP35 in mainWorkflow: Step 4 cannot be executed because Task7_SolverDirectPartition was not correctly executed');
          end
       end
       if ind_workflow == 3 
          if exec_task9 ~= 1 
             error('STOP36 in mainWorkflow: Step 4 cannot be executed because Task9_SolverInverse was not correctly executed');
          end
       end 
       if ind_workflow == 2 && ind_step4_task14 == 1 
          error('STOP37 in mainWorkflow: for ind_workflow = 2 we must have ind_step4_task14 = 0');
       end
   
       SAVERANDstartSTEP4 = SAVERANDendSTEP2;
       rng(SAVERANDstartSTEP4); 
       
       if ind_workflow  == 1 || ind_workflow  == 2 || ind_workflow  == 3
   
          % Task13_ConditionalStatistics
          if ind_step4_task13 == 1
             sub_conditional_statistics(ind_mean,ind_mean_som,ind_pdf,ind_confregion, ...
                                         n_x,n_q,nbParam,n_w,n_d,n_ar,nbMC,nu,MatRx_d,MatRxx_d,MatReta_ar,RmuPCA, ...
                                         MatRVectPCA,Indx_real,Indx_pos,Indq_obs,Indw_obs,nx_obs,Indx_obs,ind_scaling, ...
                                         Rbeta_scale_real,Ralpha_scale_real,Rbeta_scale_log,Ralpha_scale_log, ...
                                         nbw0_obs,MatRww0_obs,Ind_Qcomp,nbpoint_pdf,pc_confregion,ind_display_screen,ind_print);
             exec_task13 = 1;
          end
          
          % Task14_PolynomialChaosZwiener
          if ind_step4_task14 == 1
   
             SAVERANDstartPCE = SAVERANDstartSTEP4;
   
             [MatReta_PCE,SAVERANDendPCE] =  sub_polynomial_chaosZWiener(nu,n_d,nbMC,nbmDMAP,MatRg,MatRa,n_ar,MatReta_ar,ArrayZ_ar, ...
                                          ArrayWienner,icorrectif,coeffDeltar,ind_PCE_ident,ind_PCE_compt,nbMC_PCE,Rmu,RM, ...
                                          mu_PCE,M_PCE,SAVERANDstartPCE,ind_display_screen,ind_print,ind_plot,ind_parallel, ...
                                          MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D);
             exec_task14 = 1;
          end
   
          % Task15_PolynomialChaosQWU
          if ind_step4_task15 == 1
   
             SAVERANDstartPolynomialChaosQWU = SAVERANDstartSTEP4;  
   
             [SAVERANDendPolynomialChaosQWU] = sub_polynomialChaosQWU(n_x,n_q,n_w,n_d,n_ar,nbMC,nu,MatRx_d,MatReta_ar,RmuPCA, ...
                                               MatRVectPCA,Indx_real,Indx_pos,Indq_obs,Indw_obs,nx_obs,Indx_obs,ind_scaling, ...
                                               Rbeta_scale_real,Ralpha_scale_real,Rbeta_scale_log,Ralpha_scale_log,nbMC_PCE, ...
                                               Ng,Ndeg,ng,ndeg,MaxIter,SAVERANDstartPolynomialChaosQWU,ind_display_screen,ind_print, ...
                                               ind_plot,ind_parallel);
             exec_task15 = 1;
          end  
       end
   
       SAVERANDendSTEP4 = rng;
   
       %--- SavefileStep4.mat
       fileName = 'SavefileStep4.mat'; 
       if ind_SavefileStep4 == 1
          if exist(fileName, 'file') == 2
             fidlisting=fopen('listing.txt','a+');
             fprintf(fidlisting,'      \n '); 
             fprintf(fidlisting,'  Warning: The file "%s" already exists and will be overwritten\n', fileName);
             fprintf(fidlisting,'      \n '); 
             fclose(fidlisting); 
             disp(['Warning: The file "', fileName, '" already exists and will be overwritten']);
          end
          save(fileName, '-v7.3');
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,'  The file "%s" has been saved\n', fileName);
          fprintf(fidlisting,'      \n '); 
          fclose(fidlisting); 
          disp(['The file "', fileName, '" has been saved']);
       else
          fidlisting=fopen('listing.txt','a+');
          fprintf(fidlisting,'      \n '); 
          fprintf(fidlisting,'  The file "%s" has not been saved\n', fileName);
          fprintf(fidlisting,'      \n '); 
          fclose(fidlisting); 
          disp(['The file "', fileName, '" has not been saved']);
       end
    end 
   
    % delete temporary file: 'FileTemporary.mat'
    fileName = 'FileTemporary.mat';
    delete(fileName);
return
end
 







