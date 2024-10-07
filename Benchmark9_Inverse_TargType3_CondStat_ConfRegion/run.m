% run

%----------------------------------------------------------------------------------------------------------------------------------------
%
%  Copyright: Christian Soize, Universite Gustave Eiffel, 5 October 2024 2024
%
%  Software     : Probabilistic Learning on Manifolds (PLoM) 
%  Function name: run
%  Subject      : This function allows the job to be started
%                 The user must indicate the value of nbworkers and the value of ind_workflow
%----------------------------------------------------------------------------------------------------------------------------------------

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
%----------------------------------------------------------------------------------------------------------------------------------------

clear
close all  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     BEGIN USER DATA DEFINITION SEQUENCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--- Managing the number of workers when parallel computation is used in the job
%    nbworkers = 0 : no creation of workers
%              >=1 : creation of nbworkers

nbworkers = 96;    % The user enters either 0 or the desired number of workers

%--- Managing the type of workflow performed in the job (the description of the workflow is provided in the function mainWorkflow.m)
%    ind_workflow = 1: WorkFlow1_SolverDirect_WithoutPartition
%                 = 2: WorkFlow2_SolverDirect_WithPartition
%                 = 3: WorkFlow3_SolverInverse_WithoutPartition 

ind_workflow = 3;  % The user enters a desired integer value, which must be either 1, 2, or 3   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                           END USER DATA DEFINITION SEQUENCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               BEGIN CODE SEQUENCE - DO NOT MODIFY UNTIL THE END OF THIS FUNCTION   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(fullfile(pwd, '..', 'PLoM_software_MATLAB_functions_V1'));


if nbworkers >= 1
   sub_setupParallelPool(nbworkers)
end

if ind_workflow == 1
   mainWorkflow(@mainWorkflow_Data_WorkFlow1,ind_workflow); % Pass the function mainWorkflow_Data_WorkFlow1 as an argument
end
if ind_workflow == 2
   mainWorkflow(@mainWorkflow_Data_WorkFlow2,ind_workflow); % Pass the function mainWorkflow_Data_WorkFlow2 as an argument
end
if ind_workflow == 3
   mainWorkflow(@mainWorkflow_Data_WorkFlow3,ind_workflow); % Pass the function mainWorkflow_Data_WorkFlow3 as an argument
end
