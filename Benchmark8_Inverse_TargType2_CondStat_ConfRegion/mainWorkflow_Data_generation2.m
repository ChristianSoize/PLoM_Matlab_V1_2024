
function [ind_type_targ,Indq_targ_real,Indq_targ_pos,Indw_targ_real,Indw_targ_pos,N_r, ...
          MatRxx_targ_real,MatRxx_targ_pos,Rmeanxx_targ,MatRcovxx_targ] = mainWorkflow_Data_generation2
          
   %===================================================================================================================================
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 19 July 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: mainWorkflow_Data_generation2
   %  Subject      : This function allows to generate data for Benchmark8_Inverse_TargType2_CondStat_ConfRegion
   %
   %--- OUTPUT (data that must be generated for ind_workflow = 3) ---------------------------------------------------------
   %
   %   ind_type_targ  : = 1, targets defined by giving N_r realizations
   %                  : = 2, targets defined by giving the target mean value of the unscaled XX_targ 
   %                  : = 3, targets defined by giving the target mean value and the target covaraince matrix of the unscaled XX_targ 
   %
   %--- WARNING 1: all the following parameteres and arrays: ind_type_targ,Indq_targ_real,Indq_targ_pos,Indw_targ_real,Indw_targ_pos,N_r, 
   %               MatRxx_targ_real,MatRxx_targ_pos,Rmeanxx_targ,MatRcovxx_targ, must be defined as explained below.
   %
   %    WARNING 2: if an ArrayName is not used or empty, for the case considered, ArrayName = []
   %
   %    WARNING 3: about the organization of data:
   %               Indx_targ_real = [Indq_targ_real           % Indx_targ_real(nbreal_targ,1)
   %                                 n_q + Indw_targ_real];   % nbreal_targ component numbers of XX, which are real
   %
   %               Indx_targ_pos  = [Indq_targ_pos            % Indx_targ_pos(nbpos_targ,1)
   %                                 n_q + Indw_targ_pos];    % nbpos_targ component numbers of XX, 
   %                                                          % which are strictly positive
   %               nx_targ        = nbreal_targ + nbpos_targ; % dimension of random vector XX_targ = (QQ_targ,WW_targ)
   %               Indx_targ      = [Indx_targ_real           % nx_targ component numbers of XX_targ 
   %                                 Indx_targ_pos];          % for which a target is given 
   %
   %    WARNING 4: For the analysis of the conditional statistics in Step4, the organization of the components of the 
   %               QQ vector of the quantity of interest QoI is as follows: this organization must be planned from the 
   %               creation of the data, not olny in this function "mainWorkflow_Data_generation2.m but also in
   %               "mainWorkflow_Data_generation1.m".
   %
   %               If the QoI depends on the sampling in nbParam points of a physical system parameter
   %               (such as time or frequency), if QoI_1, QoI_2, ... are the scalar quantities of interest, and if 
   %               f_1,...,f_nbParam are the nbParam sampling points, the components of the QQ vector must be organized 
   %               as follows: 
   %                          [(QoI_1,f_1) , (QoI_1,f_2), ... ,(QoI_1,f_nbParam), (QoI_2,f_1), (QoI_2,f_2), ... ,(QoI_2,f_nbParam), ... ]'.
   %
   %               If nbParam > 1, this means that nq_obs is equal to nqobsPhys*nbParam, in which nqobsPhys is the number
   %                               of the components of the state variables that are observed. Consequently, nq_obs/nbParam must be 
   %                               an integer, if not there is an error in the given value of nbParam of in the Data generation in 
   %                               "mainWorkflow_Data_generation1.m" and "mainWorkflow_Data_generation2.m" 
   %
   %               NOTE THAT if such a physical system parameter is not considered, then nbParam = 1, but the 
   %                         information structure must be consistent with the case nbParam > 1.  
   %
   %--- DATA FOR THE CASE ind_type_targ = 1 --------------------------------------------------------------------------------------------------
   %
   %          ind_type_targ  = 1;
   %
   %          Indq_targ_real(nqreal_targ,1): nqreal_targ component numbers of QQ for which a target is real, 0 <= nqreal_targ <= n_q
   %          Indq_targ_pos(nqpos_targ,1)  : nqpos_targ  component numbers of QQ for which a target is positive, 0 <= nqpos_targ <= n_q
   %
   %          Indw_targ_real(nwreal_targ,1): nwreal_targ component numbers of WW for which a target is real, 0 <= nwreal_targ <= n_w
   %          Indw_targ_pos(nwpos_targ,1)  : nwpos_targ  component numbers of WW for which a target is positive, 0 <= nwpos_targ <= n_w
   %
   %          N_r  : number of target realizations  
   %                                                         
   %          MatRxx_targ_real(nbreal_targ,N_r) : N_r realizations (unscaled) of the nbreal_targ targets of XX that are real
   %          MatRxx_targ_pos(nbpos_targ,N_r)   : N_r realizations (unscaled) of the nbpos_targ targets of XX that are positive
   % 
   %          Rmeanxx_targ   = [];
   %          MatRcovxx_targ = [];
   %
   %--- DATA FOR THE CASE ind_type_targ = 2 --------------------------------------------------------------------------------------------------
   %
   %          ind_type_targ  = 2;
   %
   %          Indq_targ_real(nqreal_targ,1): nqreal_targ component numbers of QQ for which a target is real, 1 <= nqreal_targ <= n_q
   %          Indq_targ_pos(nqpos_targ,1)  = [];
   %
   %          Indw_targ_real(nwreal_targ,1): nwreal_targ component numbers of WW for which a target is real, 0 <= nwreal_targ <= n_w
   %          Indw_targ_pos(nwpos_targ,1)  = [];
   %
   %          N_r = 0;  
   %                                                         
   %          MatRxx_targ_real(nbreal_targ,N_r) = [];
   %          MatRxx_targ_pos(nbpos_targ,N_r)   = [];
   % 
   %          Rmeanxx_targ(nx_targ,1) : target mean value of the unscaled XX_targ 
   %          MatRcovxx_targ = [];
   %
   %--- DATA FOR THE CASE ind_type_targ = 3 --------------------------------------------------------------------------------------------------
   %
   %          ind_type_targ  = 3;
   %
   %          Indq_targ_real(nqreal_targ,1): nqreal_targ component numbers of QQ for which a target is real, 1 <= nqreal_targ <= n_q
   %          Indq_targ_pos(nqpos_targ,1)  = [];
   %
   %          Indw_targ_real(nwreal_targ,1): nwreal_targ component numbers of WW for which a target is real, 0 <= nwreal_targ <= n_w
   %          Indw_targ_pos(nwpos_targ,1)  = [];
   %
   %          N_r = 0;  
   %                                                         
   %          MatRxx_targ_real(nbreal_targ,N_r) = [];
   %          MatRxx_targ_pos(nbpos_targ,N_r)   = [];
   % 
   %          Rmeanxx_targ(nx_targ,1)           : target mean value of the unscaled XX_targ 
   %          MatRcovxx_targ(nx_targ,nx_targ)   : target covariance matrix of the unscaled XX_targ 
   %
   %======================================================================================================================================

   ind_type_targ = 2;    % = 1, targets defined by giving N_r realizations
                         % = 2, targets defined by giving the target mean value of the unscaled XX_targ 
                         % = 3, targets defined by giving the target mean value and the target covarianve matrix of the unscaled XX_targ
   %--- load data 
   %                       
   load FileMatRxxTarget2 
        % Name                 Size             Bytes  
        % Rmeanxx_targ_data    402x1            3216          

        % NOTE : 1) For column r of realizations j = 1,...,maxN_r, the information of column MatRqq_target_data(:,r) are organized as follows
        %        in which QoI_1,...,QoI_4 are the scalar quantities of interest, where f_1,...,f_nbParam are the nbParam 
        %        sampling points. The components of the QQ vector are organized (IT IS MANDATORY) as follows: 
        %        [(QoI_1,f_1) , (QoI_1,f_2), ... ,(QoI_1,f_nbParam), (QoI_2,f_1), (QoI_2,f_2), ... ,(QoI_2,f_nbParam), ... ]'
        %        For generating the training of WW = (WW_1,WW_2), 
        %        WW_1 is uniform on [0.9,1.1]
        %        WW_2 is uniform on [0.003,0.007] 
        %        2) The observations considered in mainWorkflow_Data_generation1.m sont QoI_1 and QoI_3
        %        For the updating (Inverse problem) the targets are given for QoI_2 and QoI_4 in order to not observe the QoI for which
        %        the target is given

        % n_d      = 200;    % number of independent realizations generated with the prior probability model for the training  
        % nbParam  = 100;    % number of sampling points
        % n_QoI    = 4;      % number of QoI  
        n_q        = 400;    % n_q = n_QoI*nbParam  
        n_w        = 2;      % n_w = number of control parameters
        % n_x      = 402;  

        %--- Loading Indq_targ_real(nqreal_targ,1) and Indq_targ_pos(nqpos_targ,1) for QoI_2 and QoI_4 
        startQoI2 = 101; endQoI2 = 200; startQoI4 = 301; endQoI4 = 400;
        Indq_targ_real = [(startQoI2:endQoI2)'      % Indq_targ_real(nqreal_targ,1)
                          (startQoI4:endQoI4)'];    
        Indq_targ_pos = [];                         % Indq_targ_pos(nqpos_targ,1)

        %--- Loading Indw_targ_real(nwreal_targ,1) and Indw_targ_pos(nwpos_targ,1)
        Indw_targ_real = [1 2]';                    % Indw_targ_real(nwreal_targ,1)
        Indw_targ_pos  = [];                        % Indw_targ_pos(nwpos_targ,1)

        %--- Loading Indx_targ(nx_targ,1)
        Indq_targ = [Indq_targ_real
                     Indq_targ_pos];
        Indw_targ = [Indw_targ_real
                     Indw_targ_pos];
        Indx_targ = [Indq_targ
                     n_q +  Indw_targ];

        %---- Initializing parameter N_r to 0
        N_r = 0;

        %--- Loading MatRxx_targ_real and MatRxx_targ_pos
        MatRxx_targ_real = []; 
        MatRxx_targ_pos  = []; 

        % --- ind_type_targ = 2: targets defined by giving the target mean value of the unscaled XX_targ   
        Rmeanxx_targ = Rmeanxx_targ_data(Indx_targ,1);  % Rmeanxx_targ(nx_targ,1): nx_targ components of mean value E{XX_targ} 
        MatRcovxx_targ = [];        
   return
end
      