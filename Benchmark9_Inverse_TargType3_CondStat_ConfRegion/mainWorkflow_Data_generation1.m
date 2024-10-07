
function [n_q,Indq_real,Indq_pos,n_w,Indw_real,Indw_pos,Indq_obs,Indw_obs,n_d,MatRxx_d] = mainWorkflow_Data_generation1
        
   %===================================================================================================================================
   %
   %  Copyright: Christian Soize, Universite Gustave Eiffel, 19 July 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: mainWorkflow_Data_generation1
   %  Subject      : This function allows to generate data for the Benchmark9_Inverse_TargType3_CondStat_ConfRegion
   %
   %
   %--- OUTPUT (data that must be generated, which are application dependent) 
   %
   %          n_q                    : dimension of random vector QQ (unscaled quantitity of interest)  1 <= n_q     
   %          Indq_real(nqreal,1)    : contains the nqreal component numbers of QQ, which are real (positive, negative, or zero) 
   %                                   with 0 <= nqreal <=  n_q and for which a "standard scaling" will be used
   %          Indq_pos(nqpos,1)      : contains the nqpos component numbers of QQ, which are strictly positive a "specific scaling"
   %                                   with  0 <= nqpos <=  n_q  and for which the scaling is {log + "standard scaling"}
   %                                   --- we must have n_q = nqreal + nqpos  
   %
   %          n_w                    : dimension of random vector WW (unscaled control variable) with 1 <= n_w  
   %          Indw_real(nwreal,1)    : contains the nwreal component numbers of WW, which are real (positive, negative, or zero) 
   %                                   with 0 <= nwreal <=  n_w and for which a "standard scaling" will be used
   %          Indw_pos(nwpos,1)      : contains the nwpos component numbers of WW, which are strictly positive a "specific scaling"
   %                                   with  0 <= nwpos <=  n_w  and for which the scaling is {log + "standard scaling"}
   %                                  --- we must have n_w = nwreal + nwpos 
   %
   %          Indq_obs(nq_obs,1)     : nq_obs component numbers of QQ that are observed , 1 <= nq_obs <= n_q
   %          Indw_obs(nw_obs,1)     : nw_obs component numbers of WW that are observed,  1 <= nw_obs <= n_w
   %                                   
   %                     WARNING: if Inverse Analysis is considered (ind_workflow = 3) with the option
   %                              ind_type_targ = 2 or 3, then all the components of XX and XX_targ
   %                              must be considered as real even if some components are positive. When thus must have:
   %                                    nqreal     = n_q and nwreal     = n_w
   %                                    nqpos      = 0   and nwpos      = 0
   %                                    nqpos_targ = 0   and nwpos_targ = 0
   %
   %                     WARNING: For the analysis of the conditional statistics of Step4, the organization of the components of the 
   %                              QQ vector of the quantity of interest QoI is as follows (this organization must be planned from the 
   %                              creation of the data in this function "mainWorkflow_Data_generation1.m" and  also in
   %                              "mainWorkflow_Data_generation2. m" .
   %
   %                              If the QoI depends on the sampling in nbParam points of a physical system parameter
   %                              (such as time or frequency), if QoI_1, QoI_2, ... are the scalar quantities of interest, and if 
   %                              f_1,...,f_nbParam are the nbParam sampling points, the components of the QQ vector must be organized 
   %                              as follows: 
   %                              [(QoI_1,f_1) , (QoI_1,f_2), ... ,(QoI_1,f_nbParam), (QoI_2,f_1), (QoI_2,f_2), ... ,(QoI_2,f_nbParam), ... ]'.
   %
   %                     WARNING: If nbParam > 1, this means that nq_obs is equal to nqobsPhys*nbParam, in which nqobsPhys is the number
   %                              of the components of the state variables that are observed. Consequently, nq_obs/nbParam must be 
   %                              an integer, if not there is an errod in the Data generation in "mainWorkflow_Data_generation1.m" and 
   %                              "mainWorkflow_Data_generation2. m" 
   %
   %                     WARNING: NOTE THAT if such a parameter does not exist, it must be considered that nbParam = 1, but the 
   %                              information structure must be consistent with the case nbParam > 1.  
   %
   %
   %          n_d                     : number of points in the training set for XX_d and X_d
   %          MatRxx_d(n_x,n_d)       : n_d realizations of random vector XX_d (unscale) with dimension n_x = n_q + n_w
   %
   %                                  WARNING: Matrix MatRxx_d(n_x,n_d) musy be coherent with the following construction 
   %                                           of the training dataset:
   %                                                                   MatRxx_d = [MatRqq_d     % MatRqq_d(n_q,n_d) 
   %                                                                               MatRww_d];   % MatRww_d(n_w,n_d) 
   %
   %======================================================================================================================================

   %--- load data 
   %                       
   load FileMatRxxData 
        % Name                Size          Bytes  
        % MatRqq_data      400x200            640000              
        % MatRww_data        2x200              3200    

        % NOTE : For column j of realizations j = 1,...,n_d, the information of column MatRqq_data(:,j) are organized as follows
        %        in which QoI_1,...,QoI_4 are the scalar quantities of interest, where f_1,...,f_nbParam are the nbParam 
        %        sampling points. The components of the QQ vector are organized (IT IS MANDATORY) as follows: 
        %        [(QoI_1,f_1) , (QoI_1,f_2), ... ,(QoI_1,f_nbParam), (QoI_2,f_1), (QoI_2,f_2), ... ,(QoI_2,f_nbParam), ... ]'
        %        For generating the training of WW = (WW_1,WW_2), 
        %        WW_1 is uniform on [0.9,1.1]
        %        WW_2 is uniform on [0.003,0.007] 
  
   n_d      = 200;    % number of independent realizations generated with the prior probability model for the training  
   nbParam  = 100;    % number of sampling points
   n_QoI    = 4;      % number of QoI  
   n_q      = 400;    % n_q = n_QoI*nbParam  
   n_w      = 2;      % n_w = number of control parameters
   n_x      = 402;

   %--- loading MatRqq_d(n_q,n_d) and MatRww_d(n_w,n_d)
   MatRqq_d = MatRqq_data;    % MatRqq_d(n_q,n_d)
   MatRww_d = MatRww_data;    % MatRww_d(n_w,n_d)

   %--- loading MatRxx_d(n_x,n_d) for PLoM that is constructed with the convention RXX = (RQQ,RWW)
   MatRxx_d = [MatRqq_d     % MatRqq_d(n_q,n_d)
               MatRww_d];   % MatRww_d(n_w,n_d)
   
   %--- loading Indq_real(nqreal,1) and Indq_pos(nqpos,1)
   Indq_real = (1:n_q)';
   Indq_pos  = [];
   
   %--- loading Indw_real(nwreal,1) and Indw_pos(nwpos,1)
   Indw_real = [1 2]';
   Indw_pos  = []';

   %--- loading Indq_obs(nq_obs,1) and Indw_obs(nw_obs,1): observations are QoI_1 and QoI_3
   startQoI1 = 1; endQoI1 = 100; startQoI3 = 201; endQoI3 = 300;
   Indq_obs = [(startQoI1:endQoI1)'     % nq_obs component numbers of QQ that are observed , 1 <= nq_obs <= n_q
               (startQoI3:endQoI3)'];
   Indw_obs = [1 2]';       % nw_obs component numbers of WW that are observed,  1 <= nw_obs <= n_w

   return
end
      