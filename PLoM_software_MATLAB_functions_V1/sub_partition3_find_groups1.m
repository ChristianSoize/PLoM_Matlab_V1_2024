
function [ngroup,Igroup,mmax,MatIgroup] = sub_partition3_find_groups1(NKL,nr,MatRHexp,INDEPref,ind_parallel)
   %
   % Copyright C. Soize 24 May 2024, revised  3 July 2024
   %
   %---INPUTS 
   %         NKL              : dimension of random vector H
   %         nr               : number of independent realizations of random vector H
   %         MatRHexp(NKL,nr) : nr realizations of H = (H_1,...,H_NKL)
   %         INDEPref         : value of the mutual information to obtain the dependence of H_r1 with H_r2 used as follows.
   %                            Let INDEPr1r2 = i^\nu(H_r1,H_r2) be the mutual information of random variables H_r1 and H_r2
   %                            One says that random variables H_r1 and H_r2 are DEPENDENT if INDEPr1r2 >= INDEPref.
   %         ind_parallel     : = 0 no parallel computing, = 1 parallel computing
   %--- OUPUTS
   %          ngroup                   :  number of constructed independent groups  
   %          Igroup(ngroup)           :  vector Igroup(ngroup,1), mj = Igroup(j),  mj is the number of components of Y^j = (H_jr1,... ,H_jrmj)
   %          mmax                     :  mmax = max_j mj for j = 1, ... , ngroup
   %          MatIgroup(ngroup,mmax)   :  MatIgroup1(j,r) = rj, in which rj is the component of H in group j such that Y^j_r = H_jrj  
   %
   %--- METHOD 
   %
   %    Constructing the groups using a graph approach:
   %    Step 1: computing the number of edges in the graph by analyzing the statistical dependence of the components of the random 
   %            vector H = (H_1,...,H_NKL) by testing the dependence 2 by 2. The test of the independence of two non-Gaussian normalized random 
   %            variables H_r1 et H_r2 is performed by using the MUTUAL INFORMATION criterion that is written as INDEPr1r2 = Sr1 + Sr2 - Sr1r2 
   %            with Sr1 = entropy of H_r1, Sr2 = entropy of H_r2, Sr1r2 =  entropy of (Hr1,Hr2) 
   %            The entropy that is a mathematical expectation of the log of a pdf is estimated with the Monte Carlo method by using the 
   %            same realizations that the one used for estimating the pdf by the Gaussian kernel method.
   %            The random variables H_r1 and H_r2 are assumed to be DEPENDENT if INDEPr1r2 > INDEPref.
   %
   %    Step 2: constructing the groups in exploring the common Nodes to the edges of the graph
   %
   %------------------------------------------------------------------------------------------------------------------------------------------

   %--- STEP 1: constructing the adjacenty matrix MatcurN 
   %            MatcurN(NKL,NKL): symmetric adjacenty matrix such that MatcurN(r1,r2) = 1 if r1 and r2 are the two end nodes of an edge 
                                               
                                  %--- constructing the symmetric adjacency matrix  MatcurN(NKL,NKL) for a given level INDEPref  
                                  %    MatcurN   = zeros(NKL,NKL);
                                  %    for r1 = 1:NKL-1
                                  %        for r2 = r1+1:NKL           
                                  %               [INDEPr1r2] = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2);                      
                                  %               if INDEPr1r2  > INDEPref                              % H_r1 and H_r2 are dependent  
                                  %                  MatcurN(r1,r2)  = 1;
                                  %                  MatcurN(r2,r1)  = 1;
                                  %               end
                                  %        end
                                  %    end   
   np    = NKL*(NKL-1)/2;
   Indr1 = zeros(np,1);
   Indr2 = zeros(np,1);
   p = 0;
   for r1 = 1:NKL-1
       for r2 = r1+1:NKL 
           p = p + 1;
           Indr1(p) = r1;
           Indr2(p) = r2;
       end
   end

   %--- Sequential computation
   if ind_parallel == 0
      RINDEP = zeros(np,1);
      for p = 1:np
          r1 = Indr1(p);
          r2 = Indr2(p);
          [INDEPr1r2] = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2);                      
          RINDEP(p) = INDEPr1r2;
      end
   end
   
   %--- Parallel computation
   if ind_parallel == 1
      RINDEP = zeros(np,1);
      parfor p = 1:np
          r1 = Indr1(p);
          r2 = Indr2(p);
          [INDEPr1r2] = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2);                      
          RINDEP(p) = INDEPr1r2;
      end
   end
   
   MatcurN   = zeros(NKL,NKL);
   for p = 1:np
       r1 = Indr1(p);
       r2 = Indr2(p); 
       INDEPr1r2 = RINDEP(p);
       if INDEPr1r2  > INDEPref                              % H_r1 and H_r2 are dependent  
          MatcurN(r1,r2)  = 1;
          MatcurN(r2,r1)  = 1;
       end
   end
      
   %--- STEP 2:  constructing the groups using a graph algorithm
   igroup    = 0;
   Igroup    = zeros(NKL,1);
   MatIgroup = zeros(NKL,NKL);   
   U         = zeros(NKL,1);                         % if U(r) = 0 , then node r not treated ; if U(r) = 1 , then node r has been treated     
   while isempty(find(U== 0,1)) == 0                 % if  isempty(find(U == 0,1)) == 0 do: then there are nodes that have not been treated 
        U0 = find(U== 0);                            % Nodes that have not been treated 
        x  = U0(1);                                  % node used for starting the construction of a new group
        P  = [];                                     % list of the nodes to be analyzed
        V  = [];                                     % list of the nodes already analyzed
        RS = x;                                      % RS contains the nodes of the present group in construction
        igroup = igroup + 1;
     
        P = union(P,x);
        while isempty(P) == 0                        % P not empty do
              y = P(1);  P(1) = [];                  % load a node and unstack P  
              V = union(V,y);  
              for z = 1:NKL                          % exploring all nodes z such that MatcurN(y,z) ==1 and z not in P union V do
                  if MatcurN(y,z) ==1 && isempty(find(union(P,V) == z,1))==1
                     P  = union(P,z);                % stack (P,z)
                     RS = union(RS,z);               % y belongs to the subset                         
                  end                                % end if    
              end                                    % end for
        end                                          % end while
        m_igroup                     = size(RS,2);
        Igroup(igroup)               = m_igroup;
        MatIgroup(igroup,1:m_igroup) = RS;
        U(RS)                        = 1;            % all nodes in RS have then been treated
        MatcurN(RS,RS)               = 0;            % setting to zero the nodes belonging to the group igroup that has just been indentified
   end                                               % while isempty(find(MatcurN ~= 0)) == 0       
   ngroup = igroup;
   if ngroup < NKL
      Igroup(ngroup+1:NKL) = [];        
      MatIgroup(ngroup+1:NKL,:) = [];
      mmax = max(Igroup);
      MatIgroup(:,mmax+1:NKL) = [];
   end

   if ~exist('ngroup', 'var') || ~exist('Igroup', 'var') || ~exist('mmax', 'var') || ~exist('MatIgroup', 'var')
       error('STOP in sub_partition3_find_groups1: variable mmax does not exist. Remove the greatest value introduced in RINDEPref');
   end

   return   
end
