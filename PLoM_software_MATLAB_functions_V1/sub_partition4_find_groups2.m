
function [ngroup,Igroup,mmax,MatIgroup,nedge,nindep,npair,MatPrintEdge,MatPrintIndep,MatPrintPair,RplotPair] = ...
                                                                    sub_partition4_find_groups2(NKL,nr,MatRHexp,INDEPref,ind_parallel)

   % Copyright C. Soize 24 May 2024 
   
   %---INPUTS 
   %         NKL              : dimension of random vector H
   %         nr               : number of independent realizations of random vector H
   %         MatRHexp(NKL,nr) : nr realizations of H = (H_1,...,H_NKL)
   %         INDEPref         : value of the mutual information to obtain the dependence of H_r1 with H_r2 used as follows.
   %                            Let INDEPr1r2 = i^\nu(H_r1,H_r2) be the mutual information of random variables H_r1 and H_r2
   %                            One says that random variables H_r1 and H_r2 are DEPENDENT if INDEPr1r2 >= INDEPref.
   %         ind_parallel     : = 0 no parallel computing, = 1 parallel computing
   %--- OUPUTS
   %          ngroup                  : number of constructed independent groups  
   %          Igroup(ngroup)          : vector Igroup(ngroup,1), mj = Igroup(j),  mj is the number of components of Y^j = (H_jr1,... ,H_jrmj)
   %          mmax                    : mmax = max_j mj for j = 1, ... , ngroup
   %          MatIgroup(ngroup,mmax)  : MatIgroup1(j,r) = rj, in which rj is the component of H in group j such that Y^j_r = H_jrj     
   %          nedge                   : number of pairs (r1,r2) for which H_r1 and H_r2 are dependent (number of edges in the graph)
   %          nindep                  : number of pairs (r1,r2) for which H_r1 and H_r2 are independent
   %          npair                   : total number of pairs (r1,r2)    = npairmax = NKL(NKL-1)/2
   %          MatPrintEdge(nedge,5)   : such that MatPrintEdge(edge,:)   = [edge  r1 r2 INDEPr1r2 INDEPref]
   %          MatPrintIndep(nindep,5) : such that MatPrintIndep(indep,:) = [indep r1 r2 INDEPr1r2 INDEPref]
   %          MatPrintPair(npair,5)   : such that MatPrintPair(pair,:)   = [pair  r1 r2 INDEPr1r2 INDEPref]
   %          RplotPair(npair,1)      : column matrix  RplotPair(pair,1) = INDEPr1r2 with pair=(r1,r2)
   %
   %--- METHOD 
   %
   %    Constructing the groups using a graph approach:
   %    Step 1: computing the number of edges in the graph by analyzing the statistical dependence of the components of the random 
   %            vector H = (H_1,...,H_NKL) by testing the dependence 2 by 2. The test of the independence of two non-Gaussian normalized random 
   %            variables H_r1 et H_r2 is performed by using the MUTUAL INFORMATION criterion that is written as INDEPr1r2 = Sr1 + Sr2 - Sr1r2 
   %            with Sr1 = entropy of H_r1, Sr2 = entropy of H_r2, Sr1r2 =  entropy of (Hr1,Hr2) 
   %            The entropy that is a mathematical expectation of the log of a pdf is estimated using the same realizations  that the one 
   %            used for estimating the pdf by the Gaussian kernel method.
   %            The random variables H_r1 and H_r2 are assumed to be DEPENDENT if INDEPr1r2 > INDEPref.
   %
   %    Step 2: constructing the groups in exploring the common Nodes to the edges of the graph
   %
   %------------------------------------------------------------------------------------------------------------------------------------------
       
   %--- STEP 1: constructing the symmetric adjacency matrix MatcurN(NKL,NKL) for a given level INDEPref
   %            MatcurN(NKL,NKL)       : symmetric adjacenty matrix such that MatcurN(r1,r2) = 1 if r1 and r2 are the two end nodes of an edge 
   %            MatPrintEdge(nedge,5)  : for print and plot
   %            MatPrintIndep(nindep,5): for print and plot
   %            MatPrintPair(npair,5)  : for print and plot 
   %            RplotPair(npairmax,1)  : for plot 

   npairmax      = NKL*(NKL-1)/2;  
   MatcurN       = zeros(NKL,NKL);
   RplotPair     = zeros(npairmax,1);  
   MatPrintEdge  = zeros(npairmax,5);
   MatPrintIndep = zeros(npairmax,5);
   MatPrintPair  = zeros(npairmax,5);
                              %--- Scalar sequence
                              %    for r1 = 1:NKL-1
                              %        for r2 = r1+1:NKL           
                              %               [INDEPr1r2] = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2);
                              %               pair = pair + 1;
                              %               RplotPair(pair) = INDEPr1r2;
                              %               MatPrintPair(pair,:) = [pair r1 r2 INDEPr1r2 INDEPref];               
                              %               if INDEPr1r2  > INDEPref                   % H_r1 and H_r2 are dependent        
                              %                  edge = edge + 1;
                              %                  MatcurN(r1,r2)  = 1;
                              %                  MatcurN(r2,r1)  = 1;
                              %                  MatPrintEdge(edge,:) = [edge r1 r2 INDEPr1r2 INDEPref];                 
                              %               end
                              %               if INDEPr1r2  <= INDEPref        % H_r1 and H_r2 are independent, just loaded for printing                   
                              %                  indep = indep + 1;
                              %                  MatPrintIndep(indep,:) = [indep r1 r2 INDEPr1r2 INDEPref];                 
                              %               end
                              %        end
                              %    end
   %--- Construct pairs of indices  
   Indr1 = zeros(npairmax,1);
   Indr2 = zeros(npairmax,1);      
   pair  = 0;
   for r1 = 1:NKL-1
       for r2 = r1+1:NKL 
           pair        = pair + 1;
           Indr1(pair) = r1;
           Indr2(pair) = r2;
       end
   end  

   %--- Sequential computation
   if ind_parallel == 0
      for pair = 1:npairmax
          r1              = Indr1(pair);
          r2              = Indr2(pair);
          [INDEPr1r2]     = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2);                      
          RplotPair(pair) = INDEPr1r2;
      end
   end
   
   %--- Parallel computation
   if ind_parallel == 1
      parfor pair = 1:npairmax
          r1              = Indr1(pair);
          r2              = Indr2(pair);
          [INDEPr1r2]     = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2);                      
          RplotPair(pair) = INDEPr1r2;
      end
   end

   %--- Populate the adjacency matrix and pair information
   edge  = 0;
   indep = 0;
   for pair = 1:npairmax
       r1                   = Indr1(pair);
       r2                   = Indr2(pair); 
       INDEPr1r2            = RplotPair(pair);
       MatPrintPair(pair,:) = [pair r1 r2 INDEPr1r2 INDEPref];      
       if INDEPr1r2  > INDEPref       % H_r1 and H_r2 are dependent        
          edge = edge + 1;
          MatcurN(r1,r2)  = 1;
          MatcurN(r2,r1)  = 1;
          MatPrintEdge(edge,:) = [edge r1 r2 INDEPr1r2 INDEPref];                 
       end
       if INDEPr1r2  <= INDEPref       % H_r1 and H_r2 are independent, just loaded for printing                   
          indep = indep + 1;
          MatPrintIndep(indep,:) = [indep r1 r2 INDEPr1r2 INDEPref];                 
      end
   end  
   
   %--- Adjust the matrix sizes based on actual edges and independent pairs                            
   nedge  = edge;                          % number of pairs (r1,r2) for which H_r1 and H_r2 are dependent (number of edges in the graph)
   nindep = indep;                         % number of couples (r1,r2) for which H_r1 and H_r2 are independent
   npair  = npairmax;                      % total number of pairs = npairmax  
   if nedge < npairmax                     % adjusting the matrices dimensions    
      MatPrintEdge(nedge+1:npairmax,:)=[];
   end
   if nindep < npairmax
      MatPrintIndep(nindep+1:npairmax,:)=[];
   end
      
   %--- STEP 2: constructing the groups using a graph algorithm
    igroup    = 0;
    Igroup    = zeros(NKL,1);
    MatIgroup = zeros(NKL,NKL);   
    U  = zeros(NKL,1);                             % if U(r) = 0 , then node r not treated ; if U(r) = 1 , then node r has been treated     
    while isempty(find(U== 0,1)) == 0              % if isempty(find(U == 0,1)) == 0 do: then there are nodes that have not been treated 
         U0 = find(U== 0);                         % Nodes that have not been treated 
         x  = U0(1);                               % node used for starting the construction of a new group
         P  = [];                                  % list of the nodes to be analyzed
         V  = [];                                  % list of the nodes already analyzed
         RS = x;                                   % RS contains the nodes of the present group in construction
         igroup = igroup + 1;
      
         P = union(P,x);
         while isempty(P) == 0                     % P not empty do
               y = P(1);  P(1) = [];               % load a node and unstack P  
               V = union(V,y);  
               for z = 1:NKL                       % exploring all nodes z such that MatcurN(y,z) ==1 and z not in P union V do
                   if MatcurN(y,z) ==1 && isempty(find(union(P,V) == z,1))==1
                      P  = union(P,z);             % stack (P,z)
                      RS = union(RS,z);            % y belongs to the subset                         
                   end                             % end if    
               end                                 % end for
         end                                       % end while
         m_igroup                     = size(RS,2);
         Igroup(igroup)               = m_igroup;
         MatIgroup(igroup,1:m_igroup) = RS;
         
         U(RS)          = 1;                       % all nodes in RS have then been treated
         MatcurN(RS,RS) = 0;                       % setting to zero the nodes belonging to the group igroup that has just been indentified
    end                                            % while isempty(find(MatcurN ~= 0)) == 0  

    ngroup = igroup;
    if ngroup < NKL
       Igroup(ngroup+1:NKL)      = [];        
       MatIgroup(ngroup+1:NKL,:) = [];
       mmax                      = max(Igroup);
       MatIgroup(:,mmax+1:NKL)   = [];
    end
return                                                                        
