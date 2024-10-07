function [MatRLrond] = sub_solverDirect_Lrond_constraint2(nu,n_d,MatReta_d,MatRa,MatRg,MatRZ,shss,sh,Rlambda_iter)
    
    %-------------------------------------------------------------------------------------------------------------------------------------------
    %
    %  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
    %
    %  Software     : Probabilistic Learning on Manifolds (PLoM) 
    %  Function name: sub_solverDirect_Lrond_constraint2
    %  Subject      : ind_constraints = 2 : constraints E{H_{ar,j}} = 0  and E{H_{ar,j}^2} = 1 for j = 1,...,nu  
    %
    %--- INPUTS
    %      nu                  : dimension of H_d and H_ar   
    %      n_d                 : number of realizations in the training dataset
    %      MatReta_d(nu,n_d)   : n_d realizations of H_d (training)  
    %      MatRa(n_d,nbmDMAP)  : related to MatRg(n_d,nbmDMAP) 
    %      MatRg(n_d,nbmDMAP)  : matrix of the ISDE-projection basis
    %      MatRZ(nu,nbmDMAP)   : projection of MatRH_ar on the ISDE-projection basis
    %      shss,sh             : parameters of the GKDE of pdf of H_d (training)
    %      Rlambda_iter(mhc,1) = (lambda_1,...,lambda_nu , lambda_{1+nu},...,lambda_{nu+nu})
    %
    %--- OUTPUTS
    %      MatRLrond(nu,nbmDMAP)
    %
    %--- INTERNAL PARAMETERS
    %      s       = ((4/((nu+2)*n_d))^(1/(nu+4)))   % Silver bandwidth 
    %      s2      = s*s
    %      shss    = 1 if icorrectif = 0 and = 1/sqrt(s2+(n_d-1)/n_d) if icorrectif = 1
    %      sh      = s*shss     
    %      nbmDMAP : number of ISDE-projection basis 
    %      mhc = 2*nu + nu*(nu-1)/2 (note that for nu =1, we have mhc = 2*nu) 
       
    sh2   = sh*sh;
    sh2m1 = 1/sh2;
    co1   = 1/(2*sh2);
    co2   = shss/(sh2*n_d);

    MatRU       = MatRZ*MatRg';            % MatRU(nu,n_d), MatRg(n_d,nbmDMAP), MatRZ(nu,nbmDMAP)
    MatRetashss = shss*MatReta_d;          % MatReta_d(nu,n_d)
    MatRetaco2  = co2*MatReta_d;           % MatRetaco2(nu,n_d)
                                                             
    %---Vectorial sequence without  parallelization
    MatRL  = zeros(nu,n_d);                                  % MatRL(nu,n_d)
    MatRLc = zeros(nu,n_d);                                  % MatRLc(nu,n_d)
    for ell=1:n_d
        RU = MatRU(:,ell);                                   % RU(nu,1),MatRU(nu,n_d)
        MatRexpo   = MatRetashss-RU;                         % MatRexpo(nu,n_d), RU(nu,1)
        Rexpo      = co1*(sum(MatRexpo.^2,1));               % Rexpo(1,n_d) 
        expo_min   = min(Rexpo);             
        RS         = exp(-(Rexpo-expo_min));                 % RS(1,n_d),Rexpo(1,n_d)
        q          = sum(RS)/n_d;     
        Rgraduqp   = MatRetaco2*RS';                         % Rgraduqp(nu,1)
        RL         = -sh2m1*RU + Rgraduqp/q;                 % Rgraduqp(nu,1)
        MatRL(:,ell) = RL;                                   % MatRL(nu,n_d)
        
        MatRLc(:,ell) = - (Rlambda_iter(1:nu,1) + 2*Rlambda_iter(1+nu:nu+nu,1).*RU);  % MatRLc(nu,n_d), Rlambda_iter(mhc,1), RU(nu,1)
    end
    MatRLrond = (MatRL+MatRLc)*MatRa;                        % MatRLrond(nu,m),MatRL(nu,n_d),MatRLc(nu,n_d),MatRa(n_d,nbmDMAP)
    return 
end
