function [MatRy] = sub_polynomialChaosQWU_surrogate(N,n_w,n_y,MatRww,MatRU,Ralpham1_scale_chaos,Rbeta_scale_chaos, ...
                                                  Ng,K0,MatPower0,MatRa0,ng,KU,MatPowerU,MatRaU,Jmax,MatRgamma_opt,Indm,Indk)
%
% Copyright C. Soize 20 April 2023
%
%---INPUT variables
%         N > or = 1 is the number of evaluation performed: MatRqq = PolChaos(MatRww), MatRqq(n_q,N), Rww(n_w,N) with N > or = to 1
%         n_w,n_y        with n_w = Ng
%         MatRww(n_w,N)
%         MatRU(ng,N)
%         Ralpham1_scale_chaos(Ng,1)
%         Rbeta_scale_chaos(Ng,1)
%         Ng,K0
%         MatPower0(K0,Ng)
%         MatRa(K0,K0)
%         ng,KU
%         MatPowerU(KU,ng)
%         MatRaU(KU,KU)
%         Jmax
%         MatRgamma_opt(n_y,Jmax)
%         Indm(Jmax,1)
%         Indk(Jmax,1)
%
%---OUTPUT variable
%         MatRy (n_y,N)

    if Ng ~= n_w
       error('STOP in sub_polynomialChaosQWU_surrogate: Ng must be equal to n_w')
    end
    
    %--- Normalization  of MatRww(n_w,N) into MatRxiNg(n_w,N) by scaling MatRww beteween cmin and cmax 
    MatRxiNg  = Ralpham1_scale_chaos.*(MatRww - repmat(Rbeta_scale_chaos,1,N));  % MatRxiNg(Ng,N),MatRww(n_w,N) with n_w = Ng
    
    %--- construction of monomials MatRMM0(K0,N) including multi-index (0,...,0)
    MatRMM0 = zeros(K0,N);                                       % MatRMM0(K0,N)
    MatRMM0(1,:) = 1;
    for k = 1:(K0-1)
        Rtemp = ones(1,N);                                        % Rtemp(1,N);
        for j = 1:Ng
            Rtemp = Rtemp.*MatRxiNg(j,:).^MatPower0(k,j);         % MatRxiNg(Ng,N),MatPower0(K0,Ng)
        end
        MatRMM0(k+1,:) = Rtemp;                                   % MatRMM0(K0,N)
    end

    %--- computing MatRPsi0(K0,N) = MatRa0(K0,K0)*MatRMM0(K0,N)
    MatRPsi0 = MatRa0*MatRMM0;                                    % MatRPsi0(K0,N)
    clear MatRMM0 Rtemp

       
    %--- construction of monomials MatRMMU(KU,NnbMC0) NOT including multi-index (0,...,0)
    MatRMMU = zeros(KU,N);                                        % MatRMM(KU,N)
    MatRMMU(1,:) = 1;
    for k = 1:(KU-1)
        Rtemp = ones(1,N);                                        % Rtemp(1,N);
        for j = 1:ng
            Rtemp = Rtemp.*MatRU(j,:).^MatPowerU(k,j);            % MatRU(ng,N), MatPowerU(KU,ng)
        end
        MatRMMU(k+1,:) = Rtemp;                                   % MatRMM(KU,N)
    end

    %--- computing MatRphiU(KU,N) = MatRaU(KU,KU)*MatRMMU(KU,N)
    MatRphiU = MatRaU*MatRMMU;           % MatRphiU(KU,N)
    clear MatRMMU Rtemp

    %--- construction of MatRb(Jmax,N) such that MatRb(j,ell) = MatRphiU(m,ell)*MatRPsi0(k,ell)
    MatRb = zeros(Jmax,N);
    for j=1:Jmax
        m = Indm(j);
        k = Indk(j);
        MatRb(j,:) = MatRphiU(m,:).*MatRPsi0(k,:);                % MatRb(Jmax,N)
    end

    %--- computing MatRy(n_y,N)
    MatRy = MatRgamma_opt*MatRb;                                  % MatRy(n_y,N),MatRgamma_opt(n_y,Jmax),MatRb(Jmax,N)  
return 
end

