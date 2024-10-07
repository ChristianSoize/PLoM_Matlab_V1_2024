function [RerrorOVL] = sub_polynomialChaosQWU_OVL(n_q,N_Ref,MatRqq_Ref,N,MatRqq,nbpoint,MatRxipointOVL,RbwOVL)

%----------------------------------------------------------------------------------------------------------------------------------------------
%          Copyright C. Soize, 05 October 2024
%          OVERLAPPING COEFFICIENT
%----------------------------------------------------------------------------------------------------------------------------------------------
%
%          MatRqq(n_q,N)               = MatRqqC(nbqqC,NnbMC0),     N = NnbMC0
%          MatRqq_Ref(n_q,N_Ref)       = MatRqq_ar0C(nbqqC,NnbMC0), N_ref = NnbMC0
%          MatRxipointOVL(n_q,nbpoint) = MatRxipointOVLC(nbqqC,nbpoint)
%          RbwOVL(n_q,1)               = RbwOVLC(nbqqC,1)
 
   
    RerrorOVL = zeros(n_q,1); 
    for iq = 1:n_q  
        Rxipoint = MatRxipointOVL(iq,:);
        Rqq  = MatRqq(iq,:)';                                                      % Rqq(N,1),MatRqq(n_q,N) 
        Ind  = find(abs(Rqq) > 1e-10);                                             % Removing the 0 values for using ksdensity 
        Rqqb = Rqq(Ind);     
        [Rpdf,RqqVal] = ksdensity(Rqqb,Rxipoint,'Bandwidth',RbwOVL(iq,1));         % RqqVal(1,nbpoint), Rpdf(1,nbpoint)   
        RqqRef  = MatRqq_Ref(iq,:)';                                               % RqqRef(N_Ref,1),MatRqq_Ref(n_q,N_Ref)
        IndRef  = find(abs(RqqRef) > 1e-10);                                       % Removing the 0 valuesS for using ksdensity 
        RqqRefb =  RqqRef(IndRef); 
        [RpdfRef,RqqRefVal] = ksdensity(RqqRefb,Rxipoint,'Bandwidth',RbwOVL(iq,1)); % RqqRefVal(1,nbpoint), RpdfRef(1,nbpoint)  
        MAX = max(RqqVal(1,1),RqqRefVal(1,1));
        if MAX <= 0
           qqmin = 0.999*MAX; 
        else 
           qqmin = 1.001*MAX; 
        end
        MIN= min(RqqVal(1,nbpoint),RqqRefVal(1,nbpoint));
        if MIN <= 0
           qqmax = 1.001*MIN; 
        else 
           qqmax = 0.999*MIN; 
        end  
        nbint      = nbpoint + 1;                                    
        pasqq      = 0.99999*(qqmax-qqmin)/(nbint-1);                   
        Rqqi       = qqmin - pasqq + (1:1:nbint)*pasqq;                 % Rqqi(1,nbint)  
        Ryi        = interp1(RqqVal,Rpdf,Rqqi);                         % Ryi(1,nbint)  
        RyiRef     = interp1(RqqRefVal,RpdfRef,Rqqi);                   % RyiRef(1,nbint)
        deni       = pasqq*sum(abs(Ryi)); 
        RpdfQQi    = abs(Ryi)/deni;
        deniRef    = pasqq*sum(abs(RyiRef)); 
        RpdfQQiRef = abs(RyiRef)/deniRef;
        RerrorOVL(iq) = 1 - 0.5*pasqq*sum(abs(RpdfQQi - RpdfQQiRef));   % RerrorOVL(n_q,1)
    end
 return