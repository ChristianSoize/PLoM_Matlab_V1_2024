function [Rpdf] = sub_partition10_ksdensity_mult(n,nsim,dim,nexp,MatRxData,MatRxExp)
   
   % Copyright C. Soize 24 May 2024 
   %
   
   % This function calculates the value, in nexp experimental vectors (or scalars if dim=1) x^1,...,x^nexp stored in 
   % the array MatRxExp(dim,nexp), of the joint probability density function of a random vector X of dimension dim, 
   % CONSIDERED AS THE MARGINAL DISTRIBUTION OF A RANDOM VECTOR OF DIMENSION n >= dim,
   % whose nsim realizations: X(theta_1),..., X(theta_nsim) are used to estimate the marginal probability density function 
   % using the Gaussian kernel method, and are stored in the array MatRxData(dim,nsim).   
   %
   %  Rhatsigma(j)       = empirical standard deviation of X_j estimated with MatRxData(j,:) = [X(theta_1),..., X(theta_nsim)]
   %  s                  = (4/(nsim*(2+n)))^(1/(n+4)), Silverman bandwidth for random vector of dimension n (and not ndim)
   %  Rsigma(j)          = Rhatsigma(j) * s   
   %  shss               = 1/sqrt(s2+(nsim-1)/nsim), coef correcteur sh et sh/s introduced by C. Soize
   %  sh                 = s*shss;    
   %  MatRxExpNorm(j,:)  = MatRxExp(j,:)/Rhatsigma(j);                       
   %  MatRxDataNorm(j,:) = MatRxData(j,:)/Rhatsigma(j);    
   %
   %---INPUT 
   %        MatRxData(dim,nsim) : nsim independent realizations used to estimate the pdf of random vector X(dim) 
   %        MatRxExp(dim,nexp)  : nexp experimental data of random vector X(dim)
   %
   %---OUTPUT
   %         Rpdf(nexp,1) : values of the pdf at points  Rx^1,...,Rx^nexp     
   
   Rhatsigma = std(MatRxData, 0, 2);
   s         = (4/((n+2)*nsim))^(1/(n+4));
   s2        = s*s;
   
   shss = 1/sqrt(s2+(nsim-1)/nsim);   % coef sh/s
   sh   = s*shss;                       % coed sh
   sh2  = sh*sh;
   
   cons          = 1/(prod(sqrt(2*pi)*sh*Rhatsigma));
   MatRxDataNorm = zeros(dim,nsim);                                           
   MatRxExpNorm  = zeros(dim,nexp);                                           
   for j =1:dim                                                              
       MatRxExpNorm(j,:)  = MatRxExp(j,:)/Rhatsigma(j);                       
       MatRxDataNorm(j,:) = MatRxData(j,:)/Rhatsigma(j);                      
   end
   
   Rpdf = zeros(nexp,1);                                                       % Rpdf(nexp,1)
                  %--- Scaler sequence
                  % for alpha = 1:nexp    
                  %     for ell=1:nsim                                   
                  %         norm2 = norm(shss*MatRxDataNorm(:,ell)-MatRxExpNorm(:,alpha) , 2); % MatRxDataNorm(dim,nsim), MatRxExpNorm(dim,nexp)
                  %         MatRS(ell,alpha) = exp(-0.5*norm2^2/sh2);                                       
                  %     end 
                  %     Rpdf(alpha)= cons*sum(MatRS(:,alpha))/nsim;            % Rpdf(nexp,1)
                  % end 
   %--- Vectorial sequence
   MatRbid = zeros(dim,nsim,nexp);                                             % MatRbid(dim,nsim,nexp),
   for alpha = 1:nexp                                                          % MatRxDataNorm(dim,nsim), MatRxExpNorm(dim,nexp)
       MatRbid(:,:,alpha) = shss*MatRxDataNorm(:,:)-repmat(MatRxExpNorm(:,alpha),1,nsim); 
   end
   MatRexpo = permute(MatRbid,[2 3 1]);                                        % MatRexpo(nsim,nexp,dim) = MatRbid(dim,nsim,nexp)
   MatRS    = exp(-(sum(MatRexpo.^2,3))/(2*sh2));                              % MatRS(nsim,nexp) 
   for alpha = 1:nexp 
       Rpdf(alpha)= cons*sum(MatRS(:,alpha))/nsim;                             % Rpdf(nexp,1)
   end 
   return  
end
