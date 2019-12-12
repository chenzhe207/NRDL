function S = solve_test_coeff(D, Y)

tol = 1e-6; 
maxIter = 1e3;
rho = 1.1;
max_mu = 1e8;
mu = 1e-5;
[d, m] = size(Y);
[d, K] = size(D);
T1 = zeros(K, m);
S = zeros(K, m);
G = zeros(K, m);

%% starting iterations
iter = 0;
while iter < maxIter
    iter = iter + 1; 
    
    Sk = S;
    Gk = G;
    
    %update S
    S = pinv(D' * D + mu * eye(K)) * (D' * Y + mu * Gk - T1) ;
    
    %update G   
    G = S + mu \ T1;
    G = max(G, 0);
    
    
     %% convergence check   
    leq1 = S - G;

    stopC = max(max(abs(leq1)));

    if stopC < tol || iter >= maxIter
        break;
    else
        T1 = T1 + mu * leq1;
        mu = min(max_mu, mu * rho);
    end
    if (iter==1 || mod(iter, 5 )==0 || stopC<tol)
            disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopALM=' num2str(stopC,'%2.3e') ]);
    end

end

end
