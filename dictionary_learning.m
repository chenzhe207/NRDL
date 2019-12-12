function [S, W, D] = dictionary_learning(X, Dinit, train_num, c, Htr, alpha, beta, gama, lambda)


tol = 1e-6; 
maxIter = 1e3;
rho = 1.1;
max_mu = 1e8;
mu = 1e-5;
[d, n] = size(X);
[d, K] = size(Dinit);
k = K / c;
%% initialize
S = zeros(K, n);
W = zeros(c, K);
P = zeros(c, n);
J = zeros(c, n);
T1 = zeros(c, n);
T2 = zeros(c, n);
D = Dinit;

%% construct Q
Q = [];
for i = 1 : c
    t = ones(k, train_num);
    Q = blkdiag(Q, t);
    clear t
end
A = ones(K, n) - Q;

%% construct L
dist = L2_distance(X, X);
dist = (dist + 1) .\ 1;
var = sum(dist, 2);
var = diag(var);
L = var - dist;

%% starting iterations
iter = 0;
while iter < maxIter
    iter = iter + 1; 
    
    Sk = S;
    Wk = W;
    Pk = P;    
    Jk = J;
    Dk = D;
    
    %update S
    R = Dk * (Q .* Sk);
    Stemp = pinv((2 * mu + alpha) * Wk' * Wk + (lambda + 1) * Dk' * Dk);
    S = Stemp * (Dk' * (X + lambda * R) + alpha * Wk' * Htr + mu * Wk' * Pk - W' * T1 + mu * W' * Jk - W' * T2);
    S = max(S, 0);
  
    
    %update W  
    W = (alpha * Htr + mu * Pk - T1 + mu * Jk - T2) * S' * pinv((2 * mu + alpha) * S * S');
    
    %update P class-wise  
    for i = 1 : c
      Si = S(:, (i - 1) * train_num + 1 : i * train_num);
      T1i = T1(:, (i - 1) * train_num + 1 : i * train_num);
      temp = zeros(c, c);
      for j = 1 : c         
          if j ~= i
              Pj = P(:, (j - 1) * train_num + 1 : j * train_num);
              temp = temp + Pj * Pj';
          end
      end
       P(:, (i - 1) * train_num + 1 : i * train_num) = pinv(beta * temp + mu * eye(c)) * (mu * W * Si + T1i);
       clear Si T1i 
    end

    %update J
    J = (mu * W * S + T2) * pinv(gama * L + mu * eye(n));
 
    
    %update D
    M = A .* S;
    D = X * S' * pinv(S * S' + lambda * M * M');
    
%    %% function value
%    term1 = norm(X - D * S, 'fro') ^ 2;
%    term2 = lambda * norm(D * (A .* S), 'fro') ^ 2;
%    term3 = alpha * norm(Htr - W * S, 'fro') ^ 2;
%    term4 = 0;
%    for i = 1 : c
%       Si = S(:, (i - 1) * train_num + 1 : i * train_num);  
%       WSi = W * Si;
%       for j = 1 : c         
%           if j ~= i
%               Sj = S(:, (j - 1) * train_num + 1 : j * train_num);
%               WSj = W * Sj;
%               term4 = term4 + norm(WSi' * WSj, 'fro') ^ 2;
%           end
%       end   
%    end
%    term4 = beta * term4;
%    term5 = 0;
%    for i = 1 : n
%       Si = S(:, i);  
%       WSi = W * Si;
%       for j = 1 : n         
%               Sj = S(:, j);
%               WSj = W * Sj;
%               term5 = term5 + norm(WSi - WSj) ^ 2 * dist(i,j);
%       end
%    end
%    term5 = gama * term5;
%    
%    value(iter) = term1 + term2 + term3 + term4 + term5;
%    
  
   %% convergence check   
    leq1 = W * S - P;
    leq2 = W * S - J;

    stopC = max(max(max(abs(leq1))), max(max(abs(leq2))));

    if stopC < tol || iter >= maxIter
        break;
    else
        T1 = T1 + mu * leq1;
        T2 = T2 + mu * leq2;
        mu = min(max_mu, mu * rho);
    end
    if (iter==1 || mod(iter, 5 )==0 || stopC<tol)
            disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopALM=' num2str(stopC,'%2.3e') ]);
    end

end

end

% Solving L_{21} norm minimization
function [E] = solve_l1l2(W,lambda)
n = size(W,1);
E = W;
for i = 1 : n
    E(i,:) = solve_l2(W(i,:), lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end