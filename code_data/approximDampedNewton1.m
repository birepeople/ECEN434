%approximated damped Newton Method by reduce hessian calc
clear all;
% initialize 
alpha = 0.1;
beta = 0.6;
epsilon = 1e-8;


% generate the random instance
global A;
%load('A_200_100.mat');
load('A_500_400.mat');
[m,n] = size(A);
value = {};
step = {};
%main iteration

index = 1;
%do three time
for N = [1,10,40]
    % at step 0

    value{index} = [];
    step{index} = [];
    inner_step = 0;
    x = zeros(n,1);
    grad = A'*(1./(1 - A*x)) - 1./(1+x) + 1./(1-x);
    hessian = A'*diag((1./(1-A*x)).^2)*A + diag(1./(1+x).^2 + 1./(1-x).^2);
    hessian_i = hessian ^(-1);
while 0.5*lambda(hessian_i, grad) > epsilon
    
    
    value{index} = [value{index}, func(x)];
    delta_x = -hessian_i * grad;
    t = 1;
    
    % constrain the x in dom(x) by changing t
    while ((max(A*(x+t*delta_x)) >= 1) || (max(abs(x+t*delta_x)) >= 1))
        t = t * beta;
    end
    
    % backtracking line search 
    while (func(x+t*delta_x) - func(x) > alpha * t * grad' * delta_x)
        t = t * beta;
    end
    step{index} = [step{index}, t];
    % update x by:
    x = x + t * delta_x; 
    % update new gradient at x by:
    grad = A'*(1./(1 - A*x)) - 1./(1+x) + 1./(1-x);
    
    % update new hessian and inverse at x by every N step:
    if (mod(inner_step ,N) == 0)
        hessian = A'*diag((1./(1-A*x)).^2)*A + diag(1./(1+x).^2 + 1./(1-x).^2);
        hessian_i = hessian ^ (-1);
    end
    inner_step = inner_step + 1;
end 
index = index + 1;

end

%dump result
opt = min(value{3});


figure(1)
N = [1,10,40];
for i = 1:3
subplot(3,3,i);
plot([0:(length(value{i})-2)], value{i}(1:length(value{i})-1), '-');
yl = '$f(\textbf{x}^k)$';
xlabel('iterative step');
ylim([min(value{i}) max(value{i})]);
ylabel(yl,'Interpreter','latex');
title(['N=' num2str(N(i))]);
hold on;
end

for i = 1:3
subplot(3,3,i+3)
semilogy([0:(length(value{i})-2)], value{i}(1:length(value{i})-1)-opt, '-');
xlabel('iterative step');
yl2 = '$f(\textbf{x}^k)-p^*$';
ylabel(yl2,'Interpreter','latex');
title(['N=' num2str(N(i))]);
hold on;
end

for i = 1:3
subplot(3,3,i+6)
scatter([1:length(step{i})], step{i},'filled','black');
xlabel('iterative step'); 
ylabel('$t^k$','Interpreter','latex');
title(['N=' num2str(N(i))]);
hold on;
end

function res = func(x)
global A;
res = -sum(log(1-A*x)) - sum(log(1+x)) - sum(log(1-x));
end

function res = lambda(hess_i, grad)
res = grad' * hess_i * grad;
end
    

    
    
    
    
    
    
    
    

