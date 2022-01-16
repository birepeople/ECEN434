clear all;

alpha = 0.1;
beta = 0.6;
epsilon = 1e-8;


% generate the random instance
global A;
load('A_200_100.mat');
%load('A_500_400.mat');
[m ,n] = size(A);
value = [];
step = [];
%main iteration


% at step 0
x = zeros(n,1);
grad = A'*(1./(1 - A*x)) - 1./(1+x) + 1./(1-x);

hessian = A'*diag((1./(1-A*x)).^2)*A + diag(1./(1+x).^2 + 1./(1-x).^2);

while 0.5 * lambda(hessian, grad) > epsilon
   
    value = [value, func(x)];
    delta_x = -hessian^(-1) * grad;
    t = 1;
    
    % constrain the x in dom(x) by changing t
    while ((max(A*(x+t*delta_x)) >= 1) || (max(abs(x+t*delta_x)) >= 1))
        t = t * beta;
    end
    
    % backtracking line search 
    while (func(x+t*delta_x) - func(x) > alpha * t * grad' * delta_x)
        t = t * beta;
    end
    step = [step, t];
    % update x by:
    x = x + t * delta_x; 
    % update new gradient and hessian at x by:
    grad = A'*(1./(1 - A*x)) - 1./(1+x) + 1./(1-x);
    hessian = A'*diag((1./(1-A*x)).^2)*A + diag(1./(1+x).^2 + 1./(1-x).^2);

end 

%dump result
opt = min(value);

figure(1)
subplot(1,3,1);
plot([0:(length(value)-2)], value(1:length(value)-1), '-');
yl = '$f(\textbf{x}^k)$';
xlabel('iterative step');
ylim([min(value) max(value)]);
ylabel(yl,'Interpreter','latex');
title('value  - iterative step');
hold on;

subplot(1,3,2);
semilogy([0:(length(value)-2)], value(1:length(value)-1)-opt, '-');
xlabel('iterative step');
yl2 = '$f(\textbf{x}^k)-p^*$';
ylabel(yl2,'Interpreter','latex');
title('value between opt - iterative step');
hold on;

subplot(1,3,3);
scatter([1:length(step)], step,'filled','black');
xlabel('iterative step'); 
ylabel('$t^k$','Interpreter','latex');
title('step size - iterative step');
hold on;


% figure(1)
% semilogy([0:(length(value)-2)], value(1:length(value)-1)-opt, '-');
% xlabel('iterative step');
% yl2 = '$f(\textbf{x}^k)-p^*$';
% ylabel(yl2,'Interpreter','latex');
% c1=legend(['$\alpha = $' num2str(alpha) ',$\beta$ = ' num2str(beta)],'Interpreter','latex');
% title('value between opt - iterative step');
% hold on;


function res = func(x)
global A;
res = -sum(log(1-A*x)) - sum(log(1+x)) - sum(log(1-x));
end

function res = lambda(hess, grad)
res = grad' * hess^(-1) * grad;
end
    
  
    
    
    
    
    
    
    
    

