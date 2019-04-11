data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
[X, mu, sigma] = featureNormalize(X);
X = [ones(m, 1) X];

% Choose some alpha value
alpha1 = 0.01;
alpha2 = 0.03;
alpha3 = 0.1;
num_iters = 400;

theta_NORMAL= normalEqn(X, y);

% Init Theta and Run Gradient Descent 
theta1 = zeros(3, 1);
[theta1, J_history1] = gradientDescent(X, y, theta1, alpha1, num_iters);
theta2 = zeros(3, 1);
[theta2, J_history2] = gradientDescent(X, y, theta2, alpha2, num_iters);
theta3 = zeros(3, 1);
[theta3, J_history3] = gradientDescent(X, y, theta3, alpha3, num_iters);

figure;
plot(1:numel(J_history1), J_history1, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
hold on;
plot(1:numel(J_history2), J_history2, '-r', 'LineWidth', 3);
plot(1:numel(J_history3), J_history3, '-g', 'LineWidth', 5);
legend('alpha1 = 0.01','alpha2 = 0.03','alpha3 = 0.1');
x_sample=[1,1650,3];
y_pre=x_sample*theta2;