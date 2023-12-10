clear all % Clear all variables from the workspace
close all % Close all figures
clc % Clear the command window

x = linspace(0,10,201);
x1 = x(1:200);
x2 = x(2:201);

a = 1;

y1 = a * x1;
y2 = a * x2;
y_sin = (sin(2 * x1) + 0.2 * sin(8 * x1)) .* exp(-x1);

X = [y1; y2];

plot(y1, 'k'); grid on; hold on;
plot(y2, 'k'); hold on;
plot(x1,y_sin);

zakres = [-20 20];
liczba_n_h1 = 30;
liczba_n_o = 1;

siec = newff([zakres; zakres], [liczba_n_h1 liczba_n_o], {'tansig', 'purelin'}, 'trainlm');
siec.trainParam.epochs = 200;
siec.trainParam.goal = 1e-5;
%%%%%%%%%%
siec = train(siec, X, y_sin);
%%%%%%%%%%
ynn = sim(siec, X);

figure(2)
plot(y_sin, 'k');grid on; hold on;
plot(ynn, 'r')

