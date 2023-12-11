clear all
close all
clc

% Dane
x = -10:0.1:10;
fun = (1 + cos(x)) .* (1 - sin(x));
y = (1 + cos(fun)) .* (1 - sin(fun));


zakres=[-20 20];
liczba_n_h1 = 10;
liczba_n_o=1;


% newff()
siec = newff(zakres,[liczba_n_h1 liczba_n_o],{'tansig','purelin'},'trainlm');
siec.trainParam.epochs=100;
siec.trainParam.goal=0;


siec=train(siec,fun,y);
ynn=sim(siec,fun);



% newelm()
net_newelm = newelm(zakres,[liczba_n_h1 liczba_n_o],{'tansig','purelin'},'trainlm');
net_newelm.inputs{1}.size = size(fun, 2);  % Dodaj tę linijkę
siec.trainParam.epochs=100;
siec.trainParam.goal=0;

siec=train(siec,fun,y);
ynn2=sim(siec,fun);

% newrb()
net_newrb = newrb(fun', y', 0.0, 1.0, liczba_n_h1);
siec.trainParam.epochs=100;
siec.trainParam.goal=0;

siec=train(siec,fun,y);
ynn3=sim(siec,fun);


% newrbe()
net_newrbe = newrbe(fun', y');
siec.trainParam.epochs=100;
siec.trainParam.goal=0;
siec=train(siec,fun,y);
ynn4=sim(siec,fun);


% Obliczanie błędów MSE
mse_newff = mse(y, ynn);
mse_newelm = mse(y, ynn2);
mse_newrb = mse(y, ynn3);
mse_newrbe = mse(y, ynn4);

% Wypisywanie błędów MSE
disp(['MSE dla newff(): ' num2str(mse_newff)]);
disp(['MSE dla newelm(): ' num2str(mse_newelm)]);
disp(['MSE dla newrb(): ' num2str(mse_newrb)]);
disp(['MSE dla newrbe(): ' num2str(mse_newrbe)]);

% Plotowanie wyników
figure(1);

subplot(2, 2, 1);
plot(ynn, 'r');
grid;
hold;
title('Zadanie 7, ynn');
plot(y, 'b');
legend('dopasowanie', 'rzeczywiste');

subplot(2, 2, 2);
plot(ynn2, 'g'); 
grid;
hold;
title('Zadanie 7, ynn2');
plot(y, 'b');

subplot(2, 2, 3);
plot(ynn3, 'm'); 
grid;
hold;
title('Zadanie 7, ynn3');
plot(y, 'b');

subplot(2, 2, 4);
plot(ynn4, 'c'); 
grid;
hold;
title('Zadanie 7, ynn4');
plot(y, 'b');

sgtitle('Porównanie wyników różnych sieci neuronowych')

