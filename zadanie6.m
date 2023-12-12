% Dane
a = 1.4;
b = 0.3;
x0 = 1;
y0 = 1;
zakres = [0 50];

[t, values] = ode45(@(t, f) uklad(t, f, a, b), zakres, [x0, y0]);

fun = [t(1:length(t)-1), t(2:length(t))];
y = values(1:length(values)-1, :);

liczba_n_h1 = 100;
liczba_n_o = 2;

% newff()
siec=newff([zakres; zakres],[liczba_n_h1 liczba_n_o],{'tansig','purelin'},'trainlm');
siec.trainParam.epochs=100;
siec.trainParam.goal=0;

siec=train(siec,y',fun');
ynn=sim(siec,y');

figure(1)
plot(y(:, 1)','b');
grid; hold on;
title('1, sieć jednokierunkowa, newff');
plot(ynn,'b--');
legend('matching', 'input')
mseValue1 = mse(fun,ynn);
title(['1, siec jednokierunkowa, newff, MSE = ', num2str(mseValue1)]);

% newelm()
net_newelm=newelm([zakres; zakres],[liczba_n_h1 liczba_n_o],{'tansig','purelin'},'trainlm');
net_newelm.trainParam.epochs=100;
net_newelm.trainParam.goal=0; %%czemu to tak długo działa

siec=train(net_newelm,y',fun');
ynn2=sim(siec,y');

figure(2)
plot(y(:, 1),'b');
grid; hold on;
title('1, sieć jednokierunkowa, newff');
plot(y_pred_newrbe,'b--');
legend('matching', 'input')
mseValue1 = mse(y',y_pred_newrbe);
title(['1, siec jednokierunkowa, newff, MSE = ', num2str(mseValue1)]);


% newrb()
net_newrb = newrb(fun', y', 0.0, 1.0, liczba_n_h1);
y_pred_newrb = sim(net_newrb, fun');

figure(3)
plot(y(:, 1),'b');
grid; hold on;
title('1, sieć jednokierunkowa, newff');
plot(y_pred_newrb,'b--');
legend('matching', 'input')
mseValue1 = mse(y',y_pred_newrb);
title(['1, siec jednokierunkowa, newff, MSE = ', num2str(mseValue1)]);

% newrbe()
net_newrbe = newrbe(fun', y');
y_pred_newrbe = sim(net_newrbe, fun');

figure(4)
plot(y(:, 1),'b');
grid; hold on;
title('1, sieć jednokierunkowa, newff');
plot(y_pred_newrbe,'b--');
legend('matching', 'input')
mseValue1 = mse(y',y_pred_newrbe);
title(['1, siec jednokierunkowa, newff, MSE = ', num2str(mseValue1)]);


% Obliczanie błędów MSE
mse_newff = mse(y' - ynn);
%mse_newelm = mse(y' - ynn2);
mse_newrb = mse(y' - y_pred_newrb);
mse_newrbe = mse(y' - y_pred_newrbe);

% Wypisywanie błędów MSE
disp(['MSE dla newff(): ' num2str(mse_newff)]);
%disp(['MSE dla newelm(): ' num2str(mse_newelm)]);
disp(['MSE dla newrb(): ' num2str(mse_newrb)]);
disp(['MSE dla newrbe(): ' num2str(mse_newrbe)]);

% Plotowanie wyników
figure(5);

subplot(2, 2, 1);
plot(fun(:, 1), y(:, 1), 'b', fun(:, 1), ynn, 'r');
title('newff()');

%subplot(2, 2, 2);
%plot(fun(:, 1), y(:, 1), 'b', fun(:, 1), ynn2, 'g');
%title('newelm()');

subplot(2, 2, 3);
plot(fun(:, 1), y(:, 1), 'b', fun(:, 1), y_pred_newrb, 'm');
title('newrb()');

subplot(2, 2, 4);
plot(fun(:, 1), y(:, 1), 'b', fun(:, 1), y_pred_newrbe, 'c');
title('newrbe()');

sgtitle('Porównanie wyników różnych sieci neuronowych')


function df = uklad(t, f, a, b)
    x = f(1);
    y = f(2);

    df = [y + 1 - a * x^2; b * x];
end
