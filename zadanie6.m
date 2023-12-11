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
net_newff = newff([min(fun(:,1)) max(fun(:,1)); min(fun(:,2)) max(fun(:,2))], [liczba_n_h1 liczba_n_o]);

% newelm()
net_newelm = newelm([min(fun(:,1)) max(fun(:,1)); min(fun(:,2)) max(fun(:,2))], [liczba_n_h1, liczba_n_o]);

% newrb()
net_newrb = newrb(fun', y', 0.0, 1.0, liczba_n_h1);

% newrbe()
net_newrbe = newrbe(fun', y');

% Predykcje dla danych wejściowych
y_pred_newff = sim(net_newff, fun');
y_pred_newelm = sim(net_newelm, fun');
y_pred_newrb = sim(net_newrb, fun');
y_pred_newrbe = sim(net_newrbe, fun');

% Obliczanie błędów MSE
mse_newff = mse(y' - y_pred_newff);
mse_newelm = mse(y' - y_pred_newelm);
mse_newrb = mse(y' - y_pred_newrb);
mse_newrbe = mse(y' - y_pred_newrbe);

% Wypisywanie błędów MSE
disp(['MSE dla newff(): ' num2str(mse_newff)]);
disp(['MSE dla newelm(): ' num2str(mse_newelm)]);
disp(['MSE dla newrb(): ' num2str(mse_newrb)]);
disp(['MSE dla newrbe(): ' num2str(mse_newrbe)]);

% Plotowanie wyników
figure;

subplot(2, 2, 1);
plot(fun(:, 1), y(:, 1), 'b', fun(:, 1), y_pred_newff, 'r');
title('newff()');

subplot(2, 2, 2);
plot(fun(:, 1), y(:, 1), 'b', fun(:, 1), y_pred_newelm, 'g');
title('newelm()');

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
