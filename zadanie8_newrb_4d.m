clear all
close all
warning off
clc


X = [0 1 1 1 0;
    1 0 0 0 1;
    1 0 0 0 1;
    1 0 0 0 1;
    0 1 1 1 0];

x = X(:)';
in_data_train = x;
out_data_train = [0];


X = [0; 0; 0; 0; 1;
    0; 0; 0; 1; 1;
    0; 0; 1; 0; 1;
    0; 0; 0; 0; 1;
    0; 0; 0; 0; 1];

x = X(:)';
in_data_train = [in_data_train; x];
out_data_train = [out_data_train; 1];

X = [0 1 1 1 0;
     1 0 0 1 1;
     0 0 1 0 0;
     0 1 0 0 0;
     1 1 1 1 1];
%hintonw(X)
x = X(:)';
in_data_train = [in_data_train; x];
out_data_train = [out_data_train; 2];

X = [1 1 1 1 0;
     0 0 0 0 1;
     0 1 1 1 0;
     0 0 0 0 1;
     1 1 1 1 0];

x = X(:)';
in_data_train = [in_data_train; x];
out_data_train = [out_data_train; 3];

X = [0 0 1 1 0;
     0 1 0 1 0;
     1 1 1 1 1;
     0 0 0 1 0;
     0 0 0 1 0];

x = X(:)';
in_data_train = [in_data_train; x];
out_data_train = [out_data_train; 4];

X = [1 1 1 1 1;
     1 0 0 0 0;
     1 1 1 1 0;
     0 0 0 0 1;
     1 1 1 1 0];

x = X(:)';
in_data_train = [in_data_train; x];
out_data_train = [out_data_train; 5];

X = [0 1 1 1 0;
     1 0 0 0 0;
     1 1 1 1 0;
     1 0 0 0 1;
     0 1 1 1 0];


x = X(:)';
in_data_train = [in_data_train; x];
out_data_train = [out_data_train; 6];

X = [1 1 1 1 1;
     0 0 0 0 1;
     0 0 0 1 0;
     0 0 1 0 0;
     0 1 0 0 0];


x = X(:)';
in_data_train = [in_data_train; x];
out_data_train = [out_data_train; 7];

X = [0 1 1 1 0;
     1 0 0 0 1;
     0 1 1 1 0;
     1 0 0 0 1;
     0 1 1 1 0];


x = X(:)';
in_data_train = [in_data_train; x];
out_data_train = [out_data_train; 8];

X = [0 1 1 1 0;
     1 0 0 0 1;
     0 1 1 1 1;
     0 0 0 0 1;
     0 1 1 1 0];

x = X(:)';
in_data_train = [in_data_train; x];
out_data_train = [out_data_train; 9];

%LEGENDA
% zadanie8_newrb_1a.m
   % zadanie8 - numer zadania
   % newrb - wykorzystana funkcja
   % 1 - liczba epok
   % a - SPREAD
% neural network variables
GOAL = 0;
SPREAD = 1;
liczba_neuronow = 10; % max liczba
czynnik = 1; % liczba neuronów jaką dodajemy miedzy warstwami

%%%
siec_rb = newrb(in_data_train', out_data_train', GOAL, SPREAD, liczba_neuronow, czynnik);
ynn = sim(siec_rb, in_data_train');



% Plot results
figure(3);
plot(out_data_train', 'xb'); hold on; grid on
grid; hold on;
title('4d, sieć jednokierunkowa, newrb');
plot(ynn', 'or')
legend('target', 'input')
mseValue1 = mse(out_data_train', ynn);
title('4d siec jednokierunkowa, newrb',['MSE = ' num2str(mseValue1), ' #neuronow = ' num2str(liczba_neuronow), ' #czynniku = ' num2str(czynnik), ' rozrzut = ' num2str(SPREAD)]);


