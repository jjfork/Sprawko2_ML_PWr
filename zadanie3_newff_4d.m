% Clear all variables and close all figures
clear all
close all
warning off
clc

% Define the range of x values
x_values_1 = -10:0.1:10;
x_values_2 = -9.9:0.1:10.1;
x_values = [x_values_1; x_values_2];

fun_values = exp(sin(x_values_1)+0.1*sin(5*x_values_1));
x_values = mapminmax(x_values);
fun_values = mapminmax(fun_values);

range=[-20 20];
liczba_n_h1=15;
liczba_n_h2=10;
liczba_n_h3=0;
liczba_n_o=1;
layers = [liczba_n_h1 liczba_n_h2 liczba_n_o];
liczba_warstw_ukrytych = length(layers) - 1;
liczba_epochs = 500;
fun_act = 'tansig';
learning_met = 'trainlm';


siec=newff([range; range],layers,{fun_act,fun_act, 'purelin'},learning_met);
siec.trainParam.epochs= liczba_epochs;
siec.trainParam.goal=0;

siec = train(siec, x_values, fun_values);

simulated_values = sim(siec, x_values);
mse_value = mse(fun_values, simulated_values);


figure(1)
plot(simulated_values, 'r');
grid; hold on;
title1 = '3_4d, sieć jednokierunkowa - newff - liczba epok';
plot(fun_values, 'b--');
legend('target', 'input');


titleString = sprintf(['%s\nMSE = %d\nLiczba epok ' ...
    '= %d\nFunkcja aktywacyjna = %s\nIlość neuronów w sieci = %d\nMetoda uczenia = %s\nLiczba warstw ukrytych = %d\n'], ...
    title1, mse_value, liczba_epochs, fun_act, liczba_n_h1 + liczba_n_h2 + liczba_n_h3, learning_met, liczba_warstw_ukrytych);

title(titleString);