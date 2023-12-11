clear all
close all
warning off
clc
%https://eia.pg.edu.pl/documents/184139/30898785/MSI_LT1_Material_pomocniczy.pdf

x1=-10:0.1:10;
x2=-9.9:0.1:10.1;
x= [x1;x2];
fun=sin(x1);

range=[-20 20];
liczba_n_h1=15;
liczba_n_h2=10;
liczba_n_h3=0;
liczba_n_o=1;
layers = [liczba_n_h1 liczba_n_h2 liczba_n_o];
liczba_warstw_ukrytych = length(layers) - 1;
liczba_epochs = 100;
fun_act = 'tansig';
learning_met = 'trainlm';


siec=newelm([range; range],layers,{fun_act,fun_act, 'purelin'},learning_met);
siec.trainParam.epochs= liczba_epochs;
siec.trainParam.goal=0;

siec=train(siec,x,fun);
ynn=sim(siec,x);

figure(1)
plot(ynn,'r');
grid; hold on;
title1='1_2a, sieć rekurencyjna - funkcja aktywacyjna, newelm';
plot(fun,'b--');
legend('target', 'input')
mseValue1 = mse(fun,ynn);

titleString = sprintf(['%s\nMSE = %d\nLiczba epok ' ...
    '= %d\nFunkcja aktywacyjna = %s\nIlość neuronów w sieci = %d\nMetoda uczenia = %s\nLiczba wartsw ukrytych = %d\n'], ...
    title1, mseValue1, liczba_epochs, fun_act, liczba_n_h1 + liczba_n_h2 + liczba_n_h3, learning_met, liczba_warstw_ukrytych);


title(titleString);