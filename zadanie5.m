
clear all
close all
clc

% dane nowe
x0 = [-5 5 25];

zakres = [0 4];
[t,values] = ode23(@ukladlorenza,zakres,x0);

t1 = t + 0.01;

t_c = [t;t1];

fun = [t(1:length(t)-2),t(2:length(t)-1),t(3:length(t))];
y = [values(1:length(values) - 2,1),values(1:length(values) - 2,2),values(1:length(values) - 2,3)];


liczba_n_h1 = 20;
liczba_n_o = 3;

% sieci
%%% NEWFF
siec = newff([zakres;zakres;zakres],[liczba_n_h1 liczba_n_o],{'tansig','purelin'},'trainlm');
siec.trainParam.epochs=100;
siec.trainParam.goal=1e-5;

siec=train(siec, fun', y');
ynn=sim(siec,fun');

figure(1)
plot(values(:,1),values(:,2))

figure(2)
plot3(values(:,1),values(:,2),values(:,3))

figure(3)
plot(values(:,1)')
hold on
grid on
plot(ynn');
xlabel('x')
ylabel('y')
legend('Dane treningowe','Model sieciowy 1','Model sieciowy 2','Model sieciowy 3')
title("Zad 5 - NEWFF") % tytuł zmienić w zależności od rodzaju sieci

%%% NEWELM
x2seq=con2seq(y');
funseq=con2seq(fun');

nn_elm_model=newelm(y',fun',10);
nn_elm_model = train(nn_elm_model,x2seq,funseq);

ynn1=nn_elm_model(x2seq);
ynn1=cell2mat(ynn1);


figure(4)
plot(values(:,1)')
hold on
grid on
plot(ynn1');
xlabel('x')
ylabel('y')
legend('Dane treningowe','Model sieciowy 1','Model sieciowy 2','Model sieciowy 3')
title("Zad 5 - NEWELM") % tytuł zmienić w zależności od rodzaju sieci

%%% NEWRB
MN=10;
DF=3;
GOAL=0;
SPREAD=0.5;
NN_model_rbf=newrb(y',fun',GOAL,SPREAD,MN,DF);
ynn2=sim(NN_model_rbf,y');


figure(5)
plot(values(:,1)')
hold on
grid on
plot(ynn2');
xlabel('x')
ylabel('y')
legend('Dane treningowe','Model sieciowy 1','Model sieciowy 2','Model sieciowy 3')
title("Zad 5 - NEWRB") % tytuł zmienić w zależności od rodzaju sieci

%%% 
NN_model_rbfe=newrbe(y',fun',SPREAD);
ynn3=sim(NN_model_rbfe,y');


figure(6)
plot(values(:,1)')
hold on
grid on
plot(ynn3');
xlabel('x')
ylabel('y')
legend('Dane treningowe','Model sieciowy 1','Model sieciowy 2','Model sieciowy 3')
title("Zad 5 - NEWELM") % tytuł zmienić w zależności od rodzaju sieci

function y = ukladlorenza(t,f)
s=10;
beta = 8/3;
p = 28;
y = [s*f(2)-s*f(1);p*f(1) - f(1) - f(1)*f(3) - f(2);f(1)*f(2)-beta*f(3)];
end
