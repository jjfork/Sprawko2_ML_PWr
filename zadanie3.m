
clear all
close all
clc

x1=0:0.1:10;
x2=0.1:0.1:10.1;
x= [x1;x2];
y1=x1;
y2=x2;


fun=exp(sin(x1)+0.1*sin(5*x1));

[x, xn] = mapminmax(x);
[fun, fun_n] = mapminmax(fun);



range=[-20 20];
liczba_n_h1=30;
liczba_n_h2=10;
liczba_n_o=1;

siec=newff([range; range],[liczba_n_h1 liczba_n_h2 liczba_n_o],{'tansig','tansig','tansig','purelin'},'trainlm');
siec.trainParam.epochs=100;
siec.trainParam.goal=0;

siec=train(siec,x,fun);
ynn=sim(siec,x);

x2seq=con2seq(x2);
funseq=con2seq(fun);

nn_elm_model=newelm(x2,fun,10);
nn_elm_model = train(nn_elm_model,x2seq,funseq);

ynn1=nn_elm_model(x2seq);
ynn1=cell2mat(ynn1);

figure(1)
plot(ynn,'r');
grid; hold on;
title('3, sieć jednokierunkowa, newff');
plot(fun,'b');
legend('matching', 'input')
mseValue1 = mse(fun,ynn);
title(['3, siec jednokierunkowa, newff, MSE = ', num2str(mseValue1)]);


figure(2)
plot(ynn1,'r');
grid;
hold on;
title('3, sieć rekurencyjna, newelm');
plot(fun,'b');
legend('matching', 'input')
mseValue2 = mse(fun,ynn1);
title(['3, siec rekurencyjna, newlm, MSE = ', num2str(mseValue2)]);
