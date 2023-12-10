
clear all
close all
clc

n=randn(2,4);

fun=[0 1 1 0];

X=[0 0 1 1
   0 1 0 1];
X1=X + n * 0.1;


range=[-20 20];
liczba_n_h1=10;
liczba_n_h2=5;
liczba_n_o=1;

siec=newff([range; range],[liczba_n_h1 liczba_n_h2 liczba_n_o],{'tansig','tansig','tansig','purelin'},'trainlm');
siec.trainParam.epochs=100;
siec.trainParam.goal=0;

siec=train(siec,X,fun);
ynn=sim(siec,X1);

Xseq=con2seq(X);
funseq=con2seq(fun);

nn_elm_model=newelm(X,fun,10);
nn_elm_model = train(nn_elm_model,Xseq,funseq);

ynn1=nn_elm_model(Xseq);
ynn1=cell2mat(ynn1);

figure(1)
plot(ynn,'r*');
grid;
hold on;
title('4, sieć jednokierunkowa, newff');
plot(fun,'bo');
legend('matching', 'input')
mseValue1 = mse(fun,ynn);
title(['4 - XOR, sieć jednokierunkowa, newff, MSE = ', num2str(mseValue1)]);


figure(2)
plot(ynn1,'r*');
grid;
hold on;
title('4, sieć rekurencyjna, newelm');
plot(fun,'bo');
legend('matching', 'input')
mseValue2 = mse(fun,ynn1);
title(['4 - XOR, siec rekurencyjna, newlm, MSE = ', num2str(mseValue2)]);
