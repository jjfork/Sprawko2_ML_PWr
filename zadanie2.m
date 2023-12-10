clear all
close all
clc

x1=0:0.1:10;
x2=0.1:0.1:10.1;
x= [x1;x2];
y1=x1;
y2=x2;


fun=(sin(2*x1)+0.2.*sin(8*x1)).*exp(-x1)*5;

%%%
range=[-20 20];
liczba_n_h1=30;
liczba_n_h2=10;
liczba_n_o=1;

siec=newff([range; range],[liczba_n_h1 liczba_n_h2 liczba_n_o],{'tansig','tansig','tansig','purelin'},'trainlm');
siec.trainParam.epochs=100;
siec.trainParam.goal=0;

siec=train(siec,x,fun);
ynn=sim(siec,x);

%%%
x2seq=con2seq(x2);
sseq=con2seq(fun);
nn_elm_model=newelm(x2,fun,10);
nn_elm_model = train(nn_elm_model,x2seq,sseq);

ynn1=nn_elm_model(x2seq);
ynn1=cell2mat(ynn1);
%%%

MN=10;
DF=3;
GOAL=0;
SPREAD=0.5;
NN_model_rbf=newrb(x2,fun,GOAL,SPREAD,MN,DF);
NN_model_rbfe=newrbe(x2,fun,SPREAD);
ynn2=sim(NN_model_rbf,x2);
ynn3=sim(NN_model_rbfe,x2);


figure(1)
plot(ynn,'r');
grid; hold on;
title('2, sieć jednokierunkowa, newff');
plot(fun,'b');
legend('matching', 'input')
mseValue1 = mse(fun,ynn);
title(['2, siec jednokierunkowa, newff, MSE = ', num2str(mseValue1)]);

figure(2)
plot(ynn1,'r');
grid;
hold on;
title('2, sieć rekurencyjna, newelm');
plot(fun,'b');
legend('matching', 'input')
mseValue2 = mse(fun,ynn1);
title(['2, siec rekurencyjna, newlm, MSE = ', num2str(mseValue2)]);


figure(3)
plot(ynn2, 'r-');
grid;
hold on;
title('1', 'newrb vs newrbe');
plot(ynn3, 'g-')
plot(fun, 'b--');
legend('rbf', 'rbfe', 'function');
e1=mse(fun, ynn2);
e2=mse(fun, ynn3);


text(1, 1, ['e1 = ', num2str(e1)]);
text(1, 0.9, ['e2 = ', num2str(e2)]);
