x1=-10:0.1:10;
x2=-9.9:0.1:10.1;
x= [x1;x2];
fun=sin(x1);

range=[-20 20];
liczba_n_h1=30;
liczba_n_h2=20;
liczba_n_h3=10;
liczba_n_o=1;
liczba_epochs = 50;
fun_act = 'tansig';

siec=newelm([range; range],[liczba_n_h1 liczba_n_h2 liczba_n_h3 liczba_n_o],{fun_act,fun_act,fun_act,'purelin'},'trainlm');
siec.trainParam.epochs= liczba_epochs;
siec.trainParam.goal=0;

siec=train(siec,x,fun);
ynn=sim(siec,x);

% Plot the function, the network's approximation, and the error
figure(1)
plot(ynn,'r');
grid; hold on;
title('1, sieć jednokierunkowa, newff');
plot(fun,'b--');
legend('target', 'input')
mseValue1 = mse(fun,ynn);
title('1, siec jednokierunkowa, newff ',['MSE = ' num2str(mseValue1), ' epochs = ' num2str(liczba_epochs), ' funkcja aktywacyjna = ' num2str(fun_act)]);
