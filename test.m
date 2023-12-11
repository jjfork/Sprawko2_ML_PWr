x1=-10:0.1:10;
x2=-9.9:0.1:10.1;
x= [x1;x2];
fun=sin(x1);

range=[-20 20];
liczba_n_h1=30;
liczba_n_h2=20; % Add this line to define the number of neurons in the second hidden layer
liczba_n_h3=0;
liczba_n_o=1;
layers = [liczba_n_h1 liczba_n_h2 liczba_n_o]; % Modify this line to include the second hidden layer
liczba_epochs = 100;
fun_act = 'tansig';

siec=newelm([range; range],layers,{fun_act,fun_act, 'purelin'},'trainlm');
siec.trainParam.epochs= liczba_epochs;
siec.trainParam.goal=0;

siec=train(siec,x,fun);
ynn=sim(siec,x);

figure(1)
plot(ynn,'r');
grid; hold on;
title('1, sieć jednokierunkowa, newff');
plot(fun,'b--');
legend('target', 'input')
mseValue1 = mse(fun,ynn);
title('1, siec jednokierunkowa, newff',['MSE = ' num2str(mseValue1), ' epochs = ' num2str(liczba_epochs), ' funkcja aktywacyjna = ' num2str(fun_act)]);

text(10, 0.9, ['liczba_n_h1 = ', num2str(liczba_n_h1)]);
text(10, 0.8, ['liczba_n_h2 = ', num2str(liczba_n_h2)]);
text(10, 0.7, ['liczba_n_h3 = ', num2str(liczba_n_h3)]);
text(10, 0.6, ['Ilość neuronów = ', num2str(liczba_n_h1 + liczba_n_h2 + liczba_n_h3)], 'Color', 'k');
