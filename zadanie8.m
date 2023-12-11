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


% neural network variables
range = [-20 20];
liczba_n_h1 = 30;
liczba_n_o = 1;
% training neural network
siec = newff([range;range;range;range;range;range;range;range;range;range; ...
    range;range;range;range;range;range;range;range;range;range;range;range; ...
    range;range;range;], [liczba_n_h1 liczba_n_o], {'tansig', 'purelin'}, 'trainlm');
siec.trainParam.epochs = 50;
siec.trainParam.goal = 0;
siec = train(siec, in_data_train', out_data_train');
ynn = sim(siec,in_data_train');
% view(siec) % diagram
figure(2);
plot(out_data_train,'xb'); hold on; grid on
plot(ynn,'or')


%%%
siec = newelm(in_data_train', out_data_train', 10);
siec.trainParam.epochs = 50;
siec.trainParam.goal = 0;
siec = train(siec, in_data_train', out_data_train');
ynn1 = sim(siec, in_data_train');

% Plot results
figure(3);
plot(out_data_train', 'xb'); hold on; grid on
plot(ynn1', 'or')

%%%
siec_rb = newrb(in_data_train', out_data_train', 0, 1, liczba_n_h1);
siec = train(siec, in_data_train', out_data_train');
ynn2 = sim(siec, in_data_train');

% Plot results
figure(4);
plot(out_data_train', 'xb'); hold on; grid on
plot(ynn2', 'or')

%%%
siec_rbe = newrbe(in_data_train', out_data_train');
siec = train(siec, in_data_train', out_data_train');
ynn2 = sim(siec, in_data_train');

% Plot results
figure(5);
plot(out_data_train', 'xb'); hold on; grid on
plot(ynn2', 'or')


