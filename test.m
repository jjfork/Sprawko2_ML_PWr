% Clear all variables and close all figures
clear all
close all
warning off
clc

% Define the range of x values
x_values_1 = -10:0.1:10;
x_values_2 = -9.9:0.1:10.1;
x_values = [x_values_1; x_values_2];

% Define the function to be approximated
fun_values = sin(x_values_1);

% Define the parameters for the neural network
range = [-20 20];
hidden_layer_1_neurons = 15;
hidden_layer_2_neurons = 10;
output_neurons = 1;
hidden_layers = [hidden_layer_1_neurons hidden_layer_2_neurons output_neurons];
hidden_layers_count = length(hidden_layers) - 1;
epochs = 500;
activation_function = 'tansig';
learning_method = 'trainlm';

% Create and train the neural network
neural_network = newelm([range; range], hidden_layers, {activation_function, activation_function, 'purelin'}, learning_method);
neural_network.trainParam.epochs = epochs;
neural_network.trainParam.goal = 0;
neural_network = train(neural_network, x_values, fun_values);

% Simulate the neural network and calculate the mean squared error
simulated_values = sim(neural_network, x_values);
mse_value = mse(fun_values, simulated_values);

% Plot the results
figure(1)
plot(simulated_values, 'r');
grid; hold on;
title1 = '1_4d, sieć rekurencyjna - metoda uczenia, newelm';
plot(fun_values, 'b--');
legend('target', 'input')

% Display the title with the results
title_string = sprintf(['%s\nMSE = %d\nLiczba epok = %d\nFunkcja aktywacyjna = %s\nIlość neuronów w sieci = %d\nMetoda uczenia = %s\nLiczba warstw ukrytych = %d\n'], ...
   title1, mse_value, epochs, activation_function, hidden_layer_1_neurons + hidden_layer_2_neurons + output_neurons, learning_method, hidden_layers_count);

title(title_string);
