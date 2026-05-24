clc;
clear;
close all;
rng(0);  

%% Load Data
folder1 = 'folder_1';
folder2 = 'folder_2';

files1 = dir(fullfile(folder1, '*.wav'));
files2 = dir(fullfile(folder2, '*.wav'));

features = [];
labels = [];

%% Feature Extraction
for i = 1:length(files1)
    [x, fs] = audioread(fullfile(folder1, files1(i).name));
    
    if size(x,2) > 1
        x = mean(x,2);
    end
    
    x = x / max(abs(x) + eps);
    
    coeffs = mfcc(x, fs);
    feat = mean(coeffs);
    
    features = [features; feat];
    labels = [labels; 1];
end

for i = 1:length(files2)
    [x, fs] = audioread(fullfile(folder2, files2(i).name));
    
    if size(x,2) > 1
        x = mean(x,2);
    end
    
    x = x / max(abs(x) + eps);
    
    coeffs = mfcc(x, fs);
    feat = mean(coeffs);
    
    features = [features; feat];
    labels = [labels; 2];
end

%%  Train-Test Split

num_samples = size(features,1);
idx = randperm(num_samples);

train_size = round(0.7 * num_samples);

train_idx = idx(1:train_size);
test_idx  = idx(train_size+1:end);

X_train = features(train_idx,:);
Y_train_labels = labels(train_idx);

X_test = features(test_idx,:);
Y_test_labels = labels(test_idx);

%% Normalize

mu = mean(X_train);
sigma = std(X_train);

X_train = (X_train - mu) ./ sigma;
X_test  = (X_test  - mu) ./ sigma;

%% PCA
[coeff, score_train, ~, explained] = pca(X_train);

k = find(cumsum(explained) >= 95, 1);

X_train = score_train(:,1:k);
X_test  = X_test * coeff(:,1:k);

%% One-hot Encoding

num_classes = 2;

Y_train = zeros(length(Y_train_labels), num_classes);
for i = 1:length(Y_train_labels)
    Y_train(i, Y_train_labels(i)) = 1;
end

Y_test = zeros(length(Y_test_labels), num_classes);
for i = 1:length(Y_test_labels)
    Y_test(i, Y_test_labels(i)) = 1;
end

%% Initialize Network

[n_samples, n_features] = size(X_train);

hidden_neurons = 10;
output_neurons = num_classes;

W1 = randn(n_features, hidden_neurons) * 0.01;
b1 = zeros(1, hidden_neurons);

W2 = randn(hidden_neurons, output_neurons) * 0.01;
b2 = zeros(1, output_neurons);

%% Training Parameters

epochs = 500;
lr = 0.01;

sigmoid = @(x) 1 ./ (1 + exp(-x));
softmax = @(x) exp(x) ./ sum(exp(x),2);

%% Training Loop

for epoch = 1:epochs
    
    % Forward
    Z1 = X_train * W1 + b1;
    A1 = sigmoid(Z1);
    
    Z2 = A1 * W2 + b2;
    A2 = softmax(Z2);
    
    % Loss
    loss = -sum(sum(Y_train .* log(A2 + eps))) / n_samples;
    
    % Backprop
    dZ2 = A2 - Y_train;
    dW2 = (A1' * dZ2) / n_samples;
    db2 = sum(dZ2) / n_samples;
    
    dA1 = dZ2 * W2';
    dZ1 = dA1 .* (A1 .* (1 - A1));
    
    dW1 = (X_train' * dZ1) / n_samples;
    db1 = sum(dZ1) / n_samples;
    
    % Update
    W1 = W1 - lr * dW1;
    b1 = b1 - lr * db1;
    
    W2 = W2 - lr * dW2;
    b2 = b2 - lr * db2;
    
    if mod(epoch,50) == 0
        disp(['Epoch ', num2str(epoch), ' Loss: ', num2str(loss)]);
    end
end

%% Test on Unseen Data

%%  Detailed Prediction Output

correct = 0;

fprintf('\n--- Test Predictions ---\n');

for i = 1:size(X_test,1)
    
    % Forward pass
    Z1 = X_test(i,:) * W1 + b1;
    A1 = sigmoid(Z1);

    Z2 = A1 * W2 + b2;
    exp_scores = exp(Z2);
    A2 = exp_scores / sum(exp_scores);

    [~, pred] = max(A2);
    actual_label = Y_test_labels(i);

    % Check correctness
    if pred == actual_label
        result = 'Correct';
        correct = correct + 1;
    else
        result = 'Wrong';
    end

    % Display output
    fprintf('Sample %d → Predicted: %d | Actual: %d → %s\n', ...
            i, pred, actual_label, result);
end

% Final accuracy
test_accuracy = (correct / size(X_test,1)) * 100;

fprintf('\nFinal Test Accuracy: %.2f%%\n', test_accuracy);
