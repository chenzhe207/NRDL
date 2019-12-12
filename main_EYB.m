clear all
clc
close all

load YaleB_32x32.mat; 
fea = fea';
DATA = fea./ repmat(sqrt(sum(fea .* fea)), [size(fea, 1) 1]); %normalize
Label = gnd';


train_num = 20;
K = 760;
c = length(unique(Label));
numClass = zeros(c,1);
for i=1:c
    numClass(i,1) = length(find(Label==i));
end
%% select training and test samples
for time = 1 : 10
train_data = []; test_data = []; 
train_label = []; test_label = [];
for i = 1 : c
    index = find(Label == i); 
    Index = (6 : numClass(i));
    randIndex = Index(randperm(length(Index)));
    train_data = [train_data DATA(:,index([1 : 5, randIndex(1 : 15)]))];
    train_label = [train_label  Label(index([1 : 5, randIndex(1 : 15)]))];
  
    test_data = [test_data DATA(:, index(randIndex(16 : end)))];
    test_label = [test_label  Label(index(randIndex(16 : end)))];
end
    
for i = 1 : size(train_data, 2)
    a = train_label(i);
    Htr(a, i) = 1;
end %Htr
for i = 1 : (size(test_data, 2))
    a = test_label(i);
    Htt(a, i) = 1;
end % Htt

%% dictionary initialize
[Dinit] = initializationDictionary(train_data, Htr, K, 10, 30);

%% dictionary learning
lambda = 0.0001;
alpha = 1;
beta =  0.1;
gama = 0.0001;

[S, W, D, value_EYB] = dictionary_learning(train_data, Dinit, train_num, c, Htr, alpha, beta, gama, lambda);

%% starting classification
St = solve_test_coeff(D, test_data);

accuracy(time) = classification(W, Htt, St)
end

mean(accuracy)
imagesc(St)
colormap(gray(256))

