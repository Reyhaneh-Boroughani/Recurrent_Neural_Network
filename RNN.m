clc
clear all
close all
%% Load data and rescale to [-1 1]
load('laser_dataset.mat')
data=laserTargets;
data_mat=cell2mat(data);
data_mat_normalized=rescale(data_mat,-1,1);
%% properly separate input and target data 
data_normalized=num2cell(data_mat_normalized);
input=data_normalized(1:end-1);
target=data_normalized(2:end);
%% Training, Validation, and Test Split
x_train=input(1:4000);
y_train=target(1:4000);

x_validation=input(4001:5000);
y_validation=target(4001:5000);

x_test=input(5001:end);
y_test=target(5001:end);
%% chose hyperparameters on validation set
inputDelays=1:2;
hiddenSizes=[10 20 30];
trainFcn='traingdx';  %'traingdx' is chosen over 'traingdm' and 'trainlm'
lr=[0.01, 0.001];
epoch=[500,1000];
err_thr=inf;
    for hs=1:length(hiddenSizes)
        for lri=1:length(lr)
                for epochi=1:length(epoch)
    
                    net = layrecnet(inputDelays,hiddenSizes(hs),trainFcn);
                    net.trainParam.lr=lr(lri);
                    net.trainParam.epochs=epoch(epochi);
                    [Xs,Xi,Ai,Ts] = preparets(net,x_train,y_train);
                    [net, tr] = train(net,Xs,Ts,'UseParallel','yes');

estimated_validation=net(x_validation);
err = immse(cell2mat(estimated_validation),cell2mat(y_validation));
    if err<err_thr
        chosen_hs=hs;
        chosen_lri=lri;
        chosen_epochi=epochi;
    end
    
err_thr=err;

                end
            end
    end
%% Parameters are chosen on Validation set and we Train on the selected model with whole training set
inputDelays=1:2;
hiddenSizes=hiddenSizes(chosen_hs); 
trainFcn='traingdx';
lr=lr(chosen_lri); %0.001
epoch=epoch(chosen_epochi);%%1000
tr_indices = 1:1:5000; %indices used for training
ts_indices = 5001:1:10092; %indices used for assessment

net.divideFcn = 'divideind';
net.divideParam.trainInd = tr_indices;
net.divideParam.testInd = ts_indices;
net = layrecnet(inputDelays,hiddenSizes,trainFcn);
net.trainParam.lr=lr;
net.trainParam.epochs=epoch;
[Xs,Xi,Ai,Ts] = preparets(net,input(1:5000),target(1:5000));
[net, tr] = train(net,Xs,Ts);
% save('RNN_trainingRecord.mat','tr');
% save('RNN_net.mat','net');
%% Test the RNN on test set
estimated_output_test=net(x_test);
RNN_tsMSE = immse(cell2mat(estimated_output_test),cell2mat(y_test));
% save('RNN_tsMSE.mat','RNN_tsMSE');
t_test=5001:10092;
figure;clf;
plot(t_test,cell2mat(estimated_output_test),'r--'); 
hold on
plot(t_test,cell2mat(y_test),'b--'); 
set(gca, 'xlim', [5001, 10092]);
legend('Estimated Output','Target')
title('Comparative Plot on Test Set')
% saveas(gcf,'test_target_output.png')
% saveas(gcf,'test_target_output')
%%  Test the RNN on training set
estimated_output_train=net(input(1:5000));
RNN_trMSE = immse(cell2mat(estimated_output_train),cell2mat(target(1:5000)));
% save('RNN_trMSE.mat','RNN_trMSE');
t_train=1:5000;
figure;clf;
plot(t_train,cell2mat(estimated_output_train),'r--'); 
hold on
plot(t_train,cell2mat(target(1:5000)),'b--'); 
legend('Estimated Output','Target')
title('Comparative Plot on Training Set')
% saveas(gcf,'train_target_output.png')
% saveas(gcf,'train_target_output')
%% Test the RNN on validation set
estimated_output_validation=net(x_validation);
RNN_vlMSE = immse(cell2mat(estimated_output_validation),cell2mat(y_validation));
% save('RNN_vlMSE.mat','RNN_vlMSE');