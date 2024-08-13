%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Spatially Adaptive Moment error computing %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear statements

clear all
close all
clc
format long

%% Simulation files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Specify the used simulation files %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

name_Adaptive = "linear+shockTube_Relaxation0.5_Mesh100+200_spatiallyAdaptiveNeighbour.csv";
name_Reference = "linear+shockTube_Relaxation0.5_Mesh100+200_Reference.csv";
name_LowOrder = "linear+shockTube_Relaxation0.5_Mesh100+200_LowOrder.csv";

data_Adaptive = load(name_Adaptive);
data_Reference = load(name_Reference);
data_LowOrder = load(name_LowOrder);

x_Adaptive = data_Adaptive(:,1);
rho_Adaptive = data_Adaptive(:,2);
u_Adaptive = data_Adaptive(:,3);
T_Adaptive = data_Adaptive(:,4);

x_Reference = data_Reference(:,1);
rho_Reference = data_Reference(:,2);
u_Reference = data_Reference(:,3);
T_Reference = data_Reference(:,4);

x_LowOrder = data_LowOrder(:,1);
rho_LowOrder = data_LowOrder(:,2);
u_LowOrder = data_LowOrder(:,3);
T_LowOrder = data_LowOrder(:,4);

%% Error computing

rho_Diff_LowHigh = norm(rho_LowOrder-rho_Reference);
rho_Diff_LowAdaptive = norm(rho_LowOrder-rho_Adaptive);
rho_Diff_AdaptiveHigh = norm(rho_Adaptive-rho_Reference);

u_Diff_LowHigh = norm(u_LowOrder-u_Reference);
u_Diff_LowAdaptive = norm(u_LowOrder-u_Adaptive);
u_Diff_AdaptiveHigh = norm(u_Adaptive-u_Reference);

T_Diff_LowHigh = norm(T_LowOrder-T_Reference);
T_Diff_LowAdaptive = norm(T_LowOrder-T_Adaptive);
T_Diff_AdaptiveHigh = norm(T_Adaptive-T_Reference);

%% Plotting

rho_diffs = [rho_Diff_LowHigh, rho_Diff_LowAdaptive,rho_Diff_AdaptiveHigh];
u_diffs = [u_Diff_LowHigh, u_Diff_LowAdaptive,u_Diff_AdaptiveHigh];
T_diffs = [T_Diff_LowHigh, T_Diff_LowAdaptive,T_Diff_AdaptiveHigh];

names = categorical({'L-H','L-A','A-H'});
names = reordercats(names,{'L-H','L-A','A-H'});

b=bar(names,rho_diffs);

ylabel('Difference in 2-norm')

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

%% Export figure
 %addpath('C:\Users\rikve\Gitlab\PhD-RUG\Spatially Adaptive Moment Models\RGD proceedings\Data\Export_fig\', '-end')
 %cd export_fig
 %export_fig('linearProfile+shock_mesh100+200_Time0.3_DifferenceBarGraph.pdf', '-pdf','-transparent');
