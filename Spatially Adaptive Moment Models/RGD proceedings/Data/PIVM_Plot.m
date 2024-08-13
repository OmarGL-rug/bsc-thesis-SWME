%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Spatially Adaptive Moment Simulation plotting %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% Plotting
% Define colors
 blue = [0, 0.4470, 0.7410];
 green = [0.4660, 0.6740, 0.1880];
 yellow = [0.9290, 0.6940, 0.1250];
 red = [0.8500, 0.3250, 0.0980];
 brown = [171, 104, 87]./255;

% plotting = 'all';
% plotting = 'rho';
% plotting = 'u';
 plotting = 'T';

if(strcmp(plotting,'rho'))
    plot1 = plot(x_Adaptive,rho_Adaptive,'--',x_Reference,rho_Reference,'-.', ...
        x_LowOrder,rho_LowOrder,'.')

    set(plot1(1),'Color',blue);
    set(plot1(1),'LineWidth',2);

    set(plot1(2),'Color',red);
    set(plot1(2),'LineWidth',2);

    set(plot1(3),'Color','black');
    set(plot1(3),'LineWidth',2);
     
    leg = legend('Adaptive','High Order', 'Low Order','Location','Northeast');
    set(leg,'FontSize',12); 
    
    axis([-2,1,0.9,3.1]); % Change if necessary
    xlabel('x')
    ylabel('\rho')
elseif(strcmp(plotting,'u'))
    plot1 = plot(x_Adaptive,u_Adaptive,'--',x_Reference,u_Reference,'-.', ...
        x_LowOrder,u_LowOrder,'.')

    set(plot1(1),'Color',blue);
    set(plot1(1),'LineWidth',2);

    set(plot1(2),'Color',red);
    set(plot1(2),'LineWidth',2);

    set(plot1(3),'Color','black');
    set(plot1(3),'LineWidth',2);
     
    leg = legend('Adaptive','High Order', 'Low Order','Location','Northwest');
    set(leg,'FontSize',12); 
    
    axis([-2,1,-0.1,0.8]); % Change if necessary
    xlabel('x')
    ylabel('u')
elseif(strcmp(plotting,'T'))
    plot1 = plot(x_Adaptive,T_Adaptive,'--',x_Reference,T_Reference,'-.', ...
        x_LowOrder,T_LowOrder,'.')

    set(plot1(1),'Color',blue);
    set(plot1(1),'LineWidth',2);

    set(plot1(2),'Color',red);
    set(plot1(2),'LineWidth',2);

    set(plot1(3),'Color','black');
    set(plot1(3),'LineWidth',2);
     
    leg = legend('Adaptive','High Order', 'Low Order','Location','Northwest');
    set(leg,'FontSize',12); 
    
    axis([-2,1,-0.5,0.5]); % Change if necessary
    xlabel('x')
    ylabel('\theta')
end

%% Export figure
 %addpath('C:\Users\rikve\Gitlab\PhD-RUG\Spatially Adaptive Moment Models\RGD proceedings\Data\Export_fig\', '-end')
 %cd export_fig
 %export_fig('linearProfile+shock_mesh100+200_T_Adaptive_Time0.3.pdf', '-pdf','-transparent');
