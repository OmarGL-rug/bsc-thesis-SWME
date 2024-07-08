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

data_AdaptiveZero = load("shockTube_Relaxation1.0_Mesh100+300_spatiallyAdaptiveZero.csv");
data_AdaptiveNeighbour = load("shockTube_Relaxation1.0_Mesh100+300_spatiallyAdaptiveNeighbour.csv");
data_Reference = load("shockTube_Relaxation1.0_Mesh100+300_Reference.csv");

x_AdaptiveZero = data_AdaptiveZero(:,1);
rho_AdaptiveZero = data_AdaptiveZero(:,2);
u_AdaptiveZero = data_AdaptiveZero(:,3);
T_AdaptiveZero = data_AdaptiveZero(:,4);

x_AdaptiveNeighbour = data_AdaptiveNeighbour(:,1);
rho_AdaptiveNeighbour = data_AdaptiveNeighbour(:,2);
u_AdaptiveNeighbour = data_AdaptiveNeighbour(:,3);
T_AdaptiveNeighbour = data_AdaptiveNeighbour(:,4);

x_Reference = data_Reference(:,1);
rho_Reference = data_Reference(:,2);
u_Reference = data_Reference(:,3);
T_Reference = data_Reference(:,4);

%% Plotting
% Define colors
 blue = [0, 0.4470, 0.7410];
 green = [0.4660, 0.6740, 0.1880];
 yellow = [0.9290, 0.6940, 0.1250];
 red = [0.8500, 0.3250, 0.0980];
 brown = [171, 104, 87]./255;

% plotting = 'all';
 plotting = 'rho';
% plotting = 'u';
% plotting = 'T';

if(strcmp(plotting,'rho'))
    plot1 = plot(x_AdaptiveZero,rho_AdaptiveZero,'--',x_AdaptiveNeighbour,rho_AdaptiveNeighbour,'-.', ...
        x_Reference,rho_Reference,'.')

    set(plot1(1),'Color',blue);
    set(plot1(1),'LineWidth',2);

    set(plot1(2),'Color',red);
    set(plot1(2),'LineWidth',2);

    set(plot1(3),'Color','black');
    set(plot1(3),'LineWidth',2);
     
    leg = legend('Zero','Neighbour', 'Reference','Location','Northeast');
    set(leg,'FontSize',12); 
    
    axis([-0.5,1,1.9,2.1]); % Change if necessary
    xlabel('x')
    ylabel('rho')
end

