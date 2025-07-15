clear all
close all
clc
format long

% Load data
% name1 = 'damBreak-and-smooth_lambda1.0_nu0.1_order1_t5'; type1 = 'swme';
% name2 = 'damBreak-and-smooth_lambda1.0_nu0.1_order5_t5'; type2 = 'swme';
% name3 = 'damBreak-and-smooth_lambda1.0_nu0.1_adaptiveNonConservative_t5'; type3 = 'swmeAdaptive';
% name4 = 'damBreak-and-smooth_lambda1.0_nu0.1_adaptiveConservative_t5'; type4 = 'swmeAdaptive';

% name1 = 'damBreak-and-smooth_lambda0.1_nu0.1_order2_t5'; type1 = 'swme';
% name2 = 'damBreak-and-smooth_lambda0.1_nu0.1_order5_t5'; type2 = 'swme';
% name3 = 'damBreak-and-smooth_lambda0.1_nu0.1_adaptiveNonConservative_t5'; type3 = 'swmeAdaptive';
% name4 = 'damBreak-and-smooth_lambda0.1_nu0.1_adaptiveConservative_t5'; type4 = 'swmeAdaptive';

% name1 = 'damBreak-and-smooth_lambda0.1_nu1.0_order2_t5'; type1 = 'swme';
% name2 = 'damBreak-and-smooth_lambda0.1_nu1.0_order5_t5'; type2 = 'swme';
% name3 = 'damBreak-and-smooth_lambda0.1_nu1.0_adaptiveNonConservative_t5'; type3 = 'swmeAdaptive';
% name4 = 'damBreak-and-smooth_lambda0.1_nu1.0_adaptiveConservative_t5'; type4 = 'swmeAdaptive';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load data
% name1 = 'damBreak-and-smooth_linear_lambda1.0_nu0.1_order1_t5'; type1 = 'swme';
% name2 = 'damBreak-and-smooth_linear_lambda1.0_nu0.1_order5_t5'; type2 = 'swme';
% name3 = 'damBreak-and-smooth_linear_lambda1.0_nu0.1_adaptiveNonConservative_t5'; type3 = 'swmeAdaptive';
% name4 = 'damBreak-and-smooth_linear_lambda1.0_nu0.1_adaptiveConservative_t5'; type4 = 'swmeAdaptive';

name1 = 'damBreak-and-smooth_linear_lambda0.1_nu0.1_order2_t5'; type1 = 'swme';
name2 = 'damBreak-and-smooth_linear_lambda0.1_nu0.1_order5_t5'; type2 = 'swme';
name3 = 'damBreak-and-smooth_linear_lambda0.1_nu0.1_adaptiveNonConservative_t5'; type3 = 'swmeAdaptive';
% name3 = 'test'; type3 = 'swmeAdaptive';
name4 = 'damBreak-and-smooth_linear_lambda0.1_nu0.1_adaptiveConservative_t5'; type4 = 'swmeAdaptive';
 
% name1 = 'damBreak-and-smooth_linear_lambda0.1_nu1.0_order2_t5'; type1 = 'swme';
% name2 = 'damBreak-and-smooth_linear_lambda0.1_nu1.0_order5_t5'; type2 = 'swme';
% name3 = 'damBreak-and-smooth_linear_lambda0.1_nu1.0_adaptiveNonConservative_t5'; type3 = 'swmeAdaptive';
% name4 = 'damBreak-and-smooth_linear_lambda0.1_nu1.0_adaptiveConservative_t5'; type4 = 'swmeAdaptive';

[x1,h1,u1,alpha11,alpha21,alpha31,alpha41,alpha51,moments1] = readDataAdaptiveSWME1D(name1,type1);
[x2,h2,u2,alpha12,alpha22,alpha32,alpha42,alpha52,moments2] = readDataAdaptiveSWME1D(name2,type2);
[x3,h3,u3,alpha13,alpha23,alpha33,alpha43,alpha53,moments3] = readDataAdaptiveSWME1D(name3,type3);
[x4,h4,u4,alpha14,alpha24,alpha34,alpha44,alpha54,moments4] = readDataAdaptiveSWME1D(name4,type4);

% plotting = 'all';
% plotting = 'h';
% plotting = 'u';
% plotting = 'alpha1';
% plotting = 'alpha2';
% plotting = 'alpha3';
% plotting = 'alpha4';
% plotting = 'alpha5';
plotting = 'number_of_moments';


if(strcmp(plotting,'all'))
    ax1 = subplot(2,2,1);
    ax2 = subplot(2,2,2);
    ax3 = subplot(2,2,3);
    ax4 = subplot(2,2,4);
    ax5 = subplot(2,2,5);
    ax6 = subplot(2,2,6);
    ax7 = subplot(2,2,7);

    plot(ax1,x1,h1,x2,h2,x3,h3,x4,h4)
    legend(ax1,name1,name2,name3,name4)
    plot(ax2,x1,u1,x2,u2,x3,u3,x4,u4)

    plot(ax3,x1,alpha11,x2,alpha12,x3,alpha13,x4,alpha14)

    plot(ax4,x1,alpha21,x2,alpha22,x3,alpha23,x4,alpha24)

    plot(ax5,x1,alpha31,x2,alpha32,x3,alpha33,x4,alpha34)

    plot(ax6,x1,alpha41,x2,alpha42,x3,alpha43,x4,alpha44)

    plot(ax7,x1,alpha51,x2,alpha52,x3,alpha53,x4,alpha54)
end
if(strcmp(plotting,'h'))
    plot1 = plot(x1,h1,x2,h2,x3,h3,x4,h4)
    
    axis([-20,20,2.8,4.25]);
    % axis([-3.5,3.5,2.95,3.65]);
    xlabel('x')
    ylabel('h')
    
    linewidth1 = 2;
    set(plot1(1:4),'LineWidth',linewidth1);

    grey = [0.4,0.4,0.4];
    set(plot1(1),'LineStyle','-');
    set(plot1(2),'LineStyle','-.');
    set(plot1(3),'LineStyle','--');
    set(plot1(4),'LineStyle',':');
    set(plot1(1),'Color','r');
    set(plot1(2),'Color','blue');
    set(plot1(3),'Color','black');
    set(plot1(4),'Color','cyan');
    
    leg = legend('M=2','M=5','Adaptive: PBC','Adaptive: CIF','Location','northeast');
    set(leg,'FontSize',12); 
end
if(strcmp(plotting,'u'))
    plot1 = plot(x1,u1,x2,u2,x3,u3,x4,u4)

    % axis([-20.,20,-0.05,0.35]);
    axis([-3.5,3.5,0,0.2]);
    xlabel('x')
    ylabel('u_m')
    
    linewidth1 = 2;
    set(plot1(1:4),'LineWidth',linewidth1);

    grey = [0.4,0.4,0.4];
    set(plot1(1),'LineStyle','-');
    set(plot1(2),'LineStyle','-.');
    set(plot1(3),'LineStyle','--');
    set(plot1(4),'LineStyle',':');
    set(plot1(1),'Color','r');
    set(plot1(2),'Color','blue');
    set(plot1(3),'Color','black');
    set(plot1(4),'Color','cyan');
    
    leg = legend('M=2','M=5','Adaptive: PBC','Adaptive: CIF','Location','southwest');
    set(leg,'FontSize',12); 
end
if(strcmp(plotting,'alpha1'))
    plot1 = plot(x1,alpha11,x2,alpha12,x3,alpha13,x4,alpha14)
    
    axis([-20,20,-0.235,0]);
    % axis([-3.5,3.5,-0.1,0]);
    xlabel('x')
    ylabel('\alpha_1')
    
    linewidth1 = 2;
    set(plot1(1:4),'LineWidth',linewidth1);

    grey = [0.4,0.4,0.4];
    set(plot1(1),'LineStyle','-');
    set(plot1(2),'LineStyle','-.');
    set(plot1(3),'LineStyle','--');
    set(plot1(4),'LineStyle',':');
    set(plot1(1),'Color','r');
    set(plot1(2),'Color','blue');
    set(plot1(3),'Color','black');
    set(plot1(4),'Color','cyan');
    
    leg = legend('M=2','M=5','Adaptive: PBC','Adaptive: CIF','Location','southeast');
    set(leg,'FontSize',12); 
end
if(strcmp(plotting,'alpha2'))
    plot1 = plot(x1,alpha21,x2,alpha22,x3,alpha23,x4,alpha24)
    
    axis([-20.,20,-0.165,0.005]);
    % axis([1.,13.5,-0.04,0.01]);
    xlabel('x')
    ylabel('\alpha_2')
    
    linewidth1 = 2;
    set(plot1(1:4),'LineWidth',linewidth1);

    grey = [0.4,0.4,0.4];
    set(plot1(1),'LineStyle','-');
    set(plot1(2),'LineStyle','-.');
    set(plot1(3),'LineStyle','--');
    set(plot1(4),'LineStyle',':');
    set(plot1(1),'Color','r');
    set(plot1(2),'Color','blue');
    set(plot1(3),'Color','black');
    set(plot1(4),'Color','cyan');
    
    leg = legend('Low order','High order','Adaptive: PBC','Adaptive: CIF','Location','southeast');
    set(leg,'FontSize',12); 
end
if(strcmp(plotting,'alpha3'))
    plot1 = plot(x1,alpha31,x2,alpha32,x3,alpha33,x4,alpha34)

    axis([-20.,20,-0.075,0.075]);
    xlabel('x')
    ylabel('\alpha_3')
    
    linewidth1 = 2;
    set(plot1(1:4),'LineWidth',linewidth1);

    grey = [0.4,0.4,0.4];
    set(plot1(1:4),'LineStyle','-.');
    set(plot1(1),'Color','r');
    set(plot1(2),'Color','blue');
    set(plot1(3),'Color','black');
    set(plot1(4),'Color','green');
    
    leg = legend('Low order','High order','Adaptive: noncons','Adaptive: cons','Location','northwest');
    set(leg,'FontSize',12); 
end
if(strcmp(plotting,'alpha4'))
    plot1 = plot(x1,alpha41,x2,alpha42,x3,alpha43,x4,alpha44)
%     legend(name1,name2,name3,name4)

    axis([-0.4,0.4,-0.8,0.0]);
    xlabel('x')
    ylabel('\alpha_4')
    
    linewidth1 = 2;
    set(plot1(1:3),'LineWidth',linewidth1);
    set(plot1(2:4),'LineStyle','.');
    set(plot1(2:4),'MarkerSize',linewidth1+12);
    set(plot1(4),'LineStyle','.');
    set(plot1(4),'MarkerSize',linewidth1+12);

    grey = [0.4,0.4,0.4];
%     set(plot1(1),'Color','b');
    set(plot1(2),'Color','k');
%     set(plot1(3),'Color','b');
    set(plot1(3),'Color','k');
    set(plot1(4),'Color','k');
    
%     leg = legend('SWME','HSWME','\betaHSWME','reference','Location','southwest');
    leg = legend('SWME','reference','Location','southwest');
    set(leg,'FontSize',12); 
end
if(strcmp(plotting,'alpha5'))
    plot1 = plot(x1,alpha51,x2,alpha52,x3,alpha53,x4,alpha54)
%     legend(name1,name2,name3,name4)

    axis([-0.4,0.4,-0.8,0.0]);
    xlabel('x')
    ylabel('\alpha_5')
    
    linewidth1 = 2;
    set(plot1(1:3),'LineWidth',linewidth1);
    set(plot1(2:4),'LineStyle','.');
    set(plot1(2:4),'MarkerSize',linewidth1+12);
    set(plot1(4),'LineStyle','.');
    set(plot1(4),'MarkerSize',linewidth1+12);

    grey = [0.4,0.4,0.4];
%     set(plot1(1),'Color','b');
    set(plot1(2),'Color','k');
%     set(plot1(3),'Color','b');
    set(plot1(3),'Color','k');
    set(plot1(4),'Color','k');
    
%     leg = legend('SWME','HSWME','\betaHSWME','reference','Location','southwest');
    leg = legend('SWME','reference','Location','southwest');
    set(leg,'FontSize',12); 
end
if(strcmp(plotting,'number_of_moments'))
    yyaxis right
    % plot1 = plot(x1,moments1,x2,moments2,x3,moments3,x4,moments4)
    plot1 = plot(x3,moments3,x4,moments4)
    ylabel('Order')
    ylim([-0.5 5.5])

    set(plot1(1:2),'LineStyle','None');
    set(plot1(1),'marker','x');
    set(plot1(2),'marker','+');
    set(plot1(1),'Color','red');
    set(plot1(2),'Color','blue');
    set(plot1(1:2), 'MarkerSize',3)

    yyaxis left
    % plot(x1,u1,'LineWidth',2)
    plot(x1,h1,'LineWidth',2)
    
    % axis([-20,20,-0.5,5.5]);
    % axis([-3.5,0,3.4,3.9]);
    xlabel('x')
    % ylabel('u_m')
    ylabel('h')
    ylim([2.9 4.1])
    % ylim([-0.02 0.3])

    leg = legend('h','Adaptive order: PBC','Adaptive order: CIF','Location','southwest');
    set(leg,'FontSize',12); 
end


% addpath('C:\Users\rikve\Github\PhD-RUG\Spatially Adaptive Moment Models\Nonlinear-systems\Data-processing\Results\export_fig\', '-end')
% export_fig('damBreak-and-smooth_linear_lambda0.1_nu1.0_t5_adaptiveOrder.pdf', '-pdf','-transparent');