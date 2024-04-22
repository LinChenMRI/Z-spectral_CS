clear all;close all;clc;
addpath('toolbox\');
addpath('data\');
load('demo_data.mat') % dense_x dense_y sparse_x sparse_y Z_spectra_CS_y; Noted that the Z_spectra_CS_y were generated using Python.

figure;
screenSize = get(0, 'ScreenSize');
screenSize(4) = screenSize(4)/1.1;
set(gcf, 'Position', screenSize);
% ------------- linear interploation -------------%
subplot(2,4,1);
linear_interploation = interp1 (sparse_x,sparse_y,dense_x,'linear');
linear_interploation(101) = linear_interploation(100);
plot(sparse_x,sparse_y,'x',dense_x,dense_y,'b',dense_x,linear_interploation,'r','LineWidth', 1);
set(gca,  'Xdir', 'reverse','FontWeight', 'bold', 'FontSize', 15);
set(gca, 'LineWidth', 1);
ylim([0,1]);
xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]);
xtickangle(0);
leg = legend('Sampling Point', 'Ground Truth','Linear','Location','southwest','box','off');
set(leg,  'FontWeight', 'bold', 'FontSize', 8);

diff = squeeze(dense_y - linear_interploation);
disp(sqrt(sum((diff.*diff)/101)))
loss_linear = sqrt(sum((diff.*diff)/101));
disp("Linear")
disp("----------------")

subplot(2,4,5);
plot(dense_x,diff,'black','LineWidth', 1);
set(gca, 'LineWidth', 1);
ylim([-0.25,0.25]);
yline(0, '--')
xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]);
xtickangle(0);
title('Linear diff','FontWeight', 'bold', 'FontSize', 8)
set(gca, 'Xdir', 'reverse', 'FontWeight', 'bold', 'FontSize', 15);


% ------------- pchip interploation -------------%
pchip_interploation = interp1 (sparse_x,sparse_y,dense_x,'pchip');
subplot(2,4,2);
plot(sparse_x,sparse_y,'x',dense_x,dense_y,'b',dense_x,pchip_interploation,'r','LineWidth', 1);
set(gca,  'Xdir', 'reverse','FontWeight', 'bold', 'FontSize', 15);
set(gca, 'LineWidth', 1);
ylim([0,1]);
xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]);
xtickangle(0);
leg = legend('Sampling Point', 'Ground Truth','Pchip','Location','southwest','box','off');
set(leg,  'FontWeight', 'bold', 'FontSize', 8);

diff = squeeze(dense_y - pchip_interploation);
disp(sqrt(sum((diff.*diff)/101)))
loss_pchip = sqrt(sum((diff.*diff)/101));
disp("Pchip")
disp("----------------")

subplot(2,4,6);
plot(dense_x,diff,'black','LineWidth', 1);
set(gca, 'LineWidth', 1);
ylim([-0.25,0.25]);
yline(0, '--')
xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]);
xtickangle(0);
title('Pchip diff','FontWeight', 'bold', 'FontSize', 8)
set(gca, 'Xdir', 'reverse', 'FontWeight', 'bold', 'FontSize', 15);



% ------------- Lorentzian interploation -------------%
Lorentzian_interploation = model_fitting(sparse_x,sparse_y,dense_x)';
subplot(2,4,3);
plot(sparse_x,sparse_y,'x',dense_x,dense_y,'b',dense_x,Lorentzian_interploation,'r','LineWidth', 1);
set(gca,  'Xdir', 'reverse','FontWeight', 'bold', 'FontSize', 15);
set(gca, 'LineWidth', 1);
ylim([0,1]);
xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]);
xtickangle(0);
leg = legend('Sampling Point', 'Ground Truth','Lorentzian','Location','southwest','box','off');
set(leg,  'FontWeight', 'bold', 'FontSize', 8);

diff = squeeze(dense_y - Lorentzian_interploation);
disp(sqrt(sum((diff.*diff)/101)))
loss_Lorentzian = sqrt(sum((diff.*diff)/101));
disp("Lorentzian")
disp("----------------")


subplot(2,4,7);
plot(dense_x,diff,'black','LineWidth', 1);
set(gca, 'LineWidth', 1);
ylim([-0.25,0.25]);
yline(0, '--')
xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]);
xtickangle(0);
title('Lorentzian diff','FontWeight', 'bold', 'FontSize', 8)
set(gca, 'Xdir', 'reverse', 'FontWeight', 'bold', 'FontSize', 15);

% ------------- Z-spectral CS -------------%
subplot(2,4,4);
plot(sparse_x,sparse_y,'x',dense_x,dense_y,'b',dense_x,Z_spectra_CS_y,'r','LineWidth', 1); % Noted that the Z_spectra_CS_y were generated using Python.
set(gca,  'Xdir', 'reverse','FontWeight', 'bold', 'FontSize', 15);
set(gca, 'LineWidth', 1);
ylim([0,1]);
xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]);
xtickangle(0);
leg = legend('Sampling Point', 'Ground Truth','Z-spectral CS','Location','southwest','box','off');
set(leg,  'FontWeight', 'bold', 'FontSize', 8);

diff = squeeze(dense_y - Z_spectra_CS_y);
disp(sqrt(sum((diff.*diff)/101)))
loss_spectral = sqrt(sum((diff.*diff)/101));
disp("Z-spectral CS")
disp("----------------")

subplot(2,4,8);
plot(dense_x,diff,'black','LineWidth', 1);
set(gca, 'LineWidth', 1);
ylim([-0.25,0.25]);
yline(0, '--')
xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]);
xtickangle(0);
title('Z-spectral CS diff','FontWeight', 'bold', 'FontSize', 8)
set(gca, 'Xdir', 'reverse', 'FontWeight', 'bold', 'FontSize', 15);



