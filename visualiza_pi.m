clear;
mc_pi = load('mc_pi_cpu.txt');

clf;
ph = plot(mc_pi);
set(ph, 'color', 'b', 'LineStyle', '-', 'LineWidth', 2);
hold on;
lh = line([1 length(mc_pi)], [pi pi]);
set(lh, 'color', 'r', 'LineStyle', '--', 'LineWidth', 1);
%axis tight;
axis([1 length(mc_pi) pi*0.9 pi*1.1]);

fprintf('\nmc_pi = %.8f\n\n', mc_pi(end));
