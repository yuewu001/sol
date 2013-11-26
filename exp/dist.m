close all;
clear all;
l1_list = [0.000000,0.000001,0.000005,0.000010,0.000025,0.000050,0.000075,0.000100,0.000125,0.000200,0.000250,0.000500,0.000750,0.0010000];
len = size(l1_list);
len = len(2);
for k = 1:1:len
    filename = sprintf('%.6f',l1_list(k));
    s = load(filename);
    B = -1:0.01:1;
    figure;
    H = HIST(s,B);
    %hist(s);
    plot(B,H);
    %plot(H);
    title(filename);
    print(strcat(filename,'.png'),'-dpng');
end