function draw(folder_name, ymin, ymax)
close all;

if (~exist('ymin','var'))
    ymin = 0;
end
if (~exist('ymax','var'))
    ymax = 50;
end

color_list = {'r','g','b','m'};
color_num = size(color_list,2);
shape_list = {'+','s','o','*','v','d'};
shape_num = size(shape_list,2);

cur_color_index = 1;
cur_shape_index = 1;

folder_name = strcat(folder_name,'/');
opt_list_file = strcat(folder_name,'opt_list.txt');
opt_list = textread(opt_list_file,'%s');
opt_num = size(opt_list,1);

legend_content = cell(1,opt_num);

for k = 1:1:opt_num
    result_file = opt_list{k,1};
    opt_name = strsplit(result_file,'.'){1,1};
    legend_content{1,k} = opt_name;

    result = load(strcat(folder_name, result_file));

    l_err_vec = result(:,1);
    t_err_vec = result(:,2);
    sparse_vec = result(:,3);
    l_time_vec = result(:,4);

    color_shape = strcat(color_list{1,cur_color_index}, '-');
    color_shape = strcat(color_shape, shape_list{1,cur_shape_index});
    figure(1)
    hold on
    plot(sparse_vec, l_err_vec, color_shape, 'LineWidth',2);
    figure(2)
    hold on
    plot(sparse_vec, t_err_vec, color_shape, 'LineWidth',2);

    cur_color_index = cur_color_index + 1;
    if cur_color_index > color_num
        cur_color_index = 1;
    end

    cur_shape_index = cur_shape_index + 1;
    if cur_shape_index > shape_num
        cur_shape_index = 1;
    end

end

figure(1) %learning error rate
title('learing error rate vs sparsity')
ylabel('learning error rate (%)')
xlabel('sparsity (%)')
axis([0 100 ymin ymax])
legend(legend_content,0)
print(strcat(folder_name,'learn_sparse.jpg'),'-djpg')
figure(2) %test error rate
title('test error rate vs sparsity')
ylabel('test error rate (%)')
xlabel('sparsity (%)')
axis([0 100 ymin ymax])
legend(legend_content,0)
print(strcat(folder_name,'test_sparse.jpg'),'-djpg')
