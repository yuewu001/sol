function draw(dataset,xmin, xmax, ymin, ymax)
%dataset = 'synthetic_ofs'
close all;

color_list = {'r','m','b','black'};
color_num = size(color_list,2);
shape_list = {'+','s','o','*','v','d'};
shape_num = size(shape_list,2);

cur_color_index = 1;
cur_shape_index = 1;

folder_name = strcat(dataset,'/');
mkdir figs

opt_list_file = strcat(folder_name,'opt_list.txt');
opt_list = {'AROW-FS.txt';'OFSGD.txt';'SGD-FS.txt'};

%opt_list = textread(opt_list_file,'%s');

opt_num = size(opt_list,1);

legend_content = cell(1,opt_num);

for k = 1:1:opt_num
    result_file = opt_list{k,1};
    opt_name = my_split(result_file,'.');
    opt_name = opt_name{1,1};
   
    legend_content{1,k} = opt_name;
  

    result = load(strcat(folder_name, result_file));

    l_err_vec = result(:,1);
    t_err_vec = result(:,2);
    sparse_vec = result(:,3);    

	color_shape = strcat(color_list{1,cur_color_index}, '-');
    color_shape = strcat(color_shape, shape_list{1,cur_shape_index});
    figure(1)
    grid on
    box on
    hold on
    plot(sparse_vec, l_err_vec, color_shape,'LineWidth',2,'markersize',5);
    figure(2)
    grid on
    box on
    hold on
    plot(sparse_vec, t_err_vec, color_shape,'LineWidth',2,'markersize',5);    
	
    cur_color_index = cur_color_index + 1;
    if cur_color_index > color_num
        cur_color_index = 1;
    end

    cur_shape_index = cur_shape_index + 1;
    if cur_shape_index > shape_num
        cur_shape_index = 1;
    end

end

folder_name = strcat('figs/',dataset);
folder_name = strcat(folder_name,'-');
figure(1) %learning error rate
%title('learing error rate vs sparsity', 'fontsize',14)
ylabel('learning error rate (%)', 'fontsize',28)
xlabel('sparsity (%)', 'fontsize',28)
if (exist('xmin','var'))
    axis([xmin xmax ymin ymax])
end
set(gca,'Fontsize',24);
legend(legend_content,'Location','NorthWest', 'fontsize',22)
print(strcat(folder_name,'-learn-sparse'),'-dpdf')
figure(2) %test error rate
%title('test error rate vs sparsity', 'fontsize',14)
ylabel('test error rate (%)', 'fontsize',28)
xlabel('sparsity (%)', 'fontsize',28)
if (exist('xmin','var'))
    axis([xmin xmax ymin ymax])
end
set(gca,'Fontsize',24);
legend(legend_content,'Location','NorthWest', 'fontsize',22)
print(strcat(folder_name,'-test-sparse'),'-dpng')
%close all
