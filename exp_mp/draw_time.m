function draw_time(dataset, xmin, xmax, ymin, ymax)
close all;

color_list = {'r','m','b','black'};
color_num = size(color_list,2);
shape_list = {'+','s','o','*','v','d'};
shape_num = size(shape_list,2);

cur_color_index = 1;
cur_shape_index = 1;

folder_name = strcat(dataset,'/');
mkdir figs
opt_list = {'SGD_none.txt';'SGD_all.txt';'SGD_reservior.txt'};

opt_num = size(opt_list,1);

legend_content = cell(1,opt_num);
bf_size_list = [64,128,256,512,1024,2048,4096,9182];
for k = 1:1:opt_num
    result_file = opt_list{k,1};
    opt_name = my_split(result_file,'.');
    opt_name = opt_name{1,1};
   
    if strcmp(opt_name, 'SGD_none') == 1
        legend_content{1,k} = 'Perceptron';
    elseif strcmp(opt_name, 'SGD_all') == 1
        legend_content{1,k} = 'newest';
    elseif strcmp(opt_name, 'SGD_reservior') == 1 
        legend_content{1,k} = 'reservior';
    end

    result = load(strcat(folder_name, result_file));
   
    sparse_vec = result(:,3);
    l_time_vec = result(:,4);  
    l_time_vec(1) = l_time_vec(2);

	color_shape = strcat(color_list{1,cur_color_index}, '-');
    color_shape = strcat(color_shape, shape_list{1,cur_shape_index});
    figure(1)    
	hold on
    grid on
    box on
	plot(bf_size_list, l_time_vec, color_shape,'LineWidth',2,'markersize',5);	 
    
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
%title('training time vs sparsity', 'fontsize',14)

ylabel('training time (s)', 'fontsize',28)
xlabel('buffer size (%)', 'fontsize',28)

if (exist('xmin','var'))
    axis([xmin xmax ymin ymax])
end
legend(legend_content,'location','northwest', 'fontsize',22)
set(gca,'Fontsize',24);

print(strcat(folder_name,'-time-sparse'),'-dpng')
%close all
