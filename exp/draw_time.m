function draw_time(dataset, type)
close all;
if (~exist('ymin','var'))
    ymin = 0;
end
if (~exist('ymax','var'))
    ymax = 0;
end

color_list = {'r','m','b','black'};
color_num = size(color_list,2);
shape_list = {'+','s','o','*','v','d'};
shape_num = size(shape_list,2);

cur_color_index = 1;
cur_shape_index = 1;

folder_name = strcat(dataset,'/');
mkdir figs
opt_list_file = strcat(folder_name,'opt_list.txt');
if type == 'TG'
    %opt_list = {'SSAROW.txt';'FOBOS.txt';'STG.txt';'Ada-FOBOS.txt'};
    opt_list = {'AROW-TG.txt';'STG.txt';'Ada-FOBOS.txt'};
elseif type =='DA'
    opt_list = {'AROW-DA.txt';'RDA.txt';'Ada-RDA.txt'};
elseif type == 'FS'
    opt_list = {'AROW-FS.txt';'SGD-FS.txt';'OFSGD.txt'};
elseif type == 'CMP'
    opt_list = {'AROW-FS.txt';'AROW-TG.txt';'AROW-DA.txt'};
end
%opt_list = textread(opt_list_file,'%s');

opt_num = size(opt_list,1);

legend_content = cell(1,opt_num);

for k = 1:1:opt_num
    result_file = opt_list{k,1};
    opt_name = my_split(result_file,'.');
    opt_name = opt_name{1,1};
    if strcmp(opt_name, 'SSAROW') 
        legend_content{1,k} = 'AROW-TG';
    elseif strcmp(opt_name,'CW-RDA')
        legend_content{1,k} = 'AROW-DA';
    elseif strcmp(opt_name,'ASAROW')
        legend_content{1,k} = 'AROW-FS';
    else
        legend_content{1,k} = opt_name;
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
	plot(sparse_vec, l_time_vec, color_shape,'LineWidth',2,'markersize',5);	 
    
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

ylabel('training time (s)', 'fontsize',14)
xlabel('sparsity (%)', 'fontsize',14)
legend(legend_content,'location','northwest')
print(strcat(folder_name,strcat(type,'-time-sparse')),'-dpdf')
%close all
