function draw(dataset,xmin, xmax, ymin, ymax)
%dataset = 'synthetic_ofs'
close all;

color_list = {'r','m','b','black','g'};
color_num = size(color_list,2);
shape_list = {'s','+','*','v','o'};
shape_num = size(shape_list,2);

cur_color_index = 1;
cur_shape_index = 1;

folder_name = strcat(dataset,'/');
mkdir figs

opt_list_file = strcat(folder_name,'opt_list.txt');
opt_list = {'SOFS.txt';'PreSelOGD.txt';'PET.txt';'liblinear.txt';};

%opt_list = textread(opt_list_file,'%s');

opt_num = size(opt_list,1);

legend_content = cell(1,opt_num);

for k = 1:1:opt_num
    result_file = opt_list{k,1};
    opt_name = my_split(result_file,'.');
    opt_name = opt_name{1,1};
   
    %skip the first line
    result = dlmread(strcat(folder_name, result_file),'',1,0);
    if strcmp(opt_name,'PreSelOGD') == 1
        legend_content{1,k} = 'mRMR';
    else
        legend_content{1,k} = opt_name;
    end
   
    sparse_vec = result(:,5);
    l_time_vec = result(:,3);  
    l_time_vec(1) = l_time_vec(2);

	color_shape = strcat(color_list{1,cur_color_index}, '-');
    color_shape = strcat(color_shape, shape_list{1,cur_shape_index});
    figure(1)    
    grid on
    box on
	H(k) = semilogy(sparse_vec, l_time_vec, color_shape,'LineWidth',2,'markersize',10);	 
	hold on
    
    cur_color_index = cur_color_index + 1;
    if cur_color_index > color_num
        cur_color_index = 2;
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
xlabel('#Selected Features', 'fontsize',28)

if (exist('xmin','var'))
    axis([xmin xmax ymin ymax])
end
set(gca,'Fontsize',24);
gca_legend = legend(legend_content,'Location','NorthEast', 'fontsize',16);

%set(gca_legend,'Orientation','horizon')
%po=get( gca_legend, 'Position' ); 
%set( gca_legend, 'Position', [po(1) + 0.25, po(2), po(3), po(4)] );


print(strcat(folder_name,'time-sparse'),'-dpdf')
