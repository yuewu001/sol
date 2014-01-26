#include <stdio.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <vector>
#include <random>
#include <math.h>

using namespace std;

void Usage(){
	cout<<"SyntheticData [options arg]\n";
	cout<<"Options:\n";
	cout<<"\t-vd valid dimension(100)\n";
	cout<<"\t-tr total rate: total dimenstion to valid dim rate(10)\n";
	cout<<"\t-nr noise rate: nosie number per sample to valid dim(2) \n";
	cout<<"\t-n data number: (10000)\n";
	cout<<"\t-o output file\n\n";
}

int twonorm(int argc, const char** args){
	int valid_dim = 100;
	int total_rate = 10;
	int noise_rate = 2;

	int data_num = 10000;
	string output_file = "synthetic.txt";

	for (int i = 0; i < argc; i++){
		if (strcmp(args[i], "-vd") == 0 && i + 1 < argc){
			valid_dim = atoi(args[i+1]);
		}
		else if (strcmp(args[i],"-tr") == 0 && i + 1 < argc){
			total_rate = atoi(args[i+1]);
		}
		else if (strcmp(args[i],"-nr") == 0 && i + 1 < argc){
			noise_rate = atoi(args[i+1]);
		}
		else if (strcmp(args[i],"-n") == 0 && i + 1 < argc){
			data_num = atoi(args[i+1]);
		}
		else if (strcmp(args[i],"-o") == 0 && i + 1 < argc){
			output_file = args[i+1];
		}
	}

	int total_dim = valid_dim * total_rate;
	int noise_level_min = valid_dim;
	int noise_level_max = valid_dim * noise_rate;


	float mean = 2 / sqrtf(20.f);
	float var = 1.f;

	float noise_mean = 0.f;
	float noise_std = 1.f;


	ofstream outFile(output_file.c_str(), ios::out);
	if (!outFile){
		cerr<<"open file "<<output_file<<" failed!"<<endl;
		return -1;
	}

	std::default_random_engine engine(time(NULL));

	//generate the mean vector, standard variance, and generator
	std::normal_distribution<float> class1(-mean, var);
	std::normal_distribution<float> class2(mean, var);
	std::uniform_int_distribution<int> class_selector(0,1);


	//noise generator geneator
	std::normal_distribution<float> noise_gen(noise_mean,noise_std);
	//nose dimension generator
	std::uniform_int_distribution<int> noise_level_gen(noise_level_min, noise_level_max);
	//noise dim selector
	std::uniform_int_distribution<int> noise_dim_selector(valid_dim,total_dim - 1);

	//invetrt labels
	std::uniform_int_distribution<int> inv_label_gen(0,9);
	vector<char> flags;
	flags.resize(total_dim);
	vector<float> data;
	std::normal_distribution<float> * sel_class = NULL;

	cout<<std::fixed,std::cout.precision(2);
	for (int k = 0; k < data_num; k++){
		if (k % 100 == 1){
			cout<<100.f * k / data_num <<"% ("<<k<<" samples) generated!\r";
		}

		data.clear();
		int label = -1;
		if (class_selector(engine) == 0){
			sel_class = &class1;
		}
		else{
			sel_class = &class2;
			label = 1;
		}

		//if (inv_label_gen(engine) == 0)
	//		label *= -1;

		outFile<<label;

		for (int i = 0; i < valid_dim; i++){
			outFile<<" "<<i + 1<<":"<<(*sel_class)(engine);
		}
		//clear the flags
		for (auto iter = flags.begin(); iter != flags.end(); iter++)
			*iter = 0;
		//for (auto &flag: flags) flag = 0;

		int noise_level = noise_level_gen(engine);
		for (int i = 0; i < noise_level; i++){
			flags[noise_dim_selector(engine)] = 1;
		}
		for (int i = valid_dim; i < total_dim; i++){
			if (flags[i] == 1){
				outFile<<" "<<i + 1<<":"<<noise_gen(engine);
			}
		}
		outFile<<"\n";
	}
	outFile.close();
	cout<<"100.0% ("<<data_num<<" samples) generated!\n";
	return 0;
}

int main(int argc, const char** args){
	Usage();
	return twonorm(argc, args);

	int valid_dim = 100;
	int total_rate = 10;
	int noise_rate = 2;

	int data_num = 1000000;
	string output_file = "synthetic.txt";

	for (int i = 0; i < argc; i++){
		if (strcmp(args[i], "-vd") == 0 && i + 1 < argc){
			valid_dim = atoi(args[i+1]);
		}
		else if (strcmp(args[i],"-tr") == 0 && i + 1 < argc){
			total_rate = atoi(args[i+1]);
		}
		else if (strcmp(args[i],"-nr") == 0 && i + 1 < argc){
			noise_rate = atoi(args[i+1]);
		}
		else if (strcmp(args[i],"-n") == 0 && i + 1 < argc){
			data_num = atoi(args[i+1]);
		}
		else if (strcmp(args[i],"-o") == 0 && i + 1 < argc){
			output_file = args[i+1];
		}
	}

	int total_dim = valid_dim * total_rate;
	int noise_level_min = valid_dim;
	int noise_level_max = valid_dim * noise_rate;


	float mean_min = -1.f;
	float mean_max = 1.f;
	float std_min = 0.5f;
	float std_max = 2.f;

	float noise_mean = 0.f;
	float noise_std = 2.f;


	ofstream outFile(output_file.c_str(), ios::out);
	if (!outFile){
		cerr<<"open file "<<output_file<<" failed!"<<endl;
		return -1;
	}

	std::default_random_engine engine(time(NULL));

	//generate the mean vector, standard variance, and generator
	std::uniform_real_distribution<float> mean_gen(mean_min,mean_max);
	std::uniform_real_distribution<float> stddev_gen(std_min,std_max);
	vector<std::normal_distribution<float> > norm_vec;
	//split vector
	vector<float> split_vec;
	float split_bias = 0;
	for (int i = 0; i < valid_dim; i++){
		float mean = mean_gen(engine);
		float stddev = stddev_gen(engine);
		std::normal_distribution<float> normal_gen(mean,stddev);
		float split_val = mean_gen(engine);
		split_bias -= split_val * mean;

		split_vec.push_back(split_val);
		norm_vec.push_back(normal_gen);
	}

	//noise generator geneator
	std::normal_distribution<float> noise_gen(noise_mean,noise_std);
	//nose dimension generator
	std::uniform_int_distribution<int> noise_level_gen(noise_level_min, noise_level_max);
	//noise dim selector
	std::uniform_int_distribution<int> noise_dim_selector(valid_dim,total_dim - 1);

	vector<char> flags;
	flags.resize(total_dim);
	vector<float> data;
	cout<<std::fixed,std::cout.precision(2);
	for (int k = 0; k < data_num; k++){
		if (k % 100 == 1){
			cout<<100.f * k / data_num <<"% ("<<k<<" samples) generated!\r";
		}

		data.clear();

		float predict = 0;
		for (int i = 0; i < valid_dim; i++){
			float feat = norm_vec[i](engine);
			data.push_back(feat);

			predict += feat * split_vec[i];
		}
		predict += split_bias;

		int label = predict > 0 ? 1 : -1;
		outFile<<label;
		for (int i = 0; i < valid_dim; i++)
			outFile<<" "<<i + 1<<":"<<data[i];

		//clear the flags
		for (auto iter = flags.begin(); iter != flags.end(); iter++)
			*iter = 0;
		//for (auto &flag: flags) flag = 0;

		int noise_level = noise_level_gen(engine);
		for (int i = 0; i < noise_level; i++){
			flags[noise_dim_selector(engine)] = 1;
		}
		for (int i = valid_dim; i < total_dim; i++){
			if (flags[i] == 1){
				outFile<<" "<<i + 1<<":"<<noise_gen(engine);
			}
		}
		outFile<<"\n";
	}
	outFile.close();
	cout<<"100.0% ("<<data_num<<" samples) generated!\n";
	return 0;
}
