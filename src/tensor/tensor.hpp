#pragma once

#include <vector>
#include <string>
#include <iomanip>
#include <random>
#include <iostream>
#include <fstream>

template <typename T>
class Tensor
{
public:
	using tensor_vector = std::vector<std::vector<std::vector<std::vector<T>>>>;

	Tensor() = delete;
	Tensor(std::string name) { this->name_ = name; }
	Tensor(std::string name, std::vector<int> dimension, std::string distribution_type) : Tensor(name, dimension)
	{
		if (distribution_type == "gaussian")
		{
			std::random_device device_random_;
			std::default_random_engine generator(device_random_());
			std::normal_distribution<> distribution(0, 1);

			std::vector<std::vector<std::vector<std::vector<T>>>> tensor_4d;
			for (int i = 0; i < dimension_[0]; i++)
			{
				std::vector<std::vector<std::vector<T>>> tensor_3d;
				for (int t = 0; t < dimension_[3]; t++)
				{
					std::vector<std::vector<T>> tensor_2d;
					for (int j = 0; j < dimension_[1]; j++)
					{
						std::vector<T> tensor_1d;
						for (int k = 0; k < dimension_[2]; k++)
						{
							tensor_1d.push_back(distribution(generator));
						}
						tensor_2d.push_back(tensor_1d);
					}
					tensor_3d.push_back(tensor_2d);
				}
				tensor_4d.push_back(tensor_3d);
			}
			this->tensor_ = std::move(tensor_4d);
		}
		else if (distribution_type == "ones" || distribution_type == "zeros")
		{
			tensor_vector temp_tensor(dimension_[0], std::vector<std::vector<std::vector<T>>>(dimension_[3], std::vector<std::vector<T>>(dimension_[1], std::vector<T>(dimension_[2], distribution_type == "ones" ? (T)1 : (T)0))));
			this->tensor_ = std::move(temp_tensor);
		}
	}

	Tensor(std::string name, std::vector<int> dimension, const std::vector<T> &init_tensor) : Tensor(name, dimension)
	{
		std::vector<std::vector<std::vector<std::vector<T>>>> tensor_4d;
		for (int i = 0; i < dimension_[0]; i++)
		{
			std::vector<std::vector<std::vector<T>>> tensor_3d;
			for (int t = 0; t < dimension_[3]; t++)
			{
				std::vector<std::vector<T>> tensor_2d;
				for (int j = 0; j < dimension_[1]; j++)
				{
					std::vector<T> tensor_1d;
					for (int k = 0; k < dimension_[2]; k++)
					{
						auto value = init_tensor[i * dimension_[1] * dimension_[2] * dimension_[3] + j * dimension_[2] * dimension_[3] + k * dimension_[3] + t];
						tensor_1d.push_back(value);
					}
					tensor_2d.push_back(tensor_1d);
				}
				tensor_3d.push_back(tensor_2d);
			}
			tensor_4d.push_back(tensor_3d);
		}
		this->tensor_ = std::move(tensor_4d);
	}

	Tensor(std::string name, std::initializer_list<int> dimension, const std::initializer_list<T> &init_tensor) : Tensor(name, dimension)
	{
		std::vector<std::vector<std::vector<std::vector<T>>>> tensor_4d;
		auto iter = init_tensor.begin();
		for (int i = 0; i < dimension_[0]; i++)
		{
			std::vector<std::vector<std::vector<T>>> tensor_3d;
			for (int t = 0; t < dimension_[3]; t++)
			{
				std::vector<std::vector<T>> tensor_2d;
				for (int j = 0; j < dimension_[1]; j++)
				{
					std::vector<T> tensor_1d;
					for (int k = 0; k < dimension_[2]; k++)
					{
						auto value = *iter;
						tensor_1d.push_back(value);
						++iter;
					}
					tensor_2d.push_back(tensor_1d);
				}
				tensor_3d.push_back(tensor_2d);
			}
			tensor_4d.push_back(tensor_3d);
		}
		this->tensor_ = std::move(tensor_4d);
	}

	friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
	{
		const std::vector<int> dimension = tensor.get_dimension();
		auto tensor_v = tensor.get_tensor();
		const std::string name = tensor.get_name();

		std::cout << "\nName: [`" << name << "`]\n";
		std::cout << "Dimension: ";
		for (int i = 0; i < (int)dimension.size(); i++)
		{
			std::cout << (i == 0 ? "[" : "") << dimension[i] << (i == (int)dimension.size() - 1 ? "]\n" : ", ");
		}

		try
		{
			std::cout << "  [ ";
			for (int i = 0; i < dimension[0]; i++)
			{
				for (int t = 0; t < dimension[3]; t++)
				{
					for (int j = 0; j < dimension[1]; j++)
					{
						for (int k = 0; k < dimension[2]; k++)
						{
							auto value = tensor_v[i][t][j][k];
							std::cout << (float)value << " ";
						}
					}
				}
			}
			std::cout << "]\n\n";
		}
		catch (const std::exception &e)
		{
			std::cerr << e.what() << '\n';
		}

		return os;
	}

	std::vector<int> get_dimension() const
	{
		return this->dimension_;
	}
	tensor_vector get_tensor() const
	{
		return this->tensor_;
	}
	std::string get_name() const
	{
		return this->name_;
	}
	std::vector<T> get_serialized_tensor() const
	{	
		// NHWC
		std::vector<T> output;
		for (int i = 0; i < dimension_[0]; i++)
		{
			for (int t = 0; t < dimension_[3]; t++)
			{
				for (int j = 0; j < dimension_[1]; j++)
				{
					for (int k = 0; k < dimension_[2]; k++)
					{
						output.push_back(this->tensor_[i][t][j][k]);
					}
				}
			}
		}
		return std::move(output);
	}

	std::vector<std::vector<T>> get_im2col(int kernel_row, int kernel_col, int stride = 1, int padding = 0) const
	{
		int inputs_batch = this->dimension_[0];
		int inputs_row = this->dimension_[1];
		int inputs_col = this->dimension_[2];
		int inputs_channel = this->dimension_[3];

		int output_dimension = (inputs_row + 2 * padding - kernel_row) / stride + 1;
		int out_row = output_dimension * output_dimension;
		int out_col = kernel_row * kernel_col * inputs_channel;
		std::vector<std::vector<T>> output(out_row, std::vector<T>(out_col, 0));

		/* Referenced from `https://github.com/intel/caffe/blob/master/src/caffe/util/im2col.cpp#L94-L123` */
		for (int out_w = 0; out_w < out_col; ++out_w)
		{
			int w_offset = out_w % kernel_col;
			int h_offset = (out_w / kernel_col) % kernel_row;
			int c_im = out_w / kernel_row / kernel_col;

			const int hc0 = h_offset - padding;
			const int wc0 = w_offset - padding;
			for (int h = 0; h < output_dimension; ++h)
			{
				int h_pad = h * stride + hc0;

				const int row_offset = (out_w * output_dimension + h); // * output_dimension;
				const int srow_offset = (c_im * inputs_row + h_pad); // * inputs_col;
				for (int w = 0; w < output_dimension; ++w)
				{
					int w_pad = w * stride + wc0;
					if ((((unsigned)h_pad) < ((unsigned)inputs_row)) && (((unsigned)w_pad) < ((unsigned)inputs_col))){
						output[h * output_dimension + w][out_w] = this->tensor_[0][c_im][h_pad][w_pad];
					}
					else
					{
						output[h * output_dimension + w][out_w] = 0.0;
					}
				}
			}
		}

		return output;
	}

	int size() const
	{
		if (this->dimension_.empty())
			return 0;
		int dim = 1;
		for (auto &dimension : this->dimension_)
		{
			dim *= dimension;
		}
		return dim;
	}

private:
	Tensor(const std::string &name, const std::vector<int> &dimension) : Tensor(name)
	{
		if (dimension.empty() || dimension.size() == 3 || dimension.size() > 4)
		{
			throw std::runtime_error("[ERROR] Please check the dimension size (" + std::to_string(dimension.size()) + ")...");
		}
		if (dimension.size() <= 2)
		{
			this->dimension_ = {1, dimension[0], dimension[dimension.size() - 1], 1};
		}
		else
		{
			this->dimension_ = dimension;
		}
	}

protected:
	std::string name_ = "";
	tensor_vector tensor_;
	std::vector<int> dimension_;
};
