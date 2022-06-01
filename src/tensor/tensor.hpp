#pragma once
#include <cstring>
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
	Tensor() = delete;
	Tensor(std::string name) { this->name_ = name; }
	Tensor(std::string name, std::vector<int> dimension, std::string distribution_type) : Tensor(name, dimension)
	{
		if (distribution_type == "gaussian")
		{
			std::random_device device_random_;
			std::default_random_engine generator(device_random_());
			std::normal_distribution<> distribution(0, 1);
			T *origin = this->tensor_;
			for (int b = 0; b < dimension_[0]; b++)
			{
				for (int c = 0; c < dimension_[3]; c++)
				{
					for (int h = 0; h < dimension_[1]; h++)
					{
						for (int w = 0; w < dimension_[2]; w++)
						{
							*(this->tensor_++) = distribution(generator);
						}
					}
				}
			}
			this->tensor_ = origin;
		}
		else 
		{
			int dimension_size = dimension[0] * dimension[1] * dimension[2] * dimension[3];
			if (distribution_type == "ones")
			{
				std::fill(this->tensor_, this->tensor_ + dimension_size, (T)1);
			}
			else if (distribution_type == "zeros")
			{
				std::memset(this->tensor_, (T)0, sizeof (*this->tensor_) * dimension_size);
			}
		}
	}

	Tensor(std::string name, std::vector<int> dimension, const std::vector<T> &init_tensor) : Tensor(name, dimension)
	{
		T *origin = this->tensor_;
		for (int b = 0; b < dimension_[0]; b++)
		{
			std::vector<std::vector<std::vector<T>>> tensor_3d;
			for (int c = 0; c < dimension_[3]; c++)
			{
				std::vector<std::vector<T>> tensor_2d;
				for (int h = 0; h < dimension_[1]; h++)
				{
					std::vector<T> tensor_1d;
					for (int w = 0; w < dimension_[2]; w++)
					{
						*(this->tensor_++) = init_tensor[((b * dimension_[3] + c) * dimension_[1] + h) * dimension_[2] + w];
					}
				}
			}
		}
		this->tensor_ = origin;
	}

	Tensor(std::string name, std::initializer_list<int> dimension, const std::initializer_list<T> &init_tensor) : Tensor(name, dimension)
	{
		T *origin = this->tensor_;
		auto iter = init_tensor.begin();
		for (int b = 0; b < dimension_[0]; b++)
		{
			for (int c = 0; c < dimension_[3]; c++)
			{
				for (int h = 0; h < dimension_[1]; h++)
				{
					for (int w = 0; w < dimension_[2]; w++)
					{
						*(this->tensor_++) = *iter;
						++iter;
					}
				}
			}
		}
		this->tensor_ = origin;
	}
	~Tensor()
	{
		delete[] this->tensor_;
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
			for (int b = 0; b < dimension[0]; b++)
			{
				for (int c = 0; c < dimension[3]; c++)
				{
					for (int h = 0; h < dimension[1]; h++)
					{
						for (int w = 0; w < dimension[2]; w++)
						{
							auto value = tensor_v[((b * dimension[3] + c) * dimension[1] + h) * dimension[2] + w];
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
	T* get_tensor() const
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
		for (int b = 0; b < dimension_[0]; b++)
		{
			for (int c = 0; c < dimension_[3]; c++)
			{
				for (int h = 0; h < dimension_[1]; h++)
				{
					for (int w = 0; w < dimension_[2]; w++)
					{
						output.push_back(this->tensor_[((b * dimension_[3] + c) * dimension_[1] + h) * dimension_[2] + w]);
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

protected:
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

		this->tensor_ = new T[dimension_[0] * dimension_[1] * dimension_[2] * dimension_[3]];
	}

	Tensor(const Tensor &other)
	{
		const auto &other_dimension = other.dimension_;
		int other_dimension_size = other_dimension[0] * other_dimension[1] * other_dimension[2] * other_dimension[3];
		T *other_tensor = other.get_tensor();

		this->tensor_ = new T[other_dimension_size];
		this->dimension_ = other.dimension_;
		this->name_ = other.name_;
		std::copy(other_tensor, other_tensor + other_dimension_size, this->tensor_);
	}

protected:
	std::string name_ = "";
	T *tensor_ = nullptr;
	std::vector<int> dimension_;
};
