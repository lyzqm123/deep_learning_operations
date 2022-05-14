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
	Tensor() = delete;
	Tensor(const std::string &name) { this->name_ = name; }
	Tensor(const std::string &name, const std::vector<int> &dimension) : Tensor(name)
	{
		this->dimension_ = dimension;
	}

	Tensor(const std::string &name, const std::vector<int> &dimension, const std::string &distribution_type) : Tensor(name, dimension)
	{
		int tensor_size = this->size();
		if (distribution_type == "gaussian")
		{
			std::random_device device_random_;
			std::default_random_engine generator(device_random_());
			std::normal_distribution<T> distribution(0, 1);

			for (int i = 0; i < tensor_size; i++)
			{
				this->tensor_.push_back(distribution(generator));
			}
		}
		else if (distribution_type == "ones" || distribution_type == "zeros")
		{
			for (int i = 0; i < tensor_size; i++)
			{
				this->tensor_.push_back(distribution_type == "ones" ? (T)1 : (T)0);
			}
		}
	}
	Tensor(const std::string &name, const std::vector<int> &dimension, const std::vector<T> &init_tensor) : Tensor(name, dimension)
	{
		this->tensor_.resize(this->size());
		std::copy(init_tensor.begin(), init_tensor.end(), tensor_.begin());
	}
	Tensor(const std::string &name, const std::initializer_list<int> &dimension, const std::initializer_list<T> &init_tensor) : Tensor(name, dimension)
	{
		this->tensor_.resize(this->size());
		std::copy(init_tensor.begin(), init_tensor.end(), tensor_.begin());
	}

	friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
	{
		const std::vector<int> dimension = tensor.get_dimension();
		auto vector_tensor = tensor.get_tensor();
		const std::string name = tensor.get_name();

		std::cout << "\nName: [`" << name << "`]\n";
		std::cout << "Dimension: ";
		for (int i = 0; i < (int)dimension.size(); i++)
		{
			std::cout << (i == 0 ? "[" : "") << dimension[i] << (i == (int)dimension.size() - 1 ? "]\n" : ", ");
		}

		try
		{
			if (dimension.size() == 1)
			{
				for (int i = 0; i < dimension[0]; i++)
				{
					std::cout << (i == 0 ? "  [" : "") << vector_tensor[i] << (i == dimension[0] - 1 ? "]" : ", ");
				}
			}
			else if (dimension.size() == 2)
			{
				int k = 0;
				for (int i = 0; i < dimension[0]; i++)
				{
					for (int j = 0; j < dimension[1]; j++, k++)
					{
						std::cout << (j == 0 ? "  [" : "") << vector_tensor[k] << (j == dimension[1] - 1 ? "]" : ", ");
					}
					std::cout << "\n";
				}
			}
			std::cout << "\n";
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
	std::vector<T> get_tensor() const
	{
		return this->tensor_;
	}
	std::vector<std::vector<T>> get_2d_tensor() const
	{
		try
		{
			std::vector<std::vector<T>> ret(this->dimension_[0], std::vector<T>(this->dimension_[1]));
			int k = 0;
			for (int i = 0; i < this->dimension_[0]; i++)
			{
				for (int j = 0; j < this->dimension_[1]; j++)
				{
					ret[i][j] = this->tensor_[k++];
				}
			}
			return ret;
		}
		catch (const std::exception &e)
		{
			std::cerr << e.what() << '\n';
		}
		return std::vector<std::vector<T>>();
	}
	std::string get_name() const
	{
		return this->name_;
	}

	void push_back(const T &tensor)
	{
		this->tensor_.push_back(tensor);
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
	std::string name_ = "";
	std::vector<T> tensor_;
	std::vector<int> dimension_;
};
