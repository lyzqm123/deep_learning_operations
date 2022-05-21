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
	Tensor(const std::string &name) { this->name_ = name; }
	Tensor(const std::string &name, const std::vector<int> &dimension, const std::string &distribution_type) : Tensor(name, dimension)
	{
		if (distribution_type == "gaussian")
		{
			std::random_device device_random_;
			std::default_random_engine generator(device_random_());
			std::normal_distribution<T> distribution(0, 1);

			std::vector<std::vector<std::vector<std::vector<T>>>> tensor_4d;
			for (int i = 0; i < dimension_[0]; i++)
			{
				std::vector<std::vector<std::vector<T>>> tensor_3d;
				for (int j = 0; j < dimension_[1]; j++)
				{
					std::vector<std::vector<T>> tensor_2d;
					for (int k = 0; k < dimension_[2]; k++)
					{
						std::vector<T> tensor_1d;
						for (int t = 0; t < dimension_[3]; t++)
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
			tensor_vector temp_tensor(dimension_[0], std::vector<std::vector<std::vector<T>>>(dimension_[1], std::vector<std::vector<T>>(dimension_[2], std::vector<T>(dimension_[3], distribution_type == "ones" ? (T)1 : (T)0))));
			this->tensor_ = std::move(temp_tensor);
		}
	}

	Tensor(const std::string &name, const std::vector<int> &dimension, const std::vector<T> &init_tensor) : Tensor(name, dimension)
	{
		std::vector<std::vector<std::vector<std::vector<T>>>> tensor_4d;
		for (int i = 0; i < dimension_[0]; i++)
		{
			std::vector<std::vector<std::vector<T>>> tensor_3d;
			for (int j = 0; j < dimension_[1]; j++)
			{
				std::vector<std::vector<T>> tensor_2d;
				for (int k = 0; k < dimension_[2]; k++)
				{
					std::vector<T> tensor_1d;
					for (int t = 0; t < dimension_[3]; t++)
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

	Tensor(const std::string &name, const std::initializer_list<int> &dimension, const std::initializer_list<T> &init_tensor) : Tensor(name, dimension)
	{
		std::vector<std::vector<std::vector<std::vector<T>>>> tensor_4d;
		auto iter = init_tensor.begin();
		for (int i = 0; i < dimension_[0]; i++)
		{
			std::vector<std::vector<std::vector<T>>> tensor_3d;
			for (int j = 0; j < dimension_[1]; j++)
			{
				std::vector<std::vector<T>> tensor_2d;
				for (int k = 0; k < dimension_[2]; k++)
				{
					std::vector<T> tensor_1d;
					for (int t = 0; t < dimension_[3]; t++)
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
				for (int j = 0; j < dimension[1]; j++)
				{
					for (int k = 0; k < dimension[2]; k++)
					{
						for (int t = 0; t < dimension[3]; t++)
						{
							auto value = tensor_v[i][j][k][t];
							std::cout << value << " ";
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
