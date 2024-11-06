#ifndef FILE_READER_H
#define FILE_READER_H

#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
namespace norm_k {
void dequantlinear_op(const std::vector<uint16_t>& input,
                      std::vector<float>& output, float scale,
                      uint16_t zero_point) {
  output.resize(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = (input[i] - zero_point) * scale;
  }
  return;
}

void quantizeLinear_op(const std::vector<float>& input,
                       std::vector<uint16_t>& output, float scale,
                       uint16_t zero_point) {
  output.resize(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    int32_t quantizedValue =
        static_cast<int32_t>(roundf(input[i] / scale)) + zero_point;

    if (quantizedValue < 0) {
      quantizedValue = 0;
    } else if (quantizedValue > UINT16_MAX) {
      quantizedValue = UINT16_MAX;
    }

    output[i] = static_cast<uint16_t>(quantizedValue);
  }
  return;
}

template <typename T>
void save_vec_span_2_bin(const gsl::span<const T>& span,
                         const std::string& filename) {
  std::ofstream outFile(filename, std::ios::binary);
  if (!outFile) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return;
  }

  for (const auto& data : span) {
    outFile.write(reinterpret_cast<const char*>(&data), sizeof(T));
  }

  outFile.close();
}

template <typename T>
void save_vec_span_2_bin(const std::vector<T>& vec,
                         const std::string& filename) {
  save_vec_span_2_bin(gsl::span<const T>(vec.data(), vec.size()), filename);
}

template <typename T>
void read_bin_file(const std::string& filename, std::vector<T>& buffer) {
  static_assert(std::is_trivially_copyable<T>::value,
                "T must be trivially copyable");

  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }

  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  buffer.resize(fileSize / sizeof(T));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
    throw std::runtime_error("Error reading file");
  }
}

template <typename T> class Tensor {
public:
  Tensor(const std::vector<T>& data, const std::vector<int>& shape)
      : data_(data), shape_(shape) {
    strides_ = ComputeStrides(shape_);
  }

  T Get(const std::vector<int>& index) const {
    int flat_index = 0;
    for (size_t i = 0; i < index.size(); ++i) {
      flat_index += index[i] * strides_[i];
    }
    return data_[flat_index];
  }

  void Set(const std::vector<int>& index, T value) {
    int flat_index = 0;
    for (size_t i = 0; i < index.size(); ++i) {
      flat_index += index[i] * strides_[i];
    }
    data_[flat_index] = value;
  }

  const std::vector<int>& Shape() const { return shape_; }
  std::vector<T> Data() const { return data_; }
  static std::vector<int> ComputeStrides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

private:
  std::vector<T> data_;
  std::vector<int> shape_;
  std::vector<int> strides_;
};

template <typename T>
Tensor<T> Gather(const Tensor<T>& data, const Tensor<int64_t>& indices,
                 int64_t axis = 0) {
  std::vector<int> result_shape = data.Shape();
  result_shape[axis] = indices.Shape()[axis];

  std::vector<T> result_data(
      indices.Shape()[0] * indices.Shape()[1] * data.Shape()[1 - axis], 0);
  std::vector<int> result_shape_1d = {
      indices.Shape()[0], indices.Shape()[1] * data.Shape()[1 - axis]};

  for (int i = 0; i < indices.Shape()[0]; ++i) {
    for (int j = 0; j < indices.Shape()[1]; ++j) {
      int index = indices.Get({i, j});
      if (axis == 0) {
        for (int k = 0; k < data.Shape()[1]; ++k) {
          result_data[i * result_shape_1d[1] + j * data.Shape()[1] + k] =
              data.Get({index, k});
        }
      } else {
        for (int k = 0; k < data.Shape()[0]; ++k) {
          result_data[i * result_shape_1d[1] + j * data.Shape()[0] + k] =
              data.Get({k, index});
        }
      }
    }
  }
  Tensor<T> result(result_data, result_shape_1d);

  return result;
}

std::vector<int> ComputeReducedShape(const std::vector<int>& shape, int axis) {
  std::vector<int> reduced_shape = shape;
  reduced_shape[axis] = 1;
  return reduced_shape;
}

template <typename T>
std::vector<T> ReduceMean(const Tensor<T>& data, int64_t axis) {
  int rank = data.Shape().size();
  if (axis < 0) {
    axis += rank;
  }
  std::vector<int> reduced_shape = ComputeReducedShape(data.Shape(), axis);

  int reduced_size = std::accumulate(reduced_shape.begin(), reduced_shape.end(),
                                     1, std::multiplies<int>());
  std::vector<T> reduced_data(reduced_size, 0);

  int num_elements = std::accumulate(data.Shape().begin(), data.Shape().end(),
                                     1, std::multiplies<int>());
  int num_axes_elements = data.Shape()[axis];

  std::vector<int> data_shape = data.Shape();
  std::vector<int> data_strides = Tensor<T>::ComputeStrides(data_shape);
  std::vector<int> reduced_strides = Tensor<T>::ComputeStrides(reduced_shape);

  std::vector<double> temp_reduced_data(reduced_size, 0);

  for (int i = 0; i < num_elements; ++i) {
    std::vector<int> data_index(data_shape.size());
    int temp = i;
    for (int j = 0; j < data_shape.size(); ++j) {
      data_index[j] = temp / data_strides[j];
      temp = temp % data_strides[j];
    }

    std::vector<int> reduced_index;
    for (int j = 0; j < data_shape.size(); ++j) {
      if (j != axis) {
        reduced_index.push_back(data_index[j]);
      }
    }

    int reduced_flat_index = 0;
    for (int j = 0; j < reduced_index.size(); ++j) {
      reduced_flat_index += reduced_index[j] * reduced_strides[j];
    }

    reduced_data[reduced_flat_index] += data.Get(data_index);
  }

  for (int i = 0; i < reduced_size; ++i) {
    reduced_data[i] = reduced_data[i] / num_axes_elements;
  }

  return reduced_data;
}

// broacast second input
template <typename T>
std::vector<T> Sub(const Tensor<T>& input1, const Tensor<T>& input2) {
  std::vector<int> shape1 = input1.Shape();
  std::vector<int> shape2 = input2.Shape();
  std::vector<T> data1 = input1.Data();
  std::vector<T> data2 = input2.Data();
  size_t dim1 = shape1[0];
  size_t dim2 = shape1[1];
  size_t dim3 = shape1[2];

  std::vector<int> result_shape = shape1;
  std::vector<T> result_data(data1.size());

  for (size_t i = 0; i < dim1; ++i) {
    for (size_t j = 0; j < dim2; ++j) {
      for (size_t k = 0; k < dim3; ++k) {
        result_data[i * dim2 * dim3 + j * dim3 + k] =
            data1[i * dim2 * dim3 + j * dim3 + k] - data2[i * dim2 + j];
      }
    }
  }

  return result_data;
}

// only support rank of second input is 1
template <typename T>
std::vector<T> Add(const std::vector<T>& input1, const std::vector<T>& input2) {
  std::vector<T> result_data(input1.size());
  for (size_t i = 0; i < input1.size(); ++i) {
    result_data[i] = input1[i] + input2[0];
  }
  return result_data;
}

template <typename T>
std::vector<T> Pow(const std::vector<T>& vector, float exponent) {

  std::vector<T> result;
  result.reserve(vector.size());

  for (const T& elem : vector) {
    result.push_back(static_cast<T>(std::pow(elem, exponent)));
  }

  return result;
}

template <typename T> std::vector<T> Sqrt(const std::vector<T>& input) {
  std::vector<T> result(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = std::sqrt(input[i]);
  }

  return result;
}

// broacast second input
template <typename T>
std::vector<T> Div(const Tensor<T>& input1, const Tensor<T>& input2) {
  std::vector<int> shape1 = input1.Shape();
  std::vector<int> shape2 = input2.Shape();
  std::vector<T> data1 = input1.Data();
  std::vector<T> data2 = input2.Data();
  size_t dim1 = shape1[0];
  size_t dim2 = shape1[1];
  size_t dim3 = shape1[2];

  std::vector<T> result_data(data1.size());

  for (size_t i = 0; i < dim1; ++i) {
    for (size_t j = 0; j < dim2; ++j) {
      for (size_t k = 0; k < dim3; ++k) {
        result_data[i * dim2 * dim3 + j * dim3 + k] =
            data1[i * dim2 * dim3 + j * dim3 + k] / data2[i * dim2 + j];
      }
    }
  }

  return result_data;
}

// perm is fixed to [0, 2, 1]
template <typename T> Tensor<T> Transpose(const Tensor<T>& input) {
  std::vector<int> input_shape = input.Shape();
  std::vector<int> output_shape = {input_shape[0], input_shape[2],
                                   input_shape[1]};

  std::vector<T> output_data(output_shape[0] * output_shape[1] *
                             output_shape[2]);
  Tensor<T> output(output_data, output_shape);
  for (int i = 0; i < input_shape[0]; ++i) {
    for (int j = 0; j < input_shape[1]; ++j) {
      for (int k = 0; k < input_shape[2]; ++k) {
        std::vector<int> input_index = {i, j, k};
        std::vector<int> output_index = {i, k, j};

        output.Set(output_index, input.Get(input_index));
      }
    }
  }
  Tensor<T> result(output.Data(), output_shape);
  return result;
}

} // namespace norm_k

#endif // FILE_READER_H
