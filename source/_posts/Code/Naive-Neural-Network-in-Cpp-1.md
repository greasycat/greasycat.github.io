---
title: Very Naive Neural Network in C++ Part 1 Matrix
tags:
  - c++
  - deep-learning
  - machine-learning
  - neural-network
categories:
  - Code
date: 2021-01-05 18:10:44
---
# Why even bother writing any neural networks in C++?
For production purposes, C++ implementations are assumed to be faster since it can utilize the  hardware to their limit if coded properly. But in most cases, python is powerful enough, especially when the python community is much larger.
But sometime I wonder if I rely on those beautiful toolkits a bit too much. Imagine someday the python community collapses (purely fictional). Will it be awesome if you know how to code things in those "ancient" languages?

# A 2D Matrix Class
The fundamental of any higher dimensional computation is matrix. There're actually a lot of implementation, but I still want to reinvent the wheel just for fun.

Let's create one 
1. to reduce the typing workload, I will overload most of the algebraic operators.
2. I need the dot operation and transpose
3. I also need a few getters and setters
4. I will put everything in a 2D `std::vector` 
<!-- more -->
> It might be to avoid std container if I know how to handle the power of dynamic memory.

```cpp
#include <iostream>
class Matrix {  
  
public:  
    explicit Matrix(std::vector<std::vector<double>>);  
    Matrix(int rows, int cols);  
    Matrix(Matrix const &matrix) noexcept;  
    ~Matrix();  
  
    // Operators overloading  
    double &operator()(int row, int col);  
    double operator()(int row, int col) const;  
  
    Matrix operator+(const Matrix &other) const;  
    Matrix operator+(double scalar) const;  
    Matrix operator-(const Matrix &other) const;  
    Matrix operator*(const Matrix &other) const;  
    Matrix operator*(double scalar) const;  
    Matrix operator/(double scalar) const;  
    Matrix &operator+=(const Matrix &other);  
    Matrix &operator-=(const Matrix &other);  
    Matrix &operator*=(double scalar);  
    Matrix &operator/=(double scalar);  
    Matrix &operator=(const Matrix &other);  
    Matrix &operator=(Matrix &&other) noexcept; 
  
    // Other computations  
    Matrix dot(const Matrix &other) const;  
    Matrix transpose() const;  
    //Fetching Operations  
    Matrix getRow(int row) const;  
    Matrix getCol(int col) const; 
    // Getter  
    int size() const;  
    int rows() const;  
    int cols() const;  
    double sum() const;  
    double mean() const; 
    std::pair<int, int> shape() const;  
    bool isScalar() const;  
  
    // Setter  
    Matrix changeAt(int row, int col, double value);  
    // Manipulation  
    void addRow(const Matrix &row);  
    void setRow(int row, const Matrix &newRow);  
  
    // Static functions  
    static Matrix EmptyMatrix() { return {0, 0}; };  
  
    // Print  
    friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix);  
    void print() const;  
    std::string printableShape() const;  
  
private:  
    std::vector<std::vector<double>> data;  
    void checkIfSameDimension(const Matrix &other) const;  
    void checkForDotMultiplication(const Matrix &other) const;  
  
};  
#endif //NAIVE_NN_MATRIX_H
```


# Constructors
```cpp
#include "Matrix.h"  
Matrix::Matrix(int rows, int cols){  
    // initialize the vector  
    this->data = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));  
  
}  
  
Matrix::Matrix(std::vector<std::vector<double>> data) {  
    // Copy data from the array to the matrix  
    this->data = std::move(data);  
}  


Matrix::Matrix(Matrix const &matrix) noexcept {  
    this->data = matrix.data;  
}  
Matrix::~Matrix() = default;  
```

# Operators overloading
- For the `*` multiplication, it is the similar to the broadcasting logic in numpy

```cpp  
double &Matrix::operator()(int row, int col) {  
    return data[row][col];  
}  
  
double Matrix::operator()(int row, int col) const {  
    return data[row][col];  
}  
  
Matrix Matrix::operator+(const Matrix &other) const {  
    checkIfSameDimension(other);  
    // Add two matrices  
    Matrix result = Matrix(other.rows(), other.cols());  
    for (int i = 0; i < other.rows(); i++) {  
        for (int j = 0; j < other.cols(); j++) {  
            result(i, j) = this->data[i][j] + other(i, j);  
        }  
    }  
    return result;  
}  

Matrix Matrix::operator+(double scalar) const {  
    Matrix result = Matrix(this->rows(), this->cols());  
    for (int i = 0; i < this->rows(); i++) {  
        for (int j = 0; j < this->cols(); j++) {  
            result(i, j) = this->data[i][j] + scalar;  
        }  
    }  
    return result;  
} 
  
Matrix Matrix::operator-(const Matrix &other) const {  
    checkIfSameDimension(other);  
  
    Matrix result = Matrix(other.rows(), other.cols());  
    for (int i = 0; i < other.rows(); i++) {  
        for (int j = 0; j < other.cols(); j++) {  
            result(i, j) = this->data[i][j] - other(i, j);  
        }  
    }  
    return result;  
  
}  
  
Matrix Matrix::operator*(const Matrix &other) const {  
    //Broadcasting  
    // If one of the matrix is a scalar, then we can just multiply the scalar with the other matrix    if (this->isScalar()) {  
        return other * this->data[0][0];  
    } else if (other.isScalar()) {  
        return *this * other.data[0][0];  
    }  
  
    // If the two matrices has the same dimension, then we can just multiply them  
    if (this->rows() == other.rows() && this->cols() == other.cols()) {  
        Matrix result = Matrix(this->rows(), this->cols());  
        for (int i = 0; i < this->rows(); i++) {  
            for (int j = 0; j < this->cols(); j++) {  
                result(i, j) = this->data[i][j] * other(i, j);  
            }  
        }  
        return result;  
    }  
  
    // If the two matrices has the same number of rows and one of them has only one column, then we can just multiply them element-wise  
    if (this->rows() == other.rows() && other.cols() == 1) {  
        Matrix result = Matrix(this->rows(), this->cols());  
        for (int i = 0; i < this->rows(); i++) {  
            for (int j = 0; j < this->cols(); j++) {  
                result(i, j) = this->data[i][j] * other(i, 0);  
            }  
        }  
        return result;  
    }  
  
    if (this->rows() == other.rows() && this->cols() == 1) {  
        Matrix result = Matrix(this->rows(), other.cols());  
        for (int i = 0; i < this->rows(); i++) {  
            for (int j = 0; j < other.cols(); j++) {  
                result(i, j) = this->data[i][0] * other(i, j);  
            }  
        }  
        return result;  
    }  
  
    // If the two matrices has the same number of columns and one of them has only one row, then we can just multiply them element-wise  
    if (this->cols() == other.cols() && other.rows() == 1) {  
        Matrix result = Matrix(this->rows(), this->cols());  
        for (int i = 0; i < this->rows(); i++) {  
            for (int j = 0; j < this->cols(); j++) {  
                result(i, j) = this->data[i][j] * other(0, j);  
            }  
        }  
        return result;  
    }  
  
    if (this->cols() == other.cols() && this->rows() == 1) {  
        Matrix result = Matrix(other.rows(), this->cols());  
        for (int i = 0; i < other.rows(); i++) {  
            for (int j = 0; j < this->cols(); j++) {  
                result(i, j) = this->data[0][j] * other(i, j);  
            }  
        }  
        return result;  
    }  
  
    std::cerr << "Shape mismatch: " << this->printableShape() << " and " << other.printableShape() << std::endl;  
    throw std::invalid_argument("The two matrices cannot be multiplied");  
}  
  
Matrix Matrix::operator*(double scalar) const {  
    Matrix result = Matrix(this->rows(), this->cols());  
    for (int i = 0; i < this->rows(); i++) {  
        for (int j = 0; j < this->cols(); j++) {  
            result(i, j) = this->data[i][j] * scalar;  
        }  
    }  
    return result;  
}  
  
Matrix Matrix::operator/(double scalar) const {  
    Matrix result = Matrix(this->rows(), this->cols());  
    for (int i = 0; i < this->rows(); i++) {  
        for (int j = 0; j < this->cols(); j++) {  
            result(i, j) = this->data[i][j] / scalar;  
        }  
    }  
    return result;  
}  
  
Matrix &Matrix::operator+=(const Matrix &other) {  
    checkIfSameDimension(other);  
    for (int i = 0; i < other.rows(); i++) {  
        for (int j = 0; j < other.cols(); j++) {  
            this->data[i][j] += other(i, j);  
        }  
    }  
    return *this;  
}  
  
Matrix &Matrix::operator-=(const Matrix &other) {  
    checkIfSameDimension(other);  
  
    for (int i = 0; i < other.rows(); i++) {  
        for (int j = 0; j < other.cols(); j++) {  
            this->data[i][j] -= other(i, j);  
        }  
    }  
    return *this;  
}  
  
Matrix &Matrix::operator*=(double scalar) {  
    for (int i = 0; i < this->rows(); i++) {  
        for (int j = 0; j < this->cols(); j++) {  
            this->data[i][j] *= scalar;  
        }  
    }  
    return *this;  
}  
  
Matrix &Matrix::operator/=(double scalar) {  
    for (int i = 0; i < this->rows(); i++) {  
        for (int j = 0; j < this->cols(); j++) {  
            this->data[i][j] /= scalar;  
        }  
    }  
    return *this;  
}  
  
Matrix &Matrix::operator=(const Matrix &other) = default;  
  
Matrix &Matrix::operator=(Matrix &&other) noexcept {  
    this->data = std::move(other.data);  
    return *this;  
}  
  

```

# Some helper methods
```cpp
void Matrix::checkIfSameDimension(const Matrix &other) const {  
    if (this->rows() != other.rows() || this->cols() != other.cols()) {  
        throw std::invalid_argument("Matrix dimensions do not match");  
    }  
}  
  
void Matrix::checkForDotMultiplication(const Matrix &other) const {  
    if (this->cols() != other.rows()) {  
        throw std::invalid_argument("Matrix dimensions do not match");  
    }  
}  
```

# Dot and transpose
```cpp  
Matrix Matrix::dot(const Matrix &other) const {  
    checkForDotMultiplication(other);  
    Matrix result = Matrix(this->rows(), other.cols());  
    for (int i = 0; i < this->rows(); i++) {  
        for (int j = 0; j < other.cols(); j++) {  
            double sum = 0;  
            for (int k = 0; k < this->cols(); k++) {  
                sum += this->data[i][k] * other(k, j);  
            }  
            result(i, j) = sum;  
        }  
    }  
    return result;  
} 
Matrix Matrix::transpose() const {  
    Matrix result = Matrix(this->cols(), this->rows());  
    for (int i = 0; i < this->rows(); i++) {  
        for (int j = 0; j < this->cols(); j++) {  
            result(j, i) = this->data[i][j];  
        }  
    }  
    return result;  
}  

```

# Basic stats
- sum
- mean
- size
- shape

```cpp
  
double Matrix::sum() const {  
    double sum = 0;  
    for (int i = 0; i < this->rows(); i++) {  
        for (int j = 0; j < this->cols(); j++) {  
            sum += this->data[i][j];  
        }  
    }  
    return sum;  
}  
  
double Matrix::mean() const {  
    return this->sum() / this->size();  
}  
  
int Matrix::size() const {  
    return this->rows() * this->cols();  
}  
  
 
int Matrix::rows() const {  
    return (int)data.size();  
}  
  
int Matrix::cols() const {  
    if (data.empty()) {  
        return 0;  
    }  
    return (int)data[0].size();  
  
std::pair<int, int> Matrix::shape() const {  
    return {rows(), cols()};  
}  
}  
```
# Row and column operations
```cpp
Matrix Matrix::getRow(int row) const {  
    Matrix result = Matrix(1, this->cols());  
    for (int i = 0; i < this->cols(); i++) {  
        result(0, i) = this->data[row][i];  
    }  
    return result;  
}  
  
Matrix Matrix::getCol(int col) const {  
    Matrix result = Matrix(this->rows(), 1);  
    for (int i = 0; i < this->rows(); i++) {  
        result(i, 0) = this->data[i][col];  
    }  
    return result;  
}  
  
void Matrix::addRow(const Matrix &row) {  
    if ((row.cols() != this->cols() && this->rows() > 0)|| row.rows() < 1) {  
        std::cerr << "Shape mismatch: " << this->printableShape() << " and " << row.printableShape() << std::endl;  
        throw std::invalid_argument("Row dimensions do not match");  
    }  
    this->data.push_back(row.data[0]);  
}  
  
void Matrix::setRow(int row, const Matrix &newRow) {  
    if (newRow.cols() != this->cols()) {  
        throw std::invalid_argument("Row dimensions do not match");  
    }  
    for (int j = 0; j < this->cols(); j++) {  
        this->data[row][j] = newRow(0, j);  
    }  
}  
  
Matrix Matrix::changeAt(int row, int col, double value) {  
    // check if row and col are in range  
    if (row >= this->rows() || col >= this->cols()) {  
        // print range  
        std::cout << "Row: " << row << " Col: " << col << std::endl;  
        std::cout << "Max Rows: " << this->rows() << " Max Cols: " << this->cols() << std::endl;  
        throw std::invalid_argument("Row or column out of range");  
  
    }  
  
    this->data[row][col] = value;  
    return *this;  
}  
```

# Formatted Output
```cpp
// To string  
std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {  
    for (int i = 0; i < matrix.rows(); i++) {  
        for (int j = 0; j < matrix.cols(); j++) {  
            os << matrix.data[i][j] << ", ";  
        }  
        os << std::endl;  
    }  
    return os;  
}  

void Matrix::print() const {  
    for (auto &row : data) {  
        for (auto &col : row) {  
            std::cout << col << " ";  
        }  
        std::cout << std::endl;  
    }  
}  
  
std::string Matrix::printableShape() const {  
    return "(" + std::to_string(this->rows()) + ", " + std::to_string(this->cols()) + ")";  
}  
  

```

# Simple check
```cpp
bool Matrix::isScalar() const {  
    return this->rows() == 1 && this->cols() == 1;  
}  
```

And we're done here, the build of the neural network will be in part 2