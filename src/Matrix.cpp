// SPDX-License-Identifier: MIT
// C++98.
#include "robo/Matrix.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace robo {

namespace {

std::string shapeError(const char* op, std::size_t ar, std::size_t ac,
                       std::size_t br, std::size_t bc) {
    std::ostringstream oss;
    oss << "Matrix " << op << ": incompatible shapes "
        << ar << "x" << ac << " and " << br << "x" << bc;
    return oss.str();
}

}  // namespace

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

Matrix Matrix::identity(std::size_t n) {
    Matrix m(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        m(i, i) = 1.0;
    }
    return m;
}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::columnVector(const std::vector<double>& values) {
    Matrix m(values.size(), 1, 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        m(i, 0) = values[i];
    }
    return m;
}

double& Matrix::operator()(std::size_t r, std::size_t c) {
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Matrix: index out of range");
    }
    return data_[r * cols_ + c];
}

double Matrix::operator()(std::size_t r, std::size_t c) const {
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Matrix: index out of range");
    }
    return data_[r * cols_ + c];
}

void Matrix::checkSameShape(const Matrix& rhs, const char* op) const {
    if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
        throw std::invalid_argument(
            shapeError(op, rows_, cols_, rhs.rows_, rhs.cols_));
    }
}

Matrix Matrix::operator+(const Matrix& rhs) const {
    checkSameShape(rhs, "operator+");
    Matrix out(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        out.data_[i] = data_[i] + rhs.data_[i];
    }
    return out;
}

Matrix Matrix::operator-(const Matrix& rhs) const {
    checkSameShape(rhs, "operator-");
    Matrix out(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        out.data_[i] = data_[i] - rhs.data_[i];
    }
    return out;
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    if (cols_ != rhs.rows_) {
        throw std::invalid_argument(
            shapeError("operator*", rows_, cols_, rhs.rows_, rhs.cols_));
    }
    Matrix out(rows_, rhs.cols_, 0.0);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < cols_; ++k) {
            const double a = data_[i * cols_ + k];
            if (a == 0.0) {
                continue;  // transforms and Jacobians are sparse in practice
            }
            for (std::size_t j = 0; j < rhs.cols_; ++j) {
                out.data_[i * out.cols_ + j] += a * rhs.data_[k * rhs.cols_ + j];
            }
        }
    }
    return out;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix out(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        out.data_[i] = data_[i] * scalar;
    }
    return out;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
    checkSameShape(rhs, "operator+=");
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] += rhs.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
    checkSameShape(rhs, "operator-=");
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= rhs.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

bool Matrix::operator==(const Matrix& rhs) const {
    return rows_ == rhs.rows_ && cols_ == rhs.cols_ && data_ == rhs.data_;
}

Matrix Matrix::transpose() const {
    Matrix out(cols_, rows_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            out(j, i) = data_[i * cols_ + j];
        }
    }
    return out;
}

Matrix Matrix::inverse() const {
    if (!isSquare()) {
        throw std::invalid_argument("Matrix::inverse: matrix is not square");
    }
    const std::size_t n = rows_;

    Matrix a(*this);
    Matrix inv = identity(n);

    for (std::size_t col = 0; col < n; ++col) {
        // Partial pivoting: largest magnitude pivot in this column.
        std::size_t pivot = col;
        double best = std::fabs(a(col, col));
        for (std::size_t r = col + 1; r < n; ++r) {
            const double mag = std::fabs(a(r, col));
            if (mag > best) {
                best = mag;
                pivot = r;
            }
        }
        if (best < 1e-14) {
            throw std::runtime_error("Matrix::inverse: matrix is singular");
        }
        if (pivot != col) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(a(col, c), a(pivot, c));
                std::swap(inv(col, c), inv(pivot, c));
            }
        }

        const double d = a(col, col);
        for (std::size_t c = 0; c < n; ++c) {
            a(col, c) /= d;
            inv(col, c) /= d;
        }

        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) {
                continue;
            }
            const double factor = a(r, col);
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t c = 0; c < n; ++c) {
                a(r, c) -= factor * a(col, c);
                inv(r, c) -= factor * inv(col, c);
            }
        }
    }
    return inv;
}

double Matrix::determinant() const {
    if (!isSquare()) {
        throw std::invalid_argument("Matrix::determinant: matrix is not square");
    }
    const std::size_t n = rows_;
    Matrix a(*this);
    double det = 1.0;

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double best = std::fabs(a(col, col));
        for (std::size_t r = col + 1; r < n; ++r) {
            const double mag = std::fabs(a(r, col));
            if (mag > best) {
                best = mag;
                pivot = r;
            }
        }
        if (best < 1e-300) {
            return 0.0;
        }
        if (pivot != col) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(a(col, c), a(pivot, c));
            }
            det = -det;
        }
        det *= a(col, col);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double factor = a(r, col) / a(col, col);
            for (std::size_t c = col; c < n; ++c) {
                a(r, c) -= factor * a(col, c);
            }
        }
    }
    return det;
}

double Matrix::norm() const {
    double sum = 0.0;
    for (std::size_t i = 0; i < data_.size(); ++i) {
        sum += data_[i] * data_[i];
    }
    return std::sqrt(sum);
}

Matrix Matrix::block(std::size_t r0, std::size_t c0,
                     std::size_t nrows, std::size_t ncols) const {
    if (r0 + nrows > rows_ || c0 + ncols > cols_) {
        throw std::out_of_range("Matrix::block: block does not fit");
    }
    Matrix out(nrows, ncols);
    for (std::size_t i = 0; i < nrows; ++i) {
        for (std::size_t j = 0; j < ncols; ++j) {
            out(i, j) = data_[(r0 + i) * cols_ + (c0 + j)];
        }
    }
    return out;
}

void Matrix::setBlock(std::size_t r0, std::size_t c0, const Matrix& m) {
    if (r0 + m.rows() > rows_ || c0 + m.cols() > cols_) {
        throw std::out_of_range("Matrix::setBlock: block does not fit");
    }
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            data_[(r0 + i) * cols_ + (c0 + j)] = m(i, j);
        }
    }
}

Matrix Matrix::solve(const Matrix& A, const Matrix& b) {
    if (!A.isSquare()) {
        throw std::invalid_argument("Matrix::solve: A is not square");
    }
    if (A.rows() != b.rows()) {
        throw std::invalid_argument("Matrix::solve: shape mismatch between A and b");
    }

    const std::size_t n = A.rows();
    const std::size_t m = b.cols();
    Matrix a(A);
    Matrix x(b);

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double best = std::fabs(a(col, col));
        for (std::size_t r = col + 1; r < n; ++r) {
            const double mag = std::fabs(a(r, col));
            if (mag > best) {
                best = mag;
                pivot = r;
            }
        }
        if (best < 1e-14) {
            throw std::runtime_error("Matrix::solve: matrix is singular");
        }
        if (pivot != col) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(a(col, c), a(pivot, c));
            }
            for (std::size_t c = 0; c < m; ++c) {
                std::swap(x(col, c), x(pivot, c));
            }
        }

        for (std::size_t r = col + 1; r < n; ++r) {
            const double factor = a(r, col) / a(col, col);
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t c = col; c < n; ++c) {
                a(r, c) -= factor * a(col, c);
            }
            for (std::size_t c = 0; c < m; ++c) {
                x(r, c) -= factor * x(col, c);
            }
        }
    }

    // Back substitution.
    for (std::size_t ri = n; ri > 0; --ri) {
        const std::size_t r = ri - 1;
        for (std::size_t c = 0; c < m; ++c) {
            double sum = x(r, c);
            for (std::size_t k = r + 1; k < n; ++k) {
                sum -= a(r, k) * x(k, c);
            }
            x(r, c) = sum / a(r, r);
        }
    }
    return x;
}

Matrix operator*(double scalar, const Matrix& m) {
    return m * scalar;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    os << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < m.rows(); ++i) {
        os << "[ ";
        for (std::size_t j = 0; j < m.cols(); ++j) {
            os << std::setw(9) << m(i, j) << ' ';
        }
        os << "]\n";
    }
    return os;
}

// ---------------------------------------------------------- SymmetricEigen

SymmetricEigen::SymmetricEigen(const Matrix& a, int maxSweeps, double tolerance)
    : sweeps_(0) {
    if (!a.isSquare()) {
        throw std::invalid_argument("SymmetricEigen: matrix is not square");
    }
    const std::size_t n = a.rows();

    Matrix d(a);
    Matrix v = Matrix::identity(n);

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        // Off-diagonal magnitude: the quantity Jacobi drives to zero.
        double off = 0.0;
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                off += d(p, q) * d(p, q);
            }
        }
        sweeps_ = sweep;
        if (std::sqrt(2.0 * off) < tolerance) {
            break;
        }

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                if (std::fabs(d(p, q)) < 1e-300) {
                    continue;
                }
                // Rotation angle that annihilates d(p, q).
                const double theta = (d(q, q) - d(p, p)) / (2.0 * d(p, q));
                const double sign = theta >= 0.0 ? 1.0 : -1.0;
                const double t =
                    sign / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;

                for (std::size_t k = 0; k < n; ++k) {
                    const double dkp = d(k, p);
                    const double dkq = d(k, q);
                    d(k, p) = c * dkp - s * dkq;
                    d(k, q) = s * dkp + c * dkq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double dpk = d(p, k);
                    const double dqk = d(q, k);
                    d(p, k) = c * dpk - s * dqk;
                    d(q, k) = s * dpk + c * dqk;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double vkp = v(k, p);
                    const double vkq = v(k, q);
                    v(k, p) = c * vkp - s * vkq;
                    v(k, q) = s * vkp + c * vkq;
                }
            }
        }
    }

    // Collect and sort descending, carrying the eigenvectors along.
    std::vector<std::size_t> order(n);
    values_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        values_[i] = d(i, i);
        order[i] = i;
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (values_[j] > values_[i]) {
                std::swap(values_[i], values_[j]);
                std::swap(order[i], order[j]);
            }
        }
    }

    vectors_ = Matrix(n, n);
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t row = 0; row < n; ++row) {
            vectors_(row, col) = v(row, order[col]);
        }
    }
}

}  // namespace robo
