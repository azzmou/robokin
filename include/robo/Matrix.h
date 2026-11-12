// SPDX-License-Identifier: MIT
// C++98. No external dependencies.
#ifndef ROBO_MATRIX_H
#define ROBO_MATRIX_H

#include <cstddef>
#include <iosfwd>
#include <stdexcept>
#include <vector>

namespace robo {

/// Dense matrix of doubles, stored row-major.
///
/// Written for small matrices (up to roughly 12x12), which is the range a
/// serial manipulator needs: 4x4 transforms, 6xN Jacobians, NxN inertia-like
/// blocks. The algorithms below are O(n^3) and allocate freely; that is the
/// right trade at this size and keeps the code readable.
class Matrix {
public:
    Matrix();
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0);

    static Matrix identity(std::size_t n);
    static Matrix zeros(std::size_t rows, std::size_t cols);

    /// Column vector from a plain array of length n.
    static Matrix columnVector(const std::vector<double>& values);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    bool isSquare() const { return rows_ == cols_; }

    double& operator()(std::size_t r, std::size_t c);
    double operator()(std::size_t r, std::size_t c) const;

    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;
    Matrix operator*(const Matrix& rhs) const;
    Matrix operator*(double scalar) const;

    Matrix& operator+=(const Matrix& rhs);
    Matrix& operator-=(const Matrix& rhs);
    Matrix& operator*=(double scalar);

    bool operator==(const Matrix& rhs) const;
    bool operator!=(const Matrix& rhs) const { return !(*this == rhs); }

    Matrix transpose() const;

    /// Gauss-Jordan inverse with partial pivoting.
    /// Throws std::runtime_error if the matrix is singular to working precision.
    Matrix inverse() const;

    /// Determinant by LU decomposition with partial pivoting.
    double determinant() const;

    /// Frobenius norm. For a column vector this is the Euclidean norm.
    double norm() const;

    /// Extract a sub-block. Throws std::out_of_range if it does not fit.
    Matrix block(std::size_t r0, std::size_t c0,
                 std::size_t nrows, std::size_t ncols) const;

    /// Write a sub-block in place at (r0, c0).
    void setBlock(std::size_t r0, std::size_t c0, const Matrix& m);

    /// Solve A x = b for x using Gaussian elimination with partial pivoting.
    /// Cheaper and better conditioned than forming inverse() * b.
    static Matrix solve(const Matrix& A, const Matrix& b);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;

    void checkSameShape(const Matrix& rhs, const char* op) const;
};

Matrix operator*(double scalar, const Matrix& m);
std::ostream& operator<<(std::ostream& os, const Matrix& m);

/// Eigenvalues and eigenvectors of a real symmetric matrix, by the cyclic
/// Jacobi rotation method.
///
/// Used here to get the principal axes of the manipulability ellipsoid from
/// J * J^T, which is symmetric positive semi-definite by construction. Jacobi
/// is slower than a Householder-based routine on large matrices but it is
/// short, numerically sound and exact enough for the 3x3 and 6x6 cases.
class SymmetricEigen {
public:
    /// @param a            symmetric input matrix
    /// @param maxSweeps    safety bound on the number of Jacobi sweeps
    /// @param tolerance    stop once the off-diagonal norm falls below this
    SymmetricEigen(const Matrix& a, int maxSweeps = 50, double tolerance = 1e-12);

    /// Eigenvalues, sorted descending.
    const std::vector<double>& eigenvalues() const { return values_; }

    /// Eigenvectors as columns, in the same order as eigenvalues().
    const Matrix& eigenvectors() const { return vectors_; }

    int sweepsUsed() const { return sweeps_; }

private:
    std::vector<double> values_;
    Matrix vectors_;
    int sweeps_;
};

}  // namespace robo

#endif  // ROBO_MATRIX_H
