// SPDX-License-Identifier: MIT
// C++98.
#include "robo/Kinematics.h"

#include <cmath>
#include <limits>
#include <sstream>

namespace robo {

// ------------------------------------------------------------------- Joint

Joint::Joint(const DHParameters& dh, double lower, double upper,
             const std::string& name)
    : dh_(dh), lower_(lower), upper_(upper), name_(name) {
    if (lower > upper) {
        throw std::invalid_argument("Joint: lower limit exceeds upper limit");
    }
}

Joint::~Joint() {}

double Joint::clampToLimits(double q) const {
    if (q < lower_) {
        return lower_;
    }
    if (q > upper_) {
        return upper_;
    }
    return q;
}

Matrix Joint::dhTransform(double a, double alpha, double d, double theta) {
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);

    Matrix t = Matrix::identity(4);
    t(0, 0) = ct;  t(0, 1) = -st * ca;  t(0, 2) =  st * sa;  t(0, 3) = a * ct;
    t(1, 0) = st;  t(1, 1) =  ct * ca;  t(1, 2) = -ct * sa;  t(1, 3) = a * st;
    t(2, 0) = 0.0; t(2, 1) =  sa;       t(2, 2) =  ca;       t(2, 3) = d;
    return t;
}

// ----------------------------------------------------------- RevoluteJoint

RevoluteJoint::RevoluteJoint(const DHParameters& dh, double lower, double upper,
                             const std::string& name)
    : Joint(dh, lower, upper, name) {}

Matrix RevoluteJoint::transform(double q) const {
    // The joint variable adds to the fixed theta offset.
    return dhTransform(dh_.a, dh_.alpha, dh_.d, dh_.theta + q);
}

Matrix RevoluteJoint::jacobianColumn(const Matrix& zPrev, const Matrix& oPrev,
                                     const Matrix& oEnd) const {
    // Linear part: z_{i-1} x (o_n - o_{i-1}).  Angular part: z_{i-1}.
    const Matrix r = oEnd - oPrev;

    Matrix col(6, 1, 0.0);
    col(0, 0) = zPrev(1, 0) * r(2, 0) - zPrev(2, 0) * r(1, 0);
    col(1, 0) = zPrev(2, 0) * r(0, 0) - zPrev(0, 0) * r(2, 0);
    col(2, 0) = zPrev(0, 0) * r(1, 0) - zPrev(1, 0) * r(0, 0);
    col(3, 0) = zPrev(0, 0);
    col(4, 0) = zPrev(1, 0);
    col(5, 0) = zPrev(2, 0);
    return col;
}

// ---------------------------------------------------------- PrismaticJoint

PrismaticJoint::PrismaticJoint(const DHParameters& dh, double lower,
                               double upper, const std::string& name)
    : Joint(dh, lower, upper, name) {}

Matrix PrismaticJoint::transform(double q) const {
    // The joint variable adds to the fixed d offset.
    return dhTransform(dh_.a, dh_.alpha, dh_.d + q, dh_.theta);
}

Matrix PrismaticJoint::jacobianColumn(const Matrix& zPrev, const Matrix&,
                                      const Matrix&) const {
    // A sliding joint translates along z_{i-1} and produces no rotation.
    Matrix col(6, 1, 0.0);
    col(0, 0) = zPrev(0, 0);
    col(1, 0) = zPrev(1, 0);
    col(2, 0) = zPrev(2, 0);
    return col;
}

// ------------------------------------------------------------- SerialChain

SerialChain::SerialChain(const std::string& name) : name_(name) {}

SerialChain::~SerialChain() {
    for (std::size_t i = 0; i < joints_.size(); ++i) {
        delete joints_[i];
    }
}

void SerialChain::addJoint(Joint* joint) {
    if (joint == 0) {
        throw std::invalid_argument("SerialChain::addJoint: null joint");
    }
    joints_.push_back(joint);
}

const Joint& SerialChain::joint(std::size_t i) const {
    if (i >= joints_.size()) {
        throw std::out_of_range("SerialChain::joint: index out of range");
    }
    return *joints_[i];
}

void SerialChain::checkDof(const std::vector<double>& q) const {
    if (q.size() != joints_.size()) {
        std::ostringstream oss;
        oss << "SerialChain: expected " << joints_.size()
            << " joint values, got " << q.size();
        throw std::invalid_argument(oss.str());
    }
}

Matrix SerialChain::forwardKinematics(const std::vector<double>& q,
                                      std::size_t upTo) const {
    checkDof(q);
    if (upTo > joints_.size()) {
        throw std::out_of_range("SerialChain::forwardKinematics: frame index too large");
    }
    Matrix t = Matrix::identity(4);
    for (std::size_t i = 0; i < upTo; ++i) {
        t = t * joints_[i]->transform(q[i]);
    }
    return t;
}

Matrix SerialChain::forwardKinematics(const std::vector<double>& q) const {
    return forwardKinematics(q, joints_.size());
}

Matrix SerialChain::position(const std::vector<double>& q) const {
    return forwardKinematics(q).block(0, 3, 3, 1);
}

Matrix SerialChain::jacobian(const std::vector<double>& q) const {
    checkDof(q);
    const std::size_t n = joints_.size();

    // Walk the chain once, caching each frame's z axis and origin.
    std::vector<Matrix> zAxes;
    std::vector<Matrix> origins;
    zAxes.reserve(n + 1);
    origins.reserve(n + 1);

    Matrix t = Matrix::identity(4);
    zAxes.push_back(t.block(0, 2, 3, 1));
    origins.push_back(t.block(0, 3, 3, 1));

    for (std::size_t i = 0; i < n; ++i) {
        t = t * joints_[i]->transform(q[i]);
        zAxes.push_back(t.block(0, 2, 3, 1));
        origins.push_back(t.block(0, 3, 3, 1));
    }

    const Matrix oEnd = origins[n];
    Matrix j(6, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const Matrix col = joints_[i]->jacobianColumn(zAxes[i], origins[i], oEnd);
        j.setBlock(0, i, col);
    }
    return j;
}

Matrix SerialChain::numericJacobian(const std::vector<double>& q,
                                    double step) const {
    checkDof(q);
    const std::size_t n = joints_.size();
    Matrix j(6, n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> plus(q);
        std::vector<double> minus(q);
        plus[i] += step;
        minus[i] -= step;

        const Matrix pPlus = position(plus);
        const Matrix pMinus = position(minus);
        for (std::size_t r = 0; r < 3; ++r) {
            j(r, i) = (pPlus(r, 0) - pMinus(r, 0)) / (2.0 * step);
        }

        // Angular rows from the rotation matrix derivative:
        // S = dR/dq * R^T is skew-symmetric; its vector form is the column.
        const Matrix rPlus = forwardKinematics(plus).block(0, 0, 3, 3);
        const Matrix rMinus = forwardKinematics(minus).block(0, 0, 3, 3);
        const Matrix dR = (rPlus - rMinus) * (1.0 / (2.0 * step));
        const Matrix rNow = forwardKinematics(q).block(0, 0, 3, 3);
        const Matrix s = dR * rNow.transpose();

        j(3, i) = 0.5 * (s(2, 1) - s(1, 2));
        j(4, i) = 0.5 * (s(0, 2) - s(2, 0));
        j(5, i) = 0.5 * (s(1, 0) - s(0, 1));
    }
    return j;
}

ManipulabilityReport SerialChain::analyseManipulability(
    const std::vector<double>& q, double singularTolerance) const {
    std::vector<std::size_t> linearRows;
    linearRows.push_back(0);
    linearRows.push_back(1);
    linearRows.push_back(2);
    return analyseManipulability(q, linearRows, singularTolerance);
}

ManipulabilityReport SerialChain::analyseManipulability(
    const std::vector<double>& q, const std::vector<std::size_t>& taskRows,
    double singularTolerance) const {
    if (taskRows.empty()) {
        throw std::invalid_argument(
            "analyseManipulability: taskRows must not be empty");
    }

    const Matrix j = jacobian(q);
    Matrix jv(taskRows.size(), j.cols(), 0.0);
    for (std::size_t i = 0; i < taskRows.size(); ++i) {
        if (taskRows[i] >= 6) {
            throw std::out_of_range(
                "analyseManipulability: task row index must be < 6");
        }
        for (std::size_t c = 0; c < j.cols(); ++c) {
            jv(i, c) = j(taskRows[i], c);
        }
    }

    // Singular values of Jv are the square roots of the eigenvalues of
    // Jv * Jv^T, which is symmetric positive semi-definite by construction.
    const Matrix jjt = jv * jv.transpose();
    const SymmetricEigen eig(jjt);

    ManipulabilityReport report;
    report.principalAxes = eig.eigenvectors();
    report.singularValues.resize(eig.eigenvalues().size());

    double product = 1.0;
    for (std::size_t i = 0; i < eig.eigenvalues().size(); ++i) {
        // Clamp: round-off can push a zero eigenvalue slightly negative.
        const double lambda =
            eig.eigenvalues()[i] > 0.0 ? eig.eigenvalues()[i] : 0.0;
        const double sigma = std::sqrt(lambda);
        report.singularValues[i] = sigma;
        product *= sigma;
    }

    report.yoshikawaIndex = product;
    report.minimumSingularValue = report.singularValues.empty()
                                      ? 0.0
                                      : report.singularValues.back();

    const double sMax = report.singularValues.empty() ? 0.0
                                                      : report.singularValues.front();
    if (report.minimumSingularValue < singularTolerance) {
        report.nearSingular = true;
        report.conditionNumber = std::numeric_limits<double>::infinity();
    } else {
        report.nearSingular = false;
        report.conditionNumber = sMax / report.minimumSingularValue;
    }
    return report;
}

bool SerialChain::withinLimits(const std::vector<double>& q) const {
    checkDof(q);
    for (std::size_t i = 0; i < joints_.size(); ++i) {
        if (!joints_[i]->withinLimits(q[i])) {
            return false;
        }
    }
    return true;
}

void SerialChain::clampToLimits(std::vector<double>& q) const {
    checkDof(q);
    for (std::size_t i = 0; i < joints_.size(); ++i) {
        q[i] = joints_[i]->clampToLimits(q[i]);
    }
}

// ---------------------------------------------------------------- IkSolver

IkSolver::IkSolver(const SerialChain& chain, const IkOptions& options)
    : chain_(chain), options_(options) {
    if (options.damping < 0.0) {
        throw std::invalid_argument("IkOptions: damping must be >= 0");
    }
    if (options.maxIterations <= 0) {
        throw std::invalid_argument("IkOptions: maxIterations must be > 0");
    }
}

IkResult IkSolver::solvePosition(const Matrix& target,
                                 const std::vector<double>& seed) const {
    if (target.rows() != 3 || target.cols() != 1) {
        throw std::invalid_argument("IkSolver: target must be a 3x1 position");
    }
    if (seed.size() != chain_.dof()) {
        throw std::invalid_argument("IkSolver: seed length must equal the chain DOF");
    }

    IkResult result;
    std::vector<double> q(seed);
    const std::size_t n = chain_.dof();
    const double lambda2 = options_.damping * options_.damping;

    for (int iter = 0; iter < options_.maxIterations; ++iter) {
        const Matrix error = target - chain_.position(q);
        const double err = error.norm();
        result.iterations = iter + 1;
        result.finalError = err;

        if (err < options_.positionTolerance) {
            result.converged = true;
            break;
        }

        const Matrix jv = chain_.jacobian(q).block(0, 0, 3, n);

        // dq = J^T (J J^T + lambda^2 I)^-1 e, solved rather than inverted.
        Matrix a = jv * jv.transpose();
        for (std::size_t i = 0; i < 3; ++i) {
            a(i, i) += lambda2;
        }
        const Matrix y = Matrix::solve(a, error);
        Matrix dq = jv.transpose() * y;

        // Cap the step: far from the target the linearisation is poor.
        const double stepNorm = dq.norm();
        if (stepNorm > options_.maxStep) {
            dq *= options_.maxStep / stepNorm;
        }

        for (std::size_t i = 0; i < n; ++i) {
            q[i] += dq(i, 0);
        }
        if (options_.respectJointLimits) {
            chain_.clampToLimits(q);
        }
    }

    result.solution = q;
    return result;
}

}  // namespace robo
