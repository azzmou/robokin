// SPDX-License-Identifier: MIT
// C++98. No external dependencies.
#ifndef ROBO_KINEMATICS_H
#define ROBO_KINEMATICS_H

#include "robo/Matrix.h"

#include <cstddef>
#include <string>
#include <vector>

namespace robo {

/// Denavit-Hartenberg parameters for one link, standard (distal) convention:
///
///     T = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
///
/// For a revolute joint, theta is the variable and d is fixed.
/// For a prismatic joint, d is the variable and theta is fixed.
struct DHParameters {
    double a;      ///< link length  [m]
    double alpha;  ///< link twist   [rad]
    double d;      ///< link offset  [m]
    double theta;  ///< joint angle  [rad]

    DHParameters() : a(0.0), alpha(0.0), d(0.0), theta(0.0) {}
    DHParameters(double a_, double alpha_, double d_, double theta_)
        : a(a_), alpha(alpha_), d(d_), theta(theta_) {}
};

/// Abstract joint. Subclasses decide which DH parameter the joint variable
/// drives and how the Jacobian column is formed, so SerialChain never needs
/// to branch on a joint-type tag.
class Joint {
public:
    Joint(const DHParameters& dh, double lower, double upper,
          const std::string& name);
    virtual ~Joint();

    /// Homogeneous transform of this link for joint value q.
    virtual Matrix transform(double q) const = 0;

    /// Contribution to the geometric Jacobian.
    /// @param zPrev    z axis of the previous frame, in base coordinates (3x1)
    /// @param oPrev    origin of the previous frame, in base coordinates (3x1)
    /// @param oEnd     origin of the end-effector frame, in base coordinates (3x1)
    /// @return 6x1 column: linear part on top, angular part below.
    virtual Matrix jacobianColumn(const Matrix& zPrev, const Matrix& oPrev,
                                  const Matrix& oEnd) const = 0;

    virtual bool isRevolute() const = 0;

    const DHParameters& dh() const { return dh_; }
    const std::string& name() const { return name_; }
    double lowerLimit() const { return lower_; }
    double upperLimit() const { return upper_; }

    bool withinLimits(double q) const { return q >= lower_ && q <= upper_; }
    double clampToLimits(double q) const;

protected:
    /// Build the standard DH transform from explicit parameters.
    static Matrix dhTransform(double a, double alpha, double d, double theta);

    DHParameters dh_;
    double lower_;
    double upper_;
    std::string name_;

private:
    Joint(const Joint&);             // not copyable: chains own their joints
    Joint& operator=(const Joint&);
};

class RevoluteJoint : public Joint {
public:
    RevoluteJoint(const DHParameters& dh, double lower, double upper,
                  const std::string& name = "revolute");
    virtual Matrix transform(double q) const;
    virtual Matrix jacobianColumn(const Matrix& zPrev, const Matrix& oPrev,
                                  const Matrix& oEnd) const;
    virtual bool isRevolute() const { return true; }
};

class PrismaticJoint : public Joint {
public:
    PrismaticJoint(const DHParameters& dh, double lower, double upper,
                   const std::string& name = "prismatic");
    virtual Matrix transform(double q) const;
    virtual Matrix jacobianColumn(const Matrix& zPrev, const Matrix& oPrev,
                                  const Matrix& oEnd) const;
    virtual bool isRevolute() const { return false; }
};

/// Result of a manipulability analysis at one configuration.
struct ManipulabilityReport {
    double yoshikawaIndex;              ///< sqrt(det(J J^T)): volume of the ellipsoid
    double conditionNumber;             ///< sigma_max / sigma_min, inf when singular
    double minimumSingularValue;        ///< distance to the nearest singularity
    std::vector<double> singularValues; ///< descending
    Matrix principalAxes;               ///< ellipsoid axes as columns
    bool nearSingular;

    ManipulabilityReport()
        : yoshikawaIndex(0.0), conditionNumber(0.0),
          minimumSingularValue(0.0), nearSingular(false) {}
};

/// Serial kinematic chain. Owns its joints and deletes them.
class SerialChain {
public:
    explicit SerialChain(const std::string& name = "chain");
    ~SerialChain();

    /// Takes ownership of the joint.
    void addJoint(Joint* joint);

    std::size_t dof() const { return joints_.size(); }
    const std::string& name() const { return name_; }
    const Joint& joint(std::size_t i) const;

    /// Transform from the base frame to the end-effector, 4x4.
    Matrix forwardKinematics(const std::vector<double>& q) const;

    /// Transform from the base frame to frame `upTo` (0 = base, dof() = tool).
    Matrix forwardKinematics(const std::vector<double>& q,
                             std::size_t upTo) const;

    /// End-effector position, 3x1.
    Matrix position(const std::vector<double>& q) const;

    /// Geometric Jacobian, 6 x dof: rows 0-2 linear, rows 3-5 angular.
    Matrix jacobian(const std::vector<double>& q) const;

    /// Jacobian by central finite differences. Independent of the analytic
    /// path, so the tests can check one against the other.
    Matrix numericJacobian(const std::vector<double>& q,
                           double step = 1e-6) const;

    /// Singular-value analysis of the linear (position) rows of the Jacobian,
    /// i.e. assuming a full 3-D positioning task.
    ManipulabilityReport analyseManipulability(const std::vector<double>& q,
                                               double singularTolerance = 1e-4) const;

    /// Same analysis restricted to chosen Jacobian rows.
    ///
    /// The task dimension matters: a planar arm has a structurally empty z
    /// direction, so analysing it against all three linear rows would report
    /// it as permanently singular. Pass the rows the task actually spans
    /// (for a planar arm, rows 0 and 1) and the numbers become meaningful.
    /// Rows 0-2 are linear, rows 3-5 angular.
    ManipulabilityReport analyseManipulability(const std::vector<double>& q,
                                               const std::vector<std::size_t>& taskRows,
                                               double singularTolerance = 1e-4) const;

    bool withinLimits(const std::vector<double>& q) const;
    void clampToLimits(std::vector<double>& q) const;

private:
    SerialChain(const SerialChain&);
    SerialChain& operator=(const SerialChain&);

    void checkDof(const std::vector<double>& q) const;

    std::string name_;
    std::vector<Joint*> joints_;
};

/// Settings for the iterative inverse-kinematics solver.
struct IkOptions {
    int maxIterations;
    double positionTolerance;  ///< [m]
    double damping;            ///< lambda in damped least squares
    double maxStep;            ///< cap on |dq| per iteration [rad or m]
    bool respectJointLimits;

    IkOptions()
        : maxIterations(200), positionTolerance(1e-6), damping(0.05),
          maxStep(0.2), respectJointLimits(true) {}
};

struct IkResult {
    bool converged;
    int iterations;
    double finalError;
    std::vector<double> solution;

    IkResult() : converged(false), iterations(0), finalError(0.0) {}
};

/// Position-only inverse kinematics by damped least squares
/// (Levenberg-Marquardt).
///
///     dq = J^T (J J^T + lambda^2 I)^-1 * e
///
/// The damping term keeps the step bounded near singularities, where a plain
/// pseudo-inverse would demand enormous joint velocities. The cost is a small
/// steady-state error, which is why lambda should be small away from
/// singular configurations.
class IkSolver {
public:
    IkSolver(const SerialChain& chain, const IkOptions& options = IkOptions());

    /// @param target  desired end-effector position, 3x1
    /// @param seed    initial guess; its length must equal the chain DOF
    IkResult solvePosition(const Matrix& target,
                           const std::vector<double>& seed) const;

private:
    const SerialChain& chain_;
    IkOptions options_;
};

}  // namespace robo

#endif  // ROBO_KINEMATICS_H
