// SPDX-License-Identifier: MIT
// C++98. Dependency-free test runner: no GoogleTest, so the repo builds
// anywhere with CMake and a compiler.
#include "robo/Kinematics.h"
#include "robo/Matrix.h"
#include "robo/Trajectory.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("    FAIL  %s\n", what.c_str());
    }
}

void checkNear(double got, double want, double tol, const std::string& what) {
    ++g_checks;
    if (std::fabs(got - want) > tol) {
        ++g_failures;
        std::printf("    FAIL  %s (got %.8f, want %.8f, tol %.1e)\n",
                    what.c_str(), got, want, tol);
    }
}

const double kPi = 3.14159265358979323846;

// ------------------------------------------------------------------ Matrix

void testMatrixArithmetic() {
    robo::Matrix a(2, 2);
    a(0, 0) = 1.0; a(0, 1) = 2.0;
    a(1, 0) = 3.0; a(1, 1) = 4.0;

    robo::Matrix b = robo::Matrix::identity(2);
    const robo::Matrix c = a * b;
    check(c == a, "multiplying by the identity leaves the matrix unchanged");

    const robo::Matrix sum = a + a;
    checkNear(sum(1, 1), 8.0, 1e-12, "element-wise addition");

    const robo::Matrix scaled = 2.0 * a;
    checkNear(scaled(0, 1), 4.0, 1e-12, "scalar multiplication is commutative");

    checkNear(a.determinant(), -2.0, 1e-12, "2x2 determinant");

    const robo::Matrix t = a.transpose();
    checkNear(t(0, 1), 3.0, 1e-12, "transpose swaps off-diagonal entries");
}

void testMatrixInverseAndSolve() {
    robo::Matrix a(3, 3);
    a(0, 0) = 2.0; a(0, 1) = -1.0; a(0, 2) = 0.0;
    a(1, 0) = -1.0; a(1, 1) = 2.0; a(1, 2) = -1.0;
    a(2, 0) = 0.0; a(2, 1) = -1.0; a(2, 2) = 2.0;

    const robo::Matrix inv = a.inverse();
    const robo::Matrix shouldBeI = a * inv;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            checkNear(shouldBeI(i, j), i == j ? 1.0 : 0.0, 1e-10,
                      "A * A^-1 is the identity");
        }
    }

    robo::Matrix b(3, 1);
    b(0, 0) = 1.0; b(1, 0) = 2.0; b(2, 0) = 3.0;
    const robo::Matrix x = robo::Matrix::solve(a, b);
    const robo::Matrix residual = a * x - b;
    check(residual.norm() < 1e-10, "solve() residual is numerically zero");
}

void testMatrixSingularDetection() {
    robo::Matrix s(2, 2);
    s(0, 0) = 1.0; s(0, 1) = 2.0;
    s(1, 0) = 2.0; s(1, 1) = 4.0;  // second row is twice the first

    checkNear(s.determinant(), 0.0, 1e-12, "singular matrix has zero determinant");

    bool threw = false;
    try {
        s.inverse();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "inverting a singular matrix throws");
}

void testSymmetricEigen() {
    // Known spectrum: diag(5, 2) rotated by 30 degrees.
    const double angle = kPi / 6.0;
    robo::Matrix r(2, 2);
    r(0, 0) = std::cos(angle); r(0, 1) = -std::sin(angle);
    r(1, 0) = std::sin(angle); r(1, 1) = std::cos(angle);

    robo::Matrix d(2, 2);
    d(0, 0) = 5.0; d(1, 1) = 2.0;

    const robo::Matrix a = r * d * r.transpose();
    const robo::SymmetricEigen eig(a);

    checkNear(eig.eigenvalues()[0], 5.0, 1e-9, "largest eigenvalue recovered");
    checkNear(eig.eigenvalues()[1], 2.0, 1e-9, "smallest eigenvalue recovered");

    // The first eigenvector should be the rotated x axis, up to sign.
    const robo::Matrix v = eig.eigenvectors();
    const double dot = std::fabs(v(0, 0) * r(0, 0) + v(1, 0) * r(1, 0));
    checkNear(dot, 1.0, 1e-9, "principal eigenvector matches the rotation axis");
}

// -------------------------------------------------------------- Kinematics

/// Planar 3R arm with unit links: the textbook case, and the one whose
/// closed-form answers can be checked by hand.
robo::SerialChain* makePlanar3R() {
    robo::SerialChain* chain = new robo::SerialChain("planar-3R");
    const double lim = kPi;
    chain->addJoint(new robo::RevoluteJoint(
        robo::DHParameters(1.0, 0.0, 0.0, 0.0), -lim, lim, "q1"));
    chain->addJoint(new robo::RevoluteJoint(
        robo::DHParameters(1.0, 0.0, 0.0, 0.0), -lim, lim, "q2"));
    chain->addJoint(new robo::RevoluteJoint(
        robo::DHParameters(1.0, 0.0, 0.0, 0.0), -lim, lim, "q3"));
    return chain;
}

void testForwardKinematicsAgainstClosedForm() {
    robo::SerialChain* arm = makePlanar3R();

    std::vector<double> q(3, 0.0);
    robo::Matrix p = arm->position(q);
    checkNear(p(0, 0), 3.0, 1e-12, "fully extended arm reaches x = 3");
    checkNear(p(1, 0), 0.0, 1e-12, "fully extended arm stays on the x axis");

    q[0] = kPi / 2.0;
    p = arm->position(q);
    checkNear(p(0, 0), 0.0, 1e-12, "rotating the base by 90 deg empties x");
    checkNear(p(1, 0), 3.0, 1e-12, "rotating the base by 90 deg puts reach on y");

    // Against the planar closed form: x = c1 + c12 + c123, y = s1 + s12 + s123.
    q[0] = 0.3; q[1] = -0.7; q[2] = 1.1;
    p = arm->position(q);
    const double x = std::cos(q[0]) + std::cos(q[0] + q[1]) +
                     std::cos(q[0] + q[1] + q[2]);
    const double y = std::sin(q[0]) + std::sin(q[0] + q[1]) +
                     std::sin(q[0] + q[1] + q[2]);
    checkNear(p(0, 0), x, 1e-12, "x matches the planar closed form");
    checkNear(p(1, 0), y, 1e-12, "y matches the planar closed form");

    delete arm;
}

void testAnalyticJacobianMatchesNumeric() {
    robo::SerialChain* arm = makePlanar3R();

    std::vector<double> q(3);
    q[0] = 0.4; q[1] = 0.9; q[2] = -0.6;

    const robo::Matrix ja = arm->jacobian(q);
    const robo::Matrix jn = arm->numericJacobian(q);
    const robo::Matrix diff = ja - jn;

    check(diff.norm() < 1e-5,
          "analytic Jacobian agrees with central differences");

    delete arm;
}

void testPrismaticJointJacobian() {
    robo::SerialChain chain("rp");
    chain.addJoint(new robo::RevoluteJoint(
        robo::DHParameters(0.0, kPi / 2.0, 0.5, 0.0), -kPi, kPi, "q1"));
    chain.addJoint(new robo::PrismaticJoint(
        robo::DHParameters(0.0, 0.0, 0.0, 0.0), 0.0, 1.0, "d2"));

    std::vector<double> q(2);
    q[0] = 0.3; q[1] = 0.4;

    const robo::Matrix ja = chain.jacobian(q);
    const robo::Matrix jn = chain.numericJacobian(q);
    check((ja - jn).norm() < 1e-5,
          "prismatic column agrees with central differences");

    // A sliding joint contributes no angular velocity.
    checkNear(ja(3, 1), 0.0, 1e-12, "prismatic joint has no angular x component");
    checkNear(ja(4, 1), 0.0, 1e-12, "prismatic joint has no angular y component");
    checkNear(ja(5, 1), 0.0, 1e-12, "prismatic joint has no angular z component");
}

void testManipulabilityDropsAtSingularity() {
    robo::SerialChain* arm = makePlanar3R();

    // The arm is planar, so the task spans x and y only. Analysing it
    // against all three linear rows would call it permanently singular.
    std::vector<std::size_t> planarRows;
    planarRows.push_back(0);
    planarRows.push_back(1);

    // Fully stretched: the classic boundary singularity.
    std::vector<double> stretched(3, 0.0);
    const robo::ManipulabilityReport singular =
        arm->analyseManipulability(stretched, planarRows);
    check(singular.nearSingular, "stretched arm is reported as near-singular");
    check(singular.yoshikawaIndex < 1e-6,
          "Yoshikawa index collapses at the singularity");

    // A folded configuration is well away from the boundary.
    std::vector<double> folded(3);
    folded[0] = 0.0; folded[1] = kPi / 2.0; folded[2] = -kPi / 2.0;
    const robo::ManipulabilityReport healthy =
        arm->analyseManipulability(folded, planarRows);
    check(!healthy.nearSingular, "folded arm is not near-singular");
    check(healthy.yoshikawaIndex > singular.yoshikawaIndex,
          "folded arm has a larger manipulability index than the stretched one");

    delete arm;
}

void testInverseKinematicsRoundTrip() {
    robo::SerialChain* arm = makePlanar3R();

    // Pick a reachable target by running forward kinematics on a known pose.
    std::vector<double> truth(3);
    truth[0] = 0.5; truth[1] = -0.8; truth[2] = 0.4;
    const robo::Matrix target = arm->position(truth);

    robo::IkOptions options;
    options.positionTolerance = 1e-8;
    options.maxIterations = 500;
    options.damping = 0.01;
    robo::IkSolver solver(*arm, options);

    std::vector<double> seed(3, 0.1);
    const robo::IkResult result = solver.solvePosition(target, seed);

    check(result.converged, "IK converges on a reachable target");
    check(result.finalError < 1e-7, "IK residual is below tolerance");

    // The arm is redundant in the plane, so the joint values need not match
    // the ones used to build the target; the reached position must.
    const robo::Matrix reached = arm->position(result.solution);
    check((reached - target).norm() < 1e-7,
          "IK solution reproduces the target position");

    delete arm;
}

void testInverseKinematicsUnreachableTarget() {
    robo::SerialChain* arm = makePlanar3R();

    robo::Matrix target(3, 1, 0.0);
    target(0, 0) = 10.0;  // far outside the reach of three unit links

    robo::IkOptions options;
    options.maxIterations = 100;
    robo::IkSolver solver(*arm, options);

    std::vector<double> seed(3, 0.2);
    const robo::IkResult result = solver.solvePosition(target, seed);

    check(!result.converged, "IK reports failure on an unreachable target");
    check(result.finalError > 6.0,
          "residual reflects how far out of reach the target is");

    delete arm;
}

void testJointLimitsAreRespected() {
    robo::SerialChain chain("limited");
    chain.addJoint(new robo::RevoluteJoint(
        robo::DHParameters(1.0, 0.0, 0.0, 0.0), -0.1, 0.1, "tight"));
    chain.addJoint(new robo::RevoluteJoint(
        robo::DHParameters(1.0, 0.0, 0.0, 0.0), -0.1, 0.1, "tight2"));

    std::vector<double> q(2);
    q[0] = 5.0; q[1] = -5.0;
    check(!chain.withinLimits(q), "out-of-range configuration is detected");

    chain.clampToLimits(q);
    checkNear(q[0], 0.1, 1e-12, "upper limit is applied");
    checkNear(q[1], -0.1, 1e-12, "lower limit is applied");
    check(chain.withinLimits(q), "clamped configuration is within limits");
}

void testDofMismatchThrows() {
    robo::SerialChain* arm = makePlanar3R();
    bool threw = false;
    try {
        std::vector<double> wrong(2, 0.0);
        arm->position(wrong);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "wrong number of joint values throws");
    delete arm;
}

// -------------------------------------------------------------- Trajectory

void testCubicProfileEndpoints() {
    robo::CubicProfile p(1.0, 4.0, 2.0);

    checkNear(p.at(0.0).position, 1.0, 1e-12, "cubic starts at the start value");
    checkNear(p.at(2.0).position, 4.0, 1e-12, "cubic ends at the end value");
    checkNear(p.at(0.0).velocity, 0.0, 1e-12, "cubic starts at rest");
    checkNear(p.at(2.0).velocity, 0.0, 1e-12, "cubic ends at rest");
    checkNear(p.at(1.0).position, 2.5, 1e-12, "cubic is symmetric at mid-time");
    checkNear(p.at(1.0).velocity, p.peakVelocity(), 1e-12,
              "peak velocity occurs at mid-time");
}

void testQuinticProfileIsSmoother() {
    robo::QuinticProfile q(0.0, 1.0, 1.0);

    checkNear(q.at(0.0).acceleration, 0.0, 1e-12,
              "quintic starts with zero acceleration");
    checkNear(q.at(1.0).acceleration, 0.0, 1e-12,
              "quintic ends with zero acceleration");

    robo::CubicProfile c(0.0, 1.0, 1.0);
    check(std::fabs(c.at(0.0).acceleration) > 1e-6,
          "cubic has a non-zero acceleration jump at the start");
    check(q.peakVelocity() > c.peakVelocity(),
          "smoothness costs a higher peak velocity");
}

void testTrapezoidalReachesTargetAndCruises() {
    robo::TrapezoidalProfile t(0.0, 10.0, /*cruise=*/2.0, /*accel=*/1.0);

    check(!t.isTriangular(), "long move reaches cruise velocity");
    checkNear(t.at(0.0).position, 0.0, 1e-9, "trapezoid starts at the start value");
    checkNear(t.at(t.duration()).position, 10.0, 1e-9,
              "trapezoid ends exactly at the target");
    checkNear(t.peakVelocity(), 2.0, 1e-12, "cruise velocity is honoured");

    // Mid-cruise the velocity should be flat at vMax.
    const double tMid = t.accelerationTime() + 0.5 * t.cruiseTime();
    checkNear(t.at(tMid).velocity, 2.0, 1e-9, "velocity is flat during cruise");
    checkNear(t.at(tMid).acceleration, 0.0, 1e-12, "no acceleration during cruise");
}

void testTrapezoidalDegeneratesToTriangle() {
    // Too short to ever reach the requested cruise velocity.
    robo::TrapezoidalProfile t(0.0, 0.1, /*cruise=*/5.0, /*accel=*/1.0);

    check(t.isTriangular(), "short move degenerates to a triangular profile");
    checkNear(t.cruiseTime(), 0.0, 1e-12, "triangular profile has no cruise phase");
    check(t.peakVelocity() < 5.0, "peak velocity is reduced to what fits");
    checkNear(t.at(t.duration()).position, 0.1, 1e-9,
              "triangular profile still lands on the target");
}

void testTrapezoidalBackwardsMove() {
    robo::TrapezoidalProfile t(3.0, -2.0, 1.0, 0.5);
    checkNear(t.at(0.0).position, 3.0, 1e-9, "negative move starts correctly");
    checkNear(t.at(t.duration()).position, -2.0, 1e-9, "negative move lands correctly");
    check(t.at(t.duration() * 0.5).velocity < 0.0,
          "velocity is negative while moving backwards");
}

void testProfilePolymorphism() {
    // The point of the abstract base: one loop, three different profiles.
    std::vector<robo::MotionProfile*> profiles;
    profiles.push_back(new robo::CubicProfile(0.0, 1.0, 2.0));
    profiles.push_back(new robo::QuinticProfile(0.0, -0.5, 2.0));
    profiles.push_back(new robo::TrapezoidalProfile(0.0, 2.0, 2.0, 2.0));

    robo::JointTrajectory traj(profiles);  // takes ownership
    check(traj.dof() == 3, "trajectory carries one profile per joint");

    const std::vector<double> atStart = traj.positionAt(0.0);
    checkNear(atStart[0], 0.0, 1e-9, "all joints start at their start value");
    checkNear(atStart[1], 0.0, 1e-9, "all joints start at their start value");

    const std::vector<double> atEnd = traj.positionAt(traj.duration());
    checkNear(atEnd[0], 1.0, 1e-9, "cubic joint lands");
    checkNear(atEnd[1], -0.5, 1e-9, "quintic joint lands");
    checkNear(atEnd[2], 2.0, 1e-9, "trapezoidal joint lands");
}

void testSamplingIsMonotonicInTime() {
    robo::QuinticProfile q(0.0, 1.0, 1.0);
    const std::vector<robo::TrajectoryPoint> pts = q.sample(50);

    check(pts.size() == 50, "sample() returns the requested count");
    bool monotonic = true;
    for (std::size_t i = 1; i < pts.size(); ++i) {
        if (pts[i].time <= pts[i - 1].time) {
            monotonic = false;
        }
        if (pts[i].position < pts[i - 1].position - 1e-12) {
            monotonic = false;  // a rest-to-rest quintic never backs up
        }
    }
    check(monotonic, "samples advance in time and position");
}

struct TestCase {
    const char* name;
    void (*fn)();
};

}  // namespace

int main() {
    TestCase tests[] = {
        {"matrix: arithmetic", testMatrixArithmetic},
        {"matrix: inverse and solve", testMatrixInverseAndSolve},
        {"matrix: singular detection", testMatrixSingularDetection},
        {"matrix: symmetric eigen decomposition", testSymmetricEigen},
        {"kinematics: forward vs closed form", testForwardKinematicsAgainstClosedForm},
        {"kinematics: analytic vs numeric Jacobian", testAnalyticJacobianMatchesNumeric},
        {"kinematics: prismatic joint column", testPrismaticJointJacobian},
        {"kinematics: manipulability at singularity", testManipulabilityDropsAtSingularity},
        {"kinematics: IK round trip", testInverseKinematicsRoundTrip},
        {"kinematics: IK unreachable target", testInverseKinematicsUnreachableTarget},
        {"kinematics: joint limits", testJointLimitsAreRespected},
        {"kinematics: DOF mismatch throws", testDofMismatchThrows},
        {"trajectory: cubic endpoints", testCubicProfileEndpoints},
        {"trajectory: quintic smoothness", testQuinticProfileIsSmoother},
        {"trajectory: trapezoidal cruise", testTrapezoidalReachesTargetAndCruises},
        {"trajectory: triangular degeneration", testTrapezoidalDegeneratesToTriangle},
        {"trajectory: backwards move", testTrapezoidalBackwardsMove},
        {"trajectory: polymorphic profiles", testProfilePolymorphism},
        {"trajectory: sampling", testSamplingIsMonotonicInTime},
    };

    const std::size_t count = sizeof(tests) / sizeof(tests[0]);
    for (std::size_t i = 0; i < count; ++i) {
        const int before = g_failures;
        tests[i].fn();
        std::printf("%s  %s\n", g_failures == before ? "ok  " : "FAIL", tests[i].name);
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
