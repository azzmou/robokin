// SPDX-License-Identifier: MIT
// C++98.
//
// Builds a PUMA 560 from its standard DH table, then exercises the library:
// forward kinematics, Jacobian conditioning across a sweep, a wrist
// singularity, inverse kinematics, and a synchronised joint trajectory.
//
//     ./puma560_demo

#include "robo/Kinematics.h"
#include "robo/Matrix.h"
#include "robo/Trajectory.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

const double kPi = 3.14159265358979323846;

double deg(double radians) { return radians * 180.0 / kPi; }
double rad(double degrees) { return degrees * kPi / 180.0; }

/// PUMA 560, standard DH parameters (Craig / Corke), lengths in metres.
robo::SerialChain* makePuma560() {
    robo::SerialChain* puma = new robo::SerialChain("PUMA 560");

    const double d2 = 0.2435;
    const double d3 = -0.0934;
    const double d4 = 0.4331;
    const double a2 = 0.4318;
    const double a3 = -0.0203;

    //                                  a    alpha      d     theta
    puma->addJoint(new robo::RevoluteJoint(
        robo::DHParameters(0.0,  kPi / 2.0,  0.0,   0.0), rad(-160), rad(160), "waist"));
    puma->addJoint(new robo::RevoluteJoint(
        robo::DHParameters(a2,   0.0,        d2,    0.0), rad(-225), rad(45),  "shoulder"));
    puma->addJoint(new robo::RevoluteJoint(
        robo::DHParameters(a3,  -kPi / 2.0,  d3,    0.0), rad(-45),  rad(225), "elbow"));
    puma->addJoint(new robo::RevoluteJoint(
        robo::DHParameters(0.0,  kPi / 2.0,  d4,    0.0), rad(-110), rad(170), "wrist-roll"));
    puma->addJoint(new robo::RevoluteJoint(
        robo::DHParameters(0.0, -kPi / 2.0,  0.0,   0.0), rad(-100), rad(100), "wrist-pitch"));
    puma->addJoint(new robo::RevoluteJoint(
        robo::DHParameters(0.0,  0.0,        0.0,   0.0), rad(-266), rad(266), "wrist-yaw"));

    return puma;
}

void printHeading(const char* title) {
    std::cout << "\n== " << title << " "
              << std::string(60 - std::string(title).size(), '=') << "\n\n";
}

void showForwardKinematics(const robo::SerialChain& puma) {
    printHeading("Forward kinematics");

    std::vector<double> q(6, 0.0);
    std::cout << "Zero configuration, base -> tool transform:\n"
              << puma.forwardKinematics(q);

    q[1] = rad(-45.0);
    q[2] = rad(90.0);
    q[4] = rad(45.0);
    const robo::Matrix p = puma.position(q);
    std::cout << "\nReady pose q = (0, -45, 90, 0, 45, 0) deg\n";
    std::cout << std::fixed << std::setprecision(4)
              << "  tool position: x=" << p(0, 0)
              << "  y=" << p(1, 0)
              << "  z=" << p(2, 0) << " [m]\n";
}

void showManipulabilitySweep(const robo::SerialChain& puma) {
    printHeading("Manipulability across an elbow sweep");

    std::cout << "  elbow[deg]   w=sqrt(det(J J^T))   sigma_min    cond(J)\n";
    std::cout << "  ---------------------------------------------------------\n";

    for (int elbowDeg = 0; elbowDeg <= 150; elbowDeg += 15) {
        std::vector<double> q(6, 0.0);
        q[1] = rad(-45.0);
        q[2] = rad(static_cast<double>(elbowDeg));

        const robo::ManipulabilityReport r = puma.analyseManipulability(q);

        std::cout << std::fixed << std::setprecision(5)
                  << "  " << std::setw(8) << elbowDeg
                  << "   " << std::setw(16) << r.yoshikawaIndex
                  << "   " << std::setw(9) << r.minimumSingularValue
                  << "   ";
        if (r.nearSingular) {
            std::cout << "  singular";
        } else {
            std::cout << std::setw(8) << std::setprecision(2) << r.conditionNumber;
        }
        std::cout << "\n";
    }

    std::cout << "\n  The index collapses as the elbow straightens: that is the\n"
                 "  boundary singularity, where the arm loses the ability to move\n"
                 "  radially outwards.\n";
}

void showWristSingularity(const robo::SerialChain& puma) {
    printHeading("Wrist singularity");

    // Wrist pitch at zero aligns the roll and yaw axes: the classic PUMA
    // wrist singularity, where two joints fight over the same rotation.
    std::vector<double> aligned(6, 0.0);
    aligned[1] = rad(-45.0);
    aligned[2] = rad(45.0);
    aligned[4] = 0.0;

    std::vector<double> offset(aligned);
    offset[4] = rad(30.0);

    const robo::Matrix jAligned = puma.jacobian(aligned);
    const robo::Matrix jOffset = puma.jacobian(offset);

    std::cout << std::fixed << std::setprecision(6)
              << "  wrist pitch =  0 deg : |det(J_angular)| = "
              << std::fabs(jAligned.block(3, 3, 3, 3).determinant()) << "\n"
              << "  wrist pitch = 30 deg : |det(J_angular)| = "
              << std::fabs(jOffset.block(3, 3, 3, 3).determinant()) << "\n";
    std::cout << "\n  With the pitch at zero the three wrist axes are no longer\n"
                 "  independent and the angular block loses rank.\n";
}

void showInverseKinematics(const robo::SerialChain& puma) {
    printHeading("Inverse kinematics (damped least squares)");

    std::vector<double> truth(6, 0.0);
    truth[0] = rad(25.0);
    truth[1] = rad(-60.0);
    truth[2] = rad(100.0);
    truth[3] = rad(10.0);
    truth[4] = rad(30.0);

    const robo::Matrix target = puma.position(truth);
    std::cout << std::fixed << std::setprecision(4)
              << "  target position: x=" << target(0, 0)
              << "  y=" << target(1, 0)
              << "  z=" << target(2, 0) << "\n\n";

    const double dampings[] = {0.0, 0.01, 0.2};
    for (int i = 0; i < 3; ++i) {
        robo::IkOptions opts;
        opts.damping = dampings[i];
        opts.positionTolerance = 1e-9;
        opts.maxIterations = 400;

        robo::IkSolver solver(puma, opts);
        std::vector<double> seed(6, 0.05);

        const robo::IkResult res = solver.solvePosition(target, seed);
        std::cout << "  lambda = " << std::setw(5) << std::setprecision(2)
                  << dampings[i]
                  << " | " << (res.converged ? "converged" : "  failed ")
                  << " in " << std::setw(4) << res.iterations << " iterations"
                  << " | residual = " << std::scientific << std::setprecision(2)
                  << res.finalError << std::fixed << "\n";
    }

    std::cout << "\n  Undamped iteration is fastest when the arm is well\n"
                 "  conditioned, and is exactly what blows up near a singularity.\n"
                 "  Heavy damping always converges but drags its feet.\n";
}

void showTrajectory(const robo::SerialChain& puma) {
    printHeading("Synchronised joint trajectory");

    std::vector<double> from(6, 0.0);
    std::vector<double> to(6, 0.0);
    to[0] = rad(60.0);
    to[1] = rad(-30.0);
    to[2] = rad(80.0);

    const double duration = 2.0;
    std::vector<robo::MotionProfile*> profiles;
    for (std::size_t i = 0; i < puma.dof(); ++i) {
        profiles.push_back(new robo::QuinticProfile(from[i], to[i], duration));
    }
    robo::JointTrajectory traj(profiles);

    std::cout << "   t[s]   q1[deg]   q2[deg]   q3[deg]    tool x    tool y    tool z\n";
    std::cout << "  ---------------------------------------------------------------\n";

    for (int step = 0; step <= 8; ++step) {
        const double t = duration * static_cast<double>(step) / 8.0;
        const std::vector<double> q = traj.positionAt(t);
        const robo::Matrix p = puma.position(q);

        std::cout << std::fixed << std::setprecision(2)
                  << "  " << std::setw(5) << t
                  << std::setprecision(2)
                  << "  " << std::setw(8) << deg(q[0])
                  << "  " << std::setw(8) << deg(q[1])
                  << "  " << std::setw(8) << deg(q[2])
                  << std::setprecision(4)
                  << "  " << std::setw(8) << p(0, 0)
                  << "  " << std::setw(8) << p(1, 0)
                  << "  " << std::setw(8) << p(2, 0) << "\n";
    }

    std::cout << "\n  Every joint starts and stops together, so the tool path is\n"
                 "  repeatable even though it is not a straight line in space.\n";
}

}  // namespace

int main() {
    robo::SerialChain* puma = makePuma560();

    std::cout << "robokin - " << puma->name() << ", "
              << puma->dof() << " revolute joints\n";

    showForwardKinematics(*puma);
    showManipulabilitySweep(*puma);
    showWristSingularity(*puma);
    showInverseKinematics(*puma);
    showTrajectory(*puma);

    std::cout << "\n";
    delete puma;
    return 0;
}
