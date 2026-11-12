// SPDX-License-Identifier: MIT
// C++98. No external dependencies.
#ifndef ROBO_TRAJECTORY_H
#define ROBO_TRAJECTORY_H

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace robo {

/// One sampled point of a scalar trajectory.
struct TrajectoryPoint {
    double time;
    double position;
    double velocity;
    double acceleration;

    TrajectoryPoint()
        : time(0.0), position(0.0), velocity(0.0), acceleration(0.0) {}
    TrajectoryPoint(double t, double p, double v, double a)
        : time(t), position(p), velocity(v), acceleration(a) {}
};

/// Abstract scalar motion profile between two positions over a fixed duration.
///
/// Every profile answers the same question — where should the joint be at
/// time t — so the planner can swap profiles without knowing which one it has.
class MotionProfile {
public:
    MotionProfile(double start, double end, double duration);
    virtual ~MotionProfile();

    virtual TrajectoryPoint at(double t) const = 0;

    /// Human-readable profile name, for logs and plots.
    virtual const char* kind() const = 0;

    /// Largest absolute velocity reached over the whole profile.
    virtual double peakVelocity() const = 0;

    double start() const { return start_; }
    double end() const { return end_; }
    double duration() const { return duration_; }

    /// Uniformly sample the profile. Useful for feeding a controller or
    /// writing a CSV for plotting.
    std::vector<TrajectoryPoint> sample(std::size_t samples) const;

protected:
    /// Clamp t into [0, duration] so callers cannot read off the ends.
    double clampTime(double t) const;

    double start_;
    double end_;
    double duration_;
};

/// Cubic polynomial: matches both endpoints and zero velocity at each end.
/// Acceleration is discontinuous at the endpoints, which a stiff joint will
/// feel as a jolt.
class CubicProfile : public MotionProfile {
public:
    CubicProfile(double start, double end, double duration);
    virtual TrajectoryPoint at(double t) const;
    virtual const char* kind() const { return "cubic"; }
    virtual double peakVelocity() const;

private:
    double a0_, a2_, a3_;
};

/// Quintic polynomial: also matches zero acceleration at each end, so the
/// jerk stays finite. The cost is a higher peak velocity for the same move.
class QuinticProfile : public MotionProfile {
public:
    QuinticProfile(double start, double end, double duration);
    virtual TrajectoryPoint at(double t) const;
    virtual const char* kind() const { return "quintic"; }
    virtual double peakVelocity() const;

private:
    double a0_, a3_, a4_, a5_;
};

/// Trapezoidal velocity profile: accelerate, cruise, decelerate. Reaches the
/// target in less time than a polynomial of the same peak velocity, which is
/// why industrial controllers default to it.
///
/// When the move is too short to reach cruise speed the profile degenerates
/// to a triangle, and the constructor adjusts the cruise velocity instead of
/// failing.
class TrapezoidalProfile : public MotionProfile {
public:
    /// @param cruiseVelocity maximum velocity magnitude [units/s]
    /// @param acceleration   magnitude of the ramp [units/s^2]
    TrapezoidalProfile(double start, double end, double cruiseVelocity,
                       double acceleration);

    virtual TrajectoryPoint at(double t) const;
    virtual const char* kind() const { return "trapezoidal"; }
    virtual double peakVelocity() const { return vMax_; }

    bool isTriangular() const { return triangular_; }
    double accelerationTime() const { return tAccel_; }
    double cruiseTime() const { return tCruise_; }

private:
    double vMax_;
    double accel_;
    double tAccel_;
    double tCruise_;
    double sign_;
    bool triangular_;
};

/// Drives every joint of a chain from one configuration to another, with all
/// joints starting and finishing together (synchronised motion).
class JointTrajectory {
public:
    /// Takes ownership of the profiles, one per joint.
    explicit JointTrajectory(const std::vector<MotionProfile*>& profiles);
    ~JointTrajectory();

    std::size_t dof() const { return profiles_.size(); }
    double duration() const;

    /// Joint positions at time t.
    std::vector<double> positionAt(double t) const;

    /// Joint velocities at time t.
    std::vector<double> velocityAt(double t) const;

private:
    JointTrajectory(const JointTrajectory&);
    JointTrajectory& operator=(const JointTrajectory&);

    std::vector<MotionProfile*> profiles_;
};

}  // namespace robo

#endif  // ROBO_TRAJECTORY_H
