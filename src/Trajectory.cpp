// SPDX-License-Identifier: MIT
// C++98.
#include "robo/Trajectory.h"

#include <cmath>

namespace robo {

// ----------------------------------------------------------- MotionProfile

MotionProfile::MotionProfile(double start, double end, double duration)
    : start_(start), end_(end), duration_(duration) {
    if (duration <= 0.0) {
        throw std::invalid_argument("MotionProfile: duration must be > 0");
    }
}

MotionProfile::~MotionProfile() {}

double MotionProfile::clampTime(double t) const {
    if (t < 0.0) {
        return 0.0;
    }
    if (t > duration_) {
        return duration_;
    }
    return t;
}

std::vector<TrajectoryPoint> MotionProfile::sample(std::size_t samples) const {
    if (samples < 2) {
        throw std::invalid_argument("MotionProfile::sample: need at least 2 samples");
    }
    std::vector<TrajectoryPoint> out;
    out.reserve(samples);
    const double dt = duration_ / static_cast<double>(samples - 1);
    for (std::size_t i = 0; i < samples; ++i) {
        out.push_back(at(static_cast<double>(i) * dt));
    }
    return out;
}

// ------------------------------------------------------------ CubicProfile

CubicProfile::CubicProfile(double start, double end, double duration)
    : MotionProfile(start, end, duration) {
    // q(t) = a0 + a2 t^2 + a3 t^3, with q(0)=start, q(T)=end, qd(0)=qd(T)=0.
    const double d = end - start;
    const double T = duration;
    a0_ = start;
    a2_ = 3.0 * d / (T * T);
    a3_ = -2.0 * d / (T * T * T);
}

TrajectoryPoint CubicProfile::at(double t) const {
    const double tc = clampTime(t);
    const double t2 = tc * tc;
    const double p = a0_ + a2_ * t2 + a3_ * t2 * tc;
    const double v = 2.0 * a2_ * tc + 3.0 * a3_ * t2;
    const double a = 2.0 * a2_ + 6.0 * a3_ * tc;
    return TrajectoryPoint(tc, p, v, a);
}

double CubicProfile::peakVelocity() const {
    // Velocity is a downward parabola peaking at T/2: v_max = 3 d / (2 T).
    return std::fabs(1.5 * (end_ - start_) / duration_);
}

// ---------------------------------------------------------- QuinticProfile

QuinticProfile::QuinticProfile(double start, double end, double duration)
    : MotionProfile(start, end, duration) {
    // Six boundary conditions: position, velocity and acceleration at both
    // ends. With zero velocity and acceleration at the ends the coefficients
    // reduce to the closed form below.
    const double d = end - start;
    const double T = duration;
    const double T3 = T * T * T;
    a0_ = start;
    a3_ = 10.0 * d / T3;
    a4_ = -15.0 * d / (T3 * T);
    a5_ = 6.0 * d / (T3 * T * T);
}

TrajectoryPoint QuinticProfile::at(double t) const {
    const double tc = clampTime(t);
    const double t2 = tc * tc;
    const double t3 = t2 * tc;
    const double t4 = t3 * tc;
    const double t5 = t4 * tc;

    const double p = a0_ + a3_ * t3 + a4_ * t4 + a5_ * t5;
    const double v = 3.0 * a3_ * t2 + 4.0 * a4_ * t3 + 5.0 * a5_ * t4;
    const double a = 6.0 * a3_ * tc + 12.0 * a4_ * t2 + 20.0 * a5_ * t3;
    return TrajectoryPoint(tc, p, v, a);
}

double QuinticProfile::peakVelocity() const {
    // Peak at T/2: v_max = 15 d / (8 T).
    return std::fabs(1.875 * (end_ - start_) / duration_);
}

// ------------------------------------------------------ TrapezoidalProfile

namespace {

/// Total duration of a trapezoid, needed before the base class is built.
double trapezoidDuration(double distance, double vMax, double accel) {
    const double d = std::fabs(distance);
    if (d == 0.0) {
        return 1e-9;  // degenerate but positive, so the base class accepts it
    }
    const double dAccel = vMax * vMax / accel;  // distance of ramp up + down
    if (d < dAccel) {
        // Triangular: never reaches vMax.
        return 2.0 * std::sqrt(d / accel);
    }
    const double tRamp = vMax / accel;
    const double tCruise = (d - dAccel) / vMax;
    return 2.0 * tRamp + tCruise;
}

}  // namespace

TrapezoidalProfile::TrapezoidalProfile(double start, double end,
                                       double cruiseVelocity, double acceleration)
    : MotionProfile(start, end,
                    trapezoidDuration(end - start, cruiseVelocity, acceleration)),
      accel_(acceleration),
      triangular_(false) {
    if (cruiseVelocity <= 0.0) {
        throw std::invalid_argument("TrapezoidalProfile: cruiseVelocity must be > 0");
    }
    if (acceleration <= 0.0) {
        throw std::invalid_argument("TrapezoidalProfile: acceleration must be > 0");
    }

    const double distance = end - start;
    sign_ = distance >= 0.0 ? 1.0 : -1.0;
    const double d = std::fabs(distance);

    const double dAccel = cruiseVelocity * cruiseVelocity / acceleration;
    if (d < dAccel) {
        // Short move: lower the cruise velocity to the peak actually reached.
        triangular_ = true;
        vMax_ = std::sqrt(d * acceleration);
        tAccel_ = (vMax_ > 0.0) ? vMax_ / acceleration : 0.0;
        tCruise_ = 0.0;
    } else {
        vMax_ = cruiseVelocity;
        tAccel_ = vMax_ / acceleration;
        tCruise_ = (d - dAccel) / vMax_;
    }
}

TrajectoryPoint TrapezoidalProfile::at(double t) const {
    const double tc = clampTime(t);
    double p = 0.0;
    double v = 0.0;
    double a = 0.0;

    if (tc < tAccel_) {
        a = accel_;
        v = accel_ * tc;
        p = 0.5 * accel_ * tc * tc;
    } else if (tc < tAccel_ + tCruise_) {
        const double dt = tc - tAccel_;
        a = 0.0;
        v = vMax_;
        p = 0.5 * vMax_ * tAccel_ + vMax_ * dt;
    } else {
        const double dt = tc - tAccel_ - tCruise_;
        const double tDecel = tAccel_;
        const double clamped = dt > tDecel ? tDecel : dt;
        a = -accel_;
        v = vMax_ - accel_ * clamped;
        p = 0.5 * vMax_ * tAccel_ + vMax_ * tCruise_ +
            vMax_ * clamped - 0.5 * accel_ * clamped * clamped;
    }

    return TrajectoryPoint(tc, start_ + sign_ * p, sign_ * v, sign_ * a);
}

// ---------------------------------------------------------- JointTrajectory

JointTrajectory::JointTrajectory(const std::vector<MotionProfile*>& profiles)
    : profiles_(profiles) {
    for (std::size_t i = 0; i < profiles_.size(); ++i) {
        if (profiles_[i] == 0) {
            throw std::invalid_argument("JointTrajectory: null profile");
        }
    }
}

JointTrajectory::~JointTrajectory() {
    for (std::size_t i = 0; i < profiles_.size(); ++i) {
        delete profiles_[i];
    }
}

double JointTrajectory::duration() const {
    double longest = 0.0;
    for (std::size_t i = 0; i < profiles_.size(); ++i) {
        if (profiles_[i]->duration() > longest) {
            longest = profiles_[i]->duration();
        }
    }
    return longest;
}

std::vector<double> JointTrajectory::positionAt(double t) const {
    std::vector<double> out(profiles_.size(), 0.0);
    for (std::size_t i = 0; i < profiles_.size(); ++i) {
        out[i] = profiles_[i]->at(t).position;
    }
    return out;
}

std::vector<double> JointTrajectory::velocityAt(double t) const {
    std::vector<double> out(profiles_.size(), 0.0);
    for (std::size_t i = 0; i < profiles_.size(); ++i) {
        out[i] = profiles_[i]->at(t).velocity;
    }
    return out;
}

}  // namespace robo
