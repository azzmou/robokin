# robokin

Kinematics, singularity analysis and trajectory generation for serial
manipulators. **C++98, no external dependencies.**

Written against the classical manipulator theory — Denavit-Hartenberg
convention, geometric Jacobian, damped least squares, polynomial and
trapezoidal motion profiles — with the numerics implemented rather than
imported, so each step can be read and checked.

```
git clone https://github.com/azzmou/robokin.git
cd robokin
cmake -S . -B build && cmake --build build
./build/test_robo      # 76 checks
./build/puma560_demo
```

## Why C++98 and no dependencies

Robot controllers and embedded targets often sit on old toolchains, and
pulling Eigen or Boost onto them is not always an option. Everything here
compiles with a 1998-era compiler and the standard library alone:
`-std=c++98 -Wall -Wextra -Wpedantic`, zero warnings.

The cost is that the linear algebra had to be written by hand. That is also
the point: the Gauss-Jordan inverse, the LU determinant and the Jacobi
eigenvalue solver are all in `src/Matrix.cpp`, roughly 150 lines, readable
end to end.

## What it does

### Forward kinematics

Joints are an abstract base with revolute and prismatic subclasses. Each
subclass knows which DH parameter its variable drives and how to form its own
Jacobian column, so `SerialChain` never branches on a joint-type tag — adding
a new joint type means adding a class, not editing a switch.

```cpp
robo::SerialChain arm("planar-3R");
arm.addJoint(new robo::RevoluteJoint(robo::DHParameters(1.0, 0.0, 0.0, 0.0),
                                     -M_PI, M_PI, "q1"));
// ... two more links

std::vector<double> q(3, 0.0);
q[0] = 0.3; q[1] = -0.7; q[2] = 1.1;

robo::Matrix pose = arm.forwardKinematics(q);   // 4x4
robo::Matrix tip  = arm.position(q);            // 3x1
```

### Jacobian, checked two ways

`jacobian()` builds the geometric Jacobian analytically in a single pass down
the chain. `numericJacobian()` computes the same thing by central differences,
including the angular rows via the skew-symmetric part of `dR/dq · Rᵀ`.

The two are independent implementations, so the test suite checks one against
the other. That is the difference between code that looks right and code that
is right.

### Singularity and manipulability analysis

The singular values of the task Jacobian come from a Jacobi eigenvalue
decomposition of `J Jᵀ`. From them:

- **Yoshikawa index** `w = √det(J Jᵀ)` — the volume of the manipulability
  ellipsoid
- **condition number** `σ_max / σ_min` — how badly the arm is distorted
- **minimum singular value** — distance to the nearest singularity
- **principal axes** — the directions the arm moves easily or badly

Task dimension is explicit. A planar arm has a structurally empty z direction,
so analysing it against all three linear rows would report it as permanently
singular. Pass the rows the task actually spans:

```cpp
std::vector<std::size_t> planarRows;   // x and y only
planarRows.push_back(0);
planarRows.push_back(1);

robo::ManipulabilityReport r = arm.analyseManipulability(q, planarRows);
if (r.nearSingular) { /* back off */ }
```

Running the PUMA 560 demo through an elbow sweep, the boundary singularity
shows up exactly where the theory puts it:

```
  elbow[deg]   w=sqrt(det(J J^T))   sigma_min    cond(J)
         0            0.11169       0.26321       2.63
        45            0.03592       0.25548       1.93
        75            0.00284       0.04865       9.31
        90            0.00013       0.00271     164.55   <- arm straight
       120            0.01196       0.09602       4.60
```

And the wrist singularity, where roll and yaw collapse onto the same axis:

```
  wrist pitch =  0 deg : |det(J_angular)| = 0.000000
  wrist pitch = 30 deg : |det(J_angular)| = 0.500000
```

### Inverse kinematics

Position-only IK by damped least squares:

```
Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
```

The damping bounds the step near singularities, where a plain pseudo-inverse
would demand enormous joint velocities. The system is solved rather than
inverted — cheaper and better conditioned. Steps are capped and joint limits
applied each iteration.

The trade-off is visible in the demo:

```
  lambda =  0.00 | converged in   18 iterations | residual = 8.58e-10
  lambda =  0.01 | converged in   21 iterations | residual = 8.49e-11
  lambda =  0.20 | converged in  363 iterations | residual = 9.54e-10
```

Undamped is fastest on a well-conditioned arm and is exactly what blows up
near a singularity. Heavy damping always converges but drags its feet.

### Trajectory generation

`MotionProfile` is an abstract scalar profile; three implementations:

| profile | endpoint conditions | peak velocity | notes |
|---|---|---|---|
| `CubicProfile` | position, velocity | `1.5·d/T` | acceleration jumps at the ends |
| `QuinticProfile` | position, velocity, acceleration | `1.875·d/T` | finite jerk, costs speed |
| `TrapezoidalProfile` | bounded velocity and acceleration | set by the caller | degenerates to a triangle on short moves |

`JointTrajectory` drives a whole chain with one profile per joint, all
starting and finishing together.

```cpp
std::vector<robo::MotionProfile*> profiles;
for (std::size_t i = 0; i < arm.dof(); ++i) {
    profiles.push_back(new robo::QuinticProfile(from[i], to[i], 2.0));
}
robo::JointTrajectory traj(profiles);     // takes ownership
std::vector<double> q = traj.positionAt(0.75);
```

## Layout

```
include/robo/Matrix.h        Matrix, SymmetricEigen (Jacobi)
include/robo/Kinematics.h    DHParameters, Joint hierarchy, SerialChain, IkSolver
include/robo/Trajectory.h    MotionProfile hierarchy, JointTrajectory
src/                         implementations
tests/test_robo.cpp          76 checks, no test framework needed
examples/puma560_demo.cpp    full PUMA 560 walkthrough
```

## Design notes

**Ownership is explicit.** `SerialChain` and `JointTrajectory` take ownership
of what they are given and delete it. With no `unique_ptr` available in C++98,
copy constructors and assignment operators are declared private and left
undefined, so a copy is a link error rather than a double free.

**Errors throw.** Shape mismatches, singular matrices, out-of-range indices
and bad parameters raise standard exceptions with messages that name the
actual shapes. Silent wrong answers are worse than crashes.

**Numerical care.** Partial pivoting in the inverse, determinant and solver.
Negative eigenvalues from round-off are clamped before the square root.
`Matrix::solve` is preferred over forming an explicit inverse.

## Limitations

- Position-only IK. Full 6-DOF pose IK needs the orientation error as a
  rotation-vector residual; the Jacobian already provides the angular rows.
- No dynamics: no mass, inertia or torque.
- No closed-form IK for specific geometries. A spherical-wrist PUMA has an
  analytic solution that is faster and finds all eight branches; the iterative
  solver here finds one, and which one depends on the seed.
- Dense matrices with no expression templates. Fine at 6 DOF, wrong choice
  at 100.

## Licence

MIT. See `LICENSE`.
