Theory: the critical time step of an explicit analysis
======================================================

An explicit analysis (``NED``, :doc:`/analysis/solvers`) solves no system of equations: it divides the net force by a
lumped mass and steps on. The price is that it is stable only below a time increment, the *critical time step*. This
page explains where that limit comes from, why EdelweissFE estimates it instead of computing it, how the estimates of
the parts of a model are combined, what penalty constraints add to them, and when each estimate is taken anew. The code is
:mod:`~edelweissfe.solvers.base.criticaltimestep`.

Where the limit comes from
--------------------------

Linearize the equation of motion about the current state, :math:`\boldsymbol{M}\ddot{\boldsymbol{u}} +
\boldsymbol{C}\dot{\boldsymbol{u}} + \boldsymbol{K}\boldsymbol{u} = \boldsymbol{0}`, with the lumped (diagonal) mass
:math:`\boldsymbol{M}`, damping :math:`\boldsymbol{C}` and the tangent stiffness :math:`\boldsymbol{K}`. In the
eigenvectors of :math:`\boldsymbol{M}^{-1}\boldsymbol{K}` it falls apart into single oscillators
:math:`\ddot{q} + \omega^2 q = 0`. The central difference method applied to one of them,

.. math::

   q_{n+1} - 2 q_n + q_{n-1} = -\omega^2 \Delta t^2 q_n ,

has the amplification factors :math:`z` with :math:`z^2 - (2 - \omega^2\Delta t^2) z + 1 = 0`. Their product is one;
they stay on the unit circle -- the oscillation neither grows nor decays -- as long as :math:`|2 - \omega^2 \Delta t^2|
\le 2`, i.e. :math:`\omega \Delta t \le 2`. Beyond, one of them is real and larger than one, and that mode grows
without bound. The whole system is stable if its highest mode is:

.. math::

   \Delta t \le \Delta t_\mathrm{crit} = \frac{2}{\omega_\mathrm{max}}, \qquad
   \omega_\mathrm{max}^2 = \lambda_\mathrm{max}(\boldsymbol{M}^{-1}\boldsymbol{K}) .

The damping does not lower this limit in ``NED``, which takes the damping force with the mean of the velocities before
and after the update. A field with a damping but no inertia (a first-order nonlocal field) is integrated by forward
Euler, :math:`\boldsymbol{C}\dot{\boldsymbol{u}} + \boldsymbol{K}\boldsymbol{u} = \boldsymbol{0}`, stable for
:math:`\Delta t \le 2/\lambda_\mathrm{max}(\boldsymbol{C}^{-1}\boldsymbol{K})`.

Why bounds, not eigenvalues
---------------------------

:math:`\omega_\mathrm{max}` of the assembled system is one eigenvalue of a matrix the explicit method otherwise never
assembles. Computing it -- by a few power or Lanczos iterations on the assembled tangent -- costs more than many
increments, and it changes as the state changes. What the method needs is not the eigenvalue but a time increment that
is *safe*: a lower bound of :math:`\Delta t_\mathrm{crit}`, from an upper bound of :math:`\omega_\mathrm{max}`, at a
cost small against an increment. Two such bounds are used:

* **The element eigenvalue inequality.** The highest frequency of an assembled mesh is not larger than the highest
  frequency of its elements, each taken alone with its own lumped mass:
  :math:`\omega_\mathrm{max}^2 \le \max_e \lambda_\mathrm{max}(\boldsymbol{M}_e^{-1}\boldsymbol{K}_e)` (the Rayleigh
  quotient of the mesh is a weighted mean of the elements' quotients). An element's eigenvalue problem is small, and
  Marmot solves it with the element's own strain operator and the largest wave speed of its material, which holds for
  every element order and distortion -- where the classic "element length over wave speed" is exact only for a linear
  bar.
* **Row sums.** The particles of a meshfree model (RKPM) have no such inequality: the stiffness of a particle couples
  nodes at which its own lumped mass vanishes. Every eigenvalue :math:`\lambda` of a matrix is bounded by its largest
  absolute row sum (Gershgorin), and more sharply by the Collatz-Wielandt bound of its absolute value with positive
  weights. EdelweissMeshfree applies it to the mass-scaled stiffness :math:`\boldsymbol{M}^{-1/2}\boldsymbol{K}
  \boldsymbol{M}^{-1/2}` with weights per field, and Marmot bounds the row sums of every particle without assembling
  its stiffness (EdelweissMeshfree, ``MPMDiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia``).

Both are bounds from above for :math:`\omega_\mathrm{max}`, so the time increment they give is safe; how far below the
true limit it lies is the price of not computing the eigenvalue.

How the parts combine
---------------------

A model may hold finite elements and particles. Their degrees of freedom are disjoint -- ``NED`` refuses a node that
both would carry -- so the matrix :math:`\boldsymbol{M}^{-1}\boldsymbol{K}` of the model is block diagonal in the two
parts, its eigenvalues are those of the parts, and the critical time step of the model is the **smallest** of the
parts':

.. math::

   \Delta t = \text{courant-number} \times \min\left(\Delta t_\mathrm{crit}^\mathrm{elements},\,
   \Delta t_\mathrm{crit}^\mathrm{particles}\right) .

``courant-number`` is the one safety factor: the fraction of the estimated limit the analysis runs at. With penalty
constraints, the critical time step of the degrees of freedom they couple joins the minimum (below).

The particles' estimate has one factor of its own, below one, which is not a safety factor but part of what it
estimates: their stabilization (NSNI) acts on the increment, so their explicit scheme is not exactly the central
difference method with their tangent, and it was measured to be stable up to a fraction of the tangent's limit
(EdelweissMeshfree, ``STABLE_FRACTION_OF_THE_TANGENT_LIMIT``).

Penalty constraints
-------------------

A penalty constraint -- a contact, a support held by a penalty, a tie -- is a set of springs: a point :math:`p` of it
with the spring stiffness :math:`k_p` and the gradient :math:`\boldsymbol{w}_p` of its gap adds
:math:`k_p\,\boldsymbol{w}_p\boldsymbol{w}_p^T` to the stiffness of the nodes it couples. For a contact point,
:math:`\boldsymbol{w}_p = [c, -N^m] \otimes \boldsymbol{n}` (the interpolation vector of the point on the slave side,
the shape functions of the master face, the normal), and :math:`k_p = \mathrm{d}f_n/\mathrm{d}g` is what the contact
law gives the point at its gap: the penalty times the area of the point for the linear law, zero while the gap is open
(:mod:`~edelweissfe.constraints.base.penaltylaw`). The gap is the one the law sees, measured from an adjusted initial
overlap (:doc:`/theory/contact`, "An initial overlap"): a point at rest on the adjusted surface is open. A tie holds every point with one such spring along every axis
:math:`\boldsymbol{e}_i`, :math:`\boldsymbol{w}_p = [c, -N^m] \otimes \boldsymbol{e}_i` and :math:`k_p` the penalty
times the area, open or closed (:doc:`/theory/contact`, "A tie"). The force and the estimate go through the same springs
of the law, so a friction law later gives its own stiffness, and the estimate follows without a change.

Such a constraint owns no degrees of freedom: it couples those of the parts. Taking the minimum with the parts would be
wrong -- the spring *adds* to the frequencies of the nodes it couples. A mass :math:`m` on a spring :math:`k_1` (an
element) and held by a penalty :math:`k_2` oscillates with :math:`\omega^2 = (k_1 + k_2)/m`, not with the larger of
the two. Adding the constraint's largest frequency to the parts' largest one would be safe but blunt: a stiff contact
far from the smallest element would lower the time increment of the whole model.

**The estimate: the constraint adds where it acts.** For every degree of freedom :math:`i`, let :math:`\omega_i^2` be
the bound of its part there -- for the elements the largest :math:`\lambda_e = (2/\Delta t_e)^2` of the elements at
:math:`i`, for the particles the row sum of their bound -- and let

.. math::

   r_i = \frac{1}{\sqrt{M_i}} \sum_j |K^\mathrm{c}_{ij}| \frac{1}{\sqrt{M_j}}

be the row sum of the absolute value of the constraints' stiffness, scaled by the lumped mass on both sides (every
constraint gives it: ``addBoundOfAbsoluteStiffnessTimes``). Then

.. math::

   \omega_\mathrm{max}^2 \le \Lambda = \max_i \left(\omega_i^2 + r_i\right) .

At a degree of freedom no constraint couples, :math:`r_i = 0` and the part's estimate stands; a constraint lowers the
critical time step only if the frequency it adds at its nodes, on top of the part's there, exceeds the largest one of
the model. For a mass :math:`m` held by a penalty :math:`k` alone, :math:`r = k/m` and the estimate is the exact
:math:`2\sqrt{m/k}`; for two equal masses joined by a penalty, :math:`r = 2k/m`, again exact.

**Why it is safe: lend the constraint a part of the mass.** The argument of the element eigenvalue inequality holds for
*any* split of the lumped mass among the pieces of the stiffness: if :math:`\boldsymbol{K} = \sum_a
\boldsymbol{K}_a` and :math:`\boldsymbol{M} = \sum_a \boldsymbol{M}_a` (all diagonal, nonnegative), then

.. math::

   \boldsymbol{x}^T\boldsymbol{K}\boldsymbol{x} = \sum_a \boldsymbol{x}^T\boldsymbol{K}_a\boldsymbol{x}
   \le \sum_a \lambda_a\, \boldsymbol{x}^T\boldsymbol{M}_a\boldsymbol{x}
   \le \max_a \lambda_a \; \boldsymbol{x}^T\boldsymbol{M}\boldsymbol{x} ,
   \qquad \lambda_a = \lambda_\mathrm{max}(\boldsymbol{M}_a^{-1}\boldsymbol{K}_a) .

The constraint has no mass of its own, so it is lent the fraction :math:`\theta_i = r_i/\Lambda` of the mass of every
degree of freedom it couples, and the parts keep the rest, :math:`(1 - \theta_i) M_i`.

* The constraint with the masses :math:`\theta_i M_i`: by the row sums with the weights :math:`\sqrt{\theta_i}`
  (Collatz-Wielandt), its largest eigenvalue is at most :math:`\max_i r_i/\theta_i = \Lambda`.
* An element with the masses :math:`(1-\theta_i)` times its own: its eigenvalue grows at most by :math:`1/(1 -
  \max_{i \in e}\theta_i)`, and :math:`\lambda_e \le \omega_i^2 \le \Lambda - r_i = \Lambda (1 - \theta_i)`
  at each of its nodes, so it stays below :math:`\Lambda`.
* The particles with the masses :math:`(1-\theta_i)M_i`: by their row sums with the weights scaled by
  :math:`\sqrt{1-\theta_i}`, their row at :math:`i` grows by :math:`1/(1-\theta_i)`, and stays below :math:`\Lambda`
  for the same reason.

Every piece is below :math:`\Lambda` with its share of the mass, so the whole is. The argument is only as safe as each
:math:`\lambda_e` is a bound of its element: the estimate is each element's own, and the one of the Python elements
lies above their exact critical time step (EdelweissFE#323). The argument is the Rayleigh quotient's, which needs a
symmetric stiffness: the penalty springs are (a closed point with a frozen projection), and the elements' estimate
bounds their elastic stiffness. The tangent of gradient-enhanced particles is not symmetric;
for them, the constraint's rows are added to the particles' rows inside their field matrix
(EdelweissMeshfree, ``CriticalTimeStepBoundOfEachFieldBlock``), and the bound of the field matrix holds for any matrix.
Where a constraint couples particles and finite elements at once, the two arguments are combined; that the
unsymmetric particles take part in the Rayleigh quotient of the elements is then an assumption.

**With multi-point constraints** (a ``tie`` of a fine mesh to a coarse one, hanging nodes), a slave moves with its
masters, :math:`u_s = \sum_a N_a u_{m_a}`, and the system that oscillates is the condensed one,
:math:`\boldsymbol{T}^T \boldsymbol{K} \boldsymbol{T}` with the slave masses folded onto the masters. A spring at a
slave acts on its masters: its row sums are taken with the weights of the masters and folded onto them by
:math:`|\boldsymbol{T}|`. So does an element at a slave: a master takes the largest :math:`\lambda_e` of the elements
at it *and at its slaves*. With coefficients that are nonnegative and sum to one, as an interpolation has, an element
at a slave keeps its own bound with the shares of mass it lends the masters (Jensen's inequality,
:math:`(\sum_a N_a x_a)^2 \le \sum_a N_a x_a^2`), and the argument above holds for the condensed system. Without the
second fold, a stiff element at a slave would be missing at its master just where a penalty there adds the most.

What the bound gives away is the Gershgorin step: the row sum is the exact frequency for springs between equal masses
and above it otherwise, and a normal oblique to the axes spreads the spring over the components of a node.

**What is not estimated.** The stiffness the constraints add enters from the state they have after an increment, for
the next one: a point that closes *within* an increment adds its stiffness one increment late. An increment above the
limit amplifies an oscillation by a bounded factor, and the time increment is never raised again within the step, so
this is not an instability; the energy balance of ``NED`` (the kinetic energy may not exceed the energy at the start of
the step plus the external work) remains the check. Not estimated either: the geometric stiffness of a contact with
finite sliding, or of a rigid body whose normals rotate with it (of the order of the normal force over the size of a
facet or the lever arm, small against the penalty), the coupling of the normal and the tangential force of a sliding
frictional contact (the friction coefficient times the normal stiffness; unsymmetric), and a penalty on a field
integrated by forward Euler.

**One formula for every penalty contact.** Each contact gives its stiffness as springs: a spring :math:`p` with the
gradient :math:`\boldsymbol{w}_p` of its stretch and the stiffness :math:`k_p` of its law -- the normal spring of a
closed point, a tie's spring along each axis, the tangential spring of a sticking frictional contact. The row sums are
then :math:`b_a \mathrel{+}= \sum_p |k_p|\, |w_{p,a}|\, (|\boldsymbol{w}_p| \cdot \boldsymbol{w})`
(:mod:`edelweissfe.constraints.base.penaltysprings`), and each contact takes the springs for its bound from the same
code it takes them for its force, so that a change of the force cannot leave the bound behind.

When an estimate is taken anew
------------------------------

The true critical time step changes as the analysis runs: elements are refined, particles move and deform, materials
stiffen and soften. Estimating in every increment would always be right, but an estimate of all particles costs a
noticeable part of an increment. So each part comes with a *policy*, decided by what its estimate depends on.

**The finite elements: once per mesh**
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfTheElements`). Their estimate uses the elastic
wave speed of the material, not its state. A softening material only raises the true limit, so the estimate stays safe
until the mesh changes; it is taken anew after a refinement and at the start of every step.

**The particles: when the margin may be used up**
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfEntitiesWithOwnInertia`). Their estimate depends
on their state: on their lumped masses, which change as they move over the nodes of the kernel functions, and on their
stiffness, which changes with the material and with the deformation of the particles. After an estimate :math:`v`,
``NED`` keeps in every increment a lower bound of what the estimate can be now, :math:`n` increments later:

.. math::

   L = v \, \min\left(1, \sqrt{\alpha}, \beta\right) (1 - \delta)^n ,

and takes a new estimate when :math:`\text{courant-number} \times L` falls below the time increment in use.

* :math:`\alpha` is the largest relative fall of a lumped mass since the estimate, :math:`\beta` the same for the
  damping of a forward-Euler field. This factor is **exact**. The estimate scales the stiffness with
  :math:`\boldsymbol{M}^{-1/2}` on both sides; if every mass is at least :math:`\alpha` times what it was, every scaled
  row sum, and with them the bound of :math:`\omega_\mathrm{max}^2`, is at most :math:`1/\alpha` times larger (the
  largest eigenvalue of a nonnegative matrix grows with its entries), so the estimate is at least :math:`\sqrt{\alpha}`
  times what it was. The masses cost nothing to watch: ``NED`` assembles them in every
  increment anyway. A node that gets a mass for the first time (a particle reaches a kernel function's node, whose mass
  grows from zero) is a new row whose scaling has no bound: then the particles are estimated anew at once.
* :math:`\delta`, the option ``critical-time-step-decrease-per-increment``, is the largest relative fall per increment
  the stiffness may cause. This factor is an **assumption**. No theorem bounds it: the tangent of a material can change
  abruptly from one increment to the next. What makes it reasonable is what an increment is: a fraction of the time a
  wave needs to cross a particle, so in one increment the configuration moves by a fraction of a particle, and the
  stiffness, which follows the configuration, changes by a fraction of itself. Its default was measured: in the
  impact of a steel projectile on a concrete slab of gradient-enhanced particles (GM7 and GCDP), the fall of the
  particles' estimate not explained by their masses stayed below half the default in every single increment, and far
  below it over several increments. There, the fall came from particles at the rim of the crater that are stretched
  far: the nonlocal field of a gradient-enhanced particle is balanced with gradients in the reference configuration,
  which grow with the stretch, while its micro-inertia does not -- a smooth change from increment to increment, which
  only an estimate sees, not the masses. A larger :math:`\delta` is more cautious and estimates more often; 1 estimates
  in every increment.

Where penalty constraints couple the particles, the same lower bound is kept for the particles' rows with the
constraints added: the particles' stiffness may have grown by :math:`1/f^2`, :math:`f = \min(1, \sqrt{\alpha},
\beta)(1 - \delta)^n`, the constraints' rows are those of now. The particles are estimated anew also when that bound
falls below the time increment in use -- the margin a contact at the particles' nodes leaves is smaller than their own.

What the policy does in practice follows from the margin. Where the particles set the time increment, the increment in
use *is* courant-number times their last estimate, the margin is used up after one increment, and they are estimated in
every increment -- as they must be, since their estimate is what limits the analysis. Where another part sets it, the
estimate of the particles is above the increment in use by some factor :math:`m`, and it is taken anew only every
:math:`\ln m / -\ln(1-\delta)` increments, unless their masses fall faster than expected.

**The penalty constraints: in every increment**
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfThePenaltyConstraints`). Their stiffness follows
the state of the contact -- which points are closed, and how deep, for a law whose stiffness grows with the
penetration --, and changes from increment to increment; their row sums cost about as much as their force, so they are
formed in every increment, from the state after it, and added to the parts' last estimates. Only this addition is
renewed; the parts keep their own policies.

The code
--------

.. automodule:: edelweissfe.solvers.base.criticaltimestep
   :members:
