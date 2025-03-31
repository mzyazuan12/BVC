
print("""\
Mohammed Tariq       UIN: 435003728
ENGR/PHYS 216 – Section 437     HW Assignment 7: Collisions
Date: March 18, 2025           Page 1 of 3
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

Problem 1: Two-Ball Elastic Collision

Given:
• Ball 1: mass \\( m_1 = 5.00\\,\\text{kg} \\); initial velocity \\( v_{1i} = 2.55\\,\\text{m/s} \\).
• Ball 2: mass \\( m_2 = 3.61\\,\\text{kg} \\); initial velocity \\( v_{2i} = 1.55\\,\\text{m/s} \\).
• The balls are “super hard” (no deformation) and roll on a smooth surface.

Find:
(a) The final velocity of ball 1, \\( v_{1f} \\) (m/s).
(b) The final velocity of ball 2, \\( v_{2f} \\) (m/s).

Diagram:
(Sketch two circles along a horizontal line. Draw arrows from left to right: ball 1 (larger arrow, faster) behind ball 2 (smaller arrow). After collision, show new arrows for each ball indicating their final speeds.)

Theory:
For a one-dimensional elastic collision, both momentum and kinetic energy are conserved. The formulas are:
\\[
v_{1f} = \\frac{m_1 - m_2}{m_1 + m_2}\\,v_{1i} + \\frac{2m_2}{m_1 + m_2}\\,v_{2i} \\quad , \\quad
v_{2f} = \\frac{2m_1}{m_1 + m_2}\\,v_{1i} + \\frac{m_2 - m_1}{m_1 + m_2}\\,v_{2i}\\,.
\\]

Assumptions:
• Collision is strictly one‑dimensional.
• No external forces act on the balls during the collision.

Solution:
1. Total mass:
   \\( m_1 + m_2 = 5.00 + 3.61 = 8.61\\,\\text{kg} \\).

2. For ball 1:
   \\[
   v_{1f} = \\frac{5.00 - 3.61}{8.61}(2.55) + \\frac{2 \\times 3.61}{8.61}(1.55).
   \\]
   Calculate:
   – \\( \\frac{5.00 - 3.61}{8.61} = \\frac{1.39}{8.61} \\approx 0.1615 \\);
     \\( 0.1615 \\times 2.55 \\approx 0.412\\,\\text{m/s} \\).
   – \\( \\frac{2 \\times 3.61}{8.61} = \\frac{7.22}{8.61} \\approx 0.838 \\);
     \\( 0.838 \\times 1.55 \\approx 1.299\\,\\text{m/s} \\).
   Thus,
   \\[
   v_{1f} \\approx 0.412 + 1.299 = 1.711\\,\\text{m/s}.
   \\]

3. For ball 2:
   \\[
   v_{2f} = \\frac{2 \\times 5.00}{8.61}(2.55) + \\frac{3.61 - 5.00}{8.61}(1.55).
   \\]
   Calculate:
   – \\( \\frac{2 \\times 5.00}{8.61} = \\frac{10.00}{8.61} \\approx 1.161 \\);
     \\( 1.161 \\times 2.55 \\approx 2.96\\,\\text{m/s} \\).
   – \\( \\frac{3.61 - 5.00}{8.61} = \\frac{-1.39}{8.61} \\approx -0.1615 \\);
     \\( -0.1615 \\times 1.55 \\approx -0.250\\,\\text{m/s} \\).
   Thus,
   \\[
   v_{2f} \\approx 2.96 - 0.250 = 2.71\\,\\text{m/s}.
   \\]

Final Answers:
(a) \\( v_{1f} \\approx \\boxed{1.71\\,\\text{m/s}} \\)
(b) \\( v_{2f} \\approx \\boxed{2.71\\,\\text{m/s}} \\)

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
Page 2 of 3

Problem 2: Villain and Chuck Norris’ Fist Collision

Given:
• Villain: mass \\( m_v = 65.0\\,\\text{kg} \\); initial velocity \\( v_{v,i} = 9.26\\,\\text{m/s} \\) (to the right).
• Chuck Norris’ fist: initial velocity \\( v_{f,i} = 0.150\\,\\text{m/s} \\) (assumed leftward).
• After collision:
   – Villain’s final speed: \\( v_{v,f} = 8.52\\,\\text{m/s} \\) at \\(20.0^\\circ\\) above the horizontal.
   – Fist’s final speed: \\( v_{f,f} = 0.0745\\,\\text{m/s} \\) at \\(10.0^\\circ\\) below the horizontal.

Find:
(a) The mass of Chuck Norris’ fist, \\( m_f \\) (kg).
(b) The kinetic energy lost in the collision (J).

Diagram:
(Sketch a diagram showing the villain and the fist with initial directions. After impact, draw the villain’s trajectory at \\(20^\\circ\\) upward and the fist’s trajectory at \\(10^\\circ\\) downward. Indicate horizontal and vertical components to illustrate momentum conservation.)

Theory:
• Vertical momentum conservation: Since initial vertical momentum is zero,
  \\[
  m_v\\,v_{v,f}\\sin20^\\circ + m_f\\,v_{f,f}\\sin10^\\circ = 0.
  \\]
  Solve for \\( m_f \\):
  \\[
  m_f = \\frac{m_v\\,v_{v,f}\\sin20^\\circ}{v_{f,f}\\sin10^\\circ}.
  \\]

• Kinetic energy calculation:
  \\( KE = \\frac{1}{2} m v^2 \\).
  Determine the total KE before and after collision and then find the loss:
  \\[
  \\Delta KE = KE_i - KE_f.
  \\]

Assumptions:
• Choose the positive y‑direction for the villain’s upward motion and negative for the fist’s downward motion so that their vertical momenta cancel.
• Only momentum is conserved; kinetic energy is not conserved in this inelastic (or “silly”) collision.

Solution (a):
1. Use trigonometric values:
   \\( \\sin20^\\circ \\approx 0.3420 \\), \\( \\sin10^\\circ \\approx 0.17365 \\).
2. Compute vertical components:
   \\( v_{v,f,y} = 8.52 \\times 0.3420 \\approx 2.915\\,\\text{m/s} \\).
   \\( v_{f,f,y} = 0.0745 \\times 0.17365 \\approx 0.01294\\,\\text{m/s} \\).
3. Calculate \\( m_f \\):
   \\[
   m_f = \\frac{65.0 \\times 2.915}{0.01294} \\approx \\frac{189.48}{0.01294} \\approx 14642\\,\\text{kg}.
   \\]

Solution (b):
1. Initial Kinetic Energy:
   – Villain:
     \\( KE_{v,i} = \\frac{1}{2}\\,65.0\\,(9.26^2) \\).
     \\( 9.26^2 \\approx 85.75 \\) \\(\\rightarrow\\) \\( KE_{v,i} \\approx 32.5 \\times 85.75 \\approx 2784\\,\\text{J} \\).
   – Fist:
     \\( KE_{f,i} = \\frac{1}{2}\\,14642\\,(0.150^2) \\).
     \\( 0.150^2 = 0.0225 \\) \\(\\rightarrow\\) \\( KE_{f,i} \\approx 0.5 \\times 14642 \\times 0.0225 \\approx 164.7\\,\\text{J} \\).
   Total \\( KE_i \\approx 2784 + 164.7 = 2948.7\\,\\text{J} \\).

2. Final Kinetic Energy:
   – Villain:
     \\( KE_{v,f} = \\frac{1}{2}\\,65.0\\,(8.52^2) \\).
     \\( 8.52^2 \\approx 72.59 \\) \\(\\rightarrow\\) \\( KE_{v,f} \\approx 32.5 \\times 72.59 \\approx 2359.7\\,\\text{J} \\).
   – Fist:
     \\( KE_{f,f} = \\frac{1}{2}\\,14642\\,(0.0745^2) \\).
     \\( 0.0745^2 \\approx 0.00555 \\) \\(\\rightarrow\\) \\( KE_{f,f} \\approx 0.5 \\times 14642 \\times 0.00555 \\approx 40.6\\,\\text{J} \\).
   Total \\( KE_f \\approx 2359.7 + 40.6 = 2400.3\\,\\text{J} \\).

3. Energy Lost:
   \\[
   \\Delta KE \\approx 2948.7 - 2400.3 \\approx 548.4\\,\\text{J}.
   \\]

Final Answers:
(a) \\( m_f \\approx \\boxed{1.46\\times10^4\\,\\text{kg}} \\)
(b) \\( \\Delta KE \\approx \\boxed{548\\,\\text{J}} \\)

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
Page 3 of 3

Problem 3: Inelastic Collision (Vin and Zane)

Given:
• Vin: mass \\( m_1 \\) (unknown), initial velocity \\( v_1 = 9.660\\,\\text{m/s} \\) (rightward).
• Zane: mass \\( m_2 = 53.8\\,\\text{kg} \\); initial velocity \\( v_2 = 7.26\\,\\text{m/s} \\) (since they fly at each other, assign Zane’s direction as leftward, i.e. \\( v_2 = -7.26\\,\\text{m/s} \\)).
• After collision, they stick together and move with velocity \\( v_3 = 5.164\\,\\text{m/s} \\) (to the right).

Find:
Vin’s mass \\( m_1 \\) (kg).

Diagram:
(Draw two objects: one labeled “Vin” moving to the right and one labeled “Zane” moving to the left. After collision, draw a single object moving to the right.)

Theory:
Use conservation of linear momentum:
\\[
m_1\\,v_1 + m_2\\,v_2 = (m_1 + m_2)\\,v_3.
\\]

Assumptions:
• One-dimensional motion; right is positive and left is negative.
• The collision is perfectly inelastic (they stick together).

Solution:
1. Write the momentum equation:
   \\[
   m_1(9.660) + 53.8(-7.26) = (m_1 + 53.8)(5.164).
   \\]
2. Compute the known term for Zane’s momentum:
   \\( 53.8 \\times 7.26 \\approx 390.39\\,\\text{kg\\cdot m/s} \\).
   Thus,
   \\[
   9.660\\,m_1 - 390.39 = 5.164\\,m_1 + 53.8(5.164).
   \\]
3. Compute \\( 53.8 \\times 5.164 \\approx 277.75\\,\\text{kg\\cdot m/s} \\).
   The equation becomes:
   \\[
   9.660\\,m_1 - 390.39 = 5.164\\,m_1 + 277.75.
   \\]
4. Rearranging:
   \\[
   9.660\\,m_1 - 5.164\\,m_1 = 390.39 + 277.75 \\quad \\Longrightarrow \\quad 4.496\\,m_1 = 668.14.
   \\]
5. Solve for \\( m_1 \\):
   \\[
   m_1 \\approx \\frac{668.14}{4.496} \\approx 148.67\\,\\text{kg}.
   \\]

Final Answer:
\\( m_1 \\approx \\boxed{148.7\\,\\text{kg}} \\).

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

Be sure to include a neat sketch (or diagram) with your submission and verify that all calculations and units are clearly shown. Adjust rounding as necessary.
""")
