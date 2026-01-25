
============================================================
 Maxwell-PINN-NIF Quick Test Suite
============================================================
  Python: 3.10.12
  PyTorch: 2.5.1+cu121
  CUDA Available: True

============================================================
 1. GPU/Device Check
============================================================
  Device: NVIDIA GeForce RTX 3090 Ti
  Memory: 25.28 GB
  ✓ PASS: GPU Detection
         NVIDIA GeForce RTX 3090 Ti, 25.3GB

============================================================
 2. Model Initialization
============================================================
  Architecture: 3 → 128×6 → (3, 3)
  Parameters: 83,846
  ✓ PASS: Model Creation
         83,846 parameters

============================================================
 3. Forward Pass Test
============================================================
  Input:  coords [1000, 3]
  Output: E [1000, 3], H [1000, 3]
  Time:   21.97 ms
  ✓ PASS: Forward Pass Shape
         E: [1000, 3], H: [1000, 3]
  ✓ PASS: Forward Pass Speed
         21.97 ms for 1000 points

============================================================
 4. Divergence-Free Constraint Test
============================================================
  ε values: constant 2.0
  μ values: constant 1.0
  Divergence loss: 1.016568e-04
  ✓ PASS: Divergence Loss Computation
         L_div = 1.0166e-04

============================================================
 5. Maxwell Curl Equation Residual Test
============================================================
  ω (angular frequency): 1.0
  Curl residual loss: 1.407436e-04
  ✓ PASS: Curl Residual Computation
         L_curl = 1.4074e-04

============================================================
 6. PML (Absorbing Boundary) Loss Test
============================================================
  Domain: [0,1]³
  PML thickness: 0.1
  Points in PML: 334/1000
  PML loss: 4.161752e-05
  ✓ PASS: PML Loss Computation
         L_pml = 4.1618e-05

============================================================
 7. Backward Pass (Gradient Flow) Test
============================================================
  Total loss: 2.840180e-04
  Gradients exist: True
  Gradients non-zero: True
  Gradients finite: True
  ✓ PASS: Gradient Computation
         All params have valid gradients

============================================================
 8. GPU Memory Usage Test
============================================================
  Allocated: 0.018 GB
  Reserved:  0.109 GB
  Peak:      0.102 GB
  ✓ PASS: Memory Usage
         Peak: 0.10 GB

============================================================
 9. Single Training Step Test
============================================================
  Loss before: 2.457620e-04
  Loss after:  7.280216e-05
  ✓ PASS: Training Step Execution
         2.46e-04 → 7.28e-05

============================================================
 10. Short Training Convergence Test
============================================================
  Running 50 training iterations...
  Initial loss (avg first 5):  6.8424e-04
  Final loss (avg last 5):     3.7028e-06
  Reduction: 99.5%
  ✓ PASS: Training Convergence
         Reduced by 99.5%

============================================================
 TEST SUMMARY
============================================================
  Passed: 11/11

  🎉 ALL TESTS PASSED!
