Site switch (mock-up): blend each atom between the MTP and an NLH screened-Coulomb pair potential
====================================================================================================

  E_i = w_i * V_i + (nlh_fixed ? (1 - w_i) * eps_i : 0)

  V_i    plain MTP site energy (any radial basis)
  eps_i  1/2 * sum_j phi(r_ij),  phi(r) = 14.39964548 * Z1*Z2 / r * sum_k a_k exp(-b_k r)   [eV, A]
  w_i    1 - smootherstep(t_i);  w = 1: MTP fully on,  w = 0: the atom is a pure NLH atom

Optional block in the .almtp, immediately before the radial_basis_type line. Without the block the
potential behaves exactly as before.

  site_switch = nlh_energy        criterion: q_i = (sum_j phi(r_ij)^p)^(1/p), p = softmax_p (1 = plain sum)
  	switch_on = 3.4               eV: MTP fully on for q <= switch_on   (3.4 eV  ~ one Ge-Ge pair at 1.70 A)
  	switch_off = 29.8             eV: MTP fully off for q >= switch_off (29.8 eV ~ one Ge-Ge pair at 1.25 A)
  	softmax_p = 8
  	switch_scale = log            interpolate in ln(q) (log) or in q (linear)
  	nlh_fixed = true              true: train directly on RAW DFT data; mlp train subtracts the fixed NLH
  	                              part internally. false: only w_i * V_i (externally subtracted data / overlay)
  	nlh_pairs = 1
  		0-0 {32, 32, 0.05121, 49.82156, 0.50355, 12.56998, 0.44524, 4.13684}
                                  species pair (internal indices) {Z1, Z2, a1, b1, a2, b2, a3, b3}

  site_switch = distance          criterion: w_i = prod_j smootherstep((r_ij - switch_off)/(switch_on - switch_off))
  	switch_on = 1.7               A: MTP fully on when every neighbour is farther than this
  	switch_off = 1.25             A: MTP fully off when any neighbour is closer than this
  	nlh_fixed = true              (nlh_pairs is only needed when nlh_fixed = true)
  	nlh_pairs = 1
  		0-0 {32, 32, 0.05121, 49.82156, 0.50355, 12.56998, 0.44524, 4.13684}

Usage: mlp train 08_nlh_energy.almtp raw_train.cfg --save_to=pot.almtp
       mlp calculate_efs / check_errors with the trained potential return TOTAL energies (MTP + NLH).

Not covered yet: LAMMPS interface, prune/mask tools (no internal subtraction there), magnetic MTPs,
active-learning selection is untested, MPI is implemented in the trainer hook but untested.
phi is not shifted to zero at the MTP cutoff (jump ~1e-6 eV for Ge at 5 A, only for atoms with w < 1).
