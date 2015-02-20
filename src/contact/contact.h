#ifndef _CPY_CONTACT_H
#define _CPY_CONTACT_H

void atomic_contact(const double *xyzlist, const int *contacts, int num_contacts, int traj_length, int num_atoms, double *results);
void atomic_displacement(const double *xyzlist, const int *contacts, int num_contacts, int traj_length, int num_atoms, double *results_dr, double *results_dx);
void closest_contact(const double *xyzlist, const int *residues, const int num_residues, const int residue_width, const int* atoms_per_residue,  const int *contacts, int num_contacts, int traj_length, int num_atoms, double *results);

#endif
