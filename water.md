# Polarizability of water

[CCCBDB](https://cccbdb.nist.gov/exp2x.asp) says that the experimental electric
dipole polarizability of water is 1.501 Å³, which I converted to 10.129 Bohr³.

The calculation in this directory uses HF/cc-pVDZ, which is also available in
[CCCBDB](https://cccbdb.nist.gov/polcalc3x.asp?method=1&basis=5)
|  ɑ (Å³) |   x   |   y   |   z   |
| ------- | ----- | ----- | ----- |
|    x    | 0.451 | 0.000 | 0.000 |
|    y    | 0.000 | 0.992 | 0.000 |
|    z    | 0.000 | 0.000 | 0.738 |

which I can translate to 
|  ɑ (au³) |   x   |   y   |   z   |
| -------- | ----- | ----- | ----- |
|     x    | 3.043 | 0.000 | 0.000 |
|     y    | 0.000 | 6.694 | 0.000 |
|     z    | 0.000 | 0.000 | 4.980 |

## CCSD

### STO-3G
|  ɑ (au³) |   x  |   y  |  z   |
| -------- | ---- | ---- | ---- |
|     x    | 0.06 | 0.00 | 0.00 |
|     y    | 0.00 | 6.69 | 0.00 |
|     z    | 0.00 | 0.00 | 2.79 |

### 3-21G
|  ɑ (au³) |   x  |   y  |  z   |
| -------- | ---- | ---- | ---- |
|     x    | 0.66 | 0.00 | 0.00 |
|     y    | 0.00 | 9.53 | 0.00 |
|     z    | 0.00 | 0.00 | 5.04 |

### 3-21G*
|  ɑ (au³) |   x  |   y  |  z   |
| -------- | ---- | ---- | ---- |
|     x    | 0.66 | 0.00 | 0.00 |
|     y    | 0.00 | 9.53 | 0.00 |
|     z    | 0.00 | 0.00 | 5.04 |

### 6-31G
|  ɑ (au³) |   x  |   y  |  z   |
| -------- | ---- | ---- | ---- |
|     x    | 1.26 | 0.00 | 0.00 |
|     y    | 0.00 | 9.82 | 0.00 |
|     z    | 0.00 | 0.00 | 5.94 |

### 6-31G**
|  ɑ (au³) |   x  |   y   |  z   |
| -------- | ---- | ----- | ---- |
|     x    | 2.95 |  0.00 | 0.00 |
|     y    | 0.00 | 10.03 | 0.00 |
|     z    | 0.00 |  0.00 | 6.50 |

### 6-31+G**
|  ɑ (au³) |   x  |   y   |  z   |
| -------- | ---- | ----- | ---- |
|     x    | 6.52 |  0.00 | 0.00 |
|     y    | 0.00 | 10.21 | 0.00 |
|     z    | 0.00 |  0.00 | 7.66 |

### 6-311G*
|  ɑ (au³) |   x  |   y   |  z   |
| -------- | ---- | ----- | ---- |
|     x    | 2.92 |  0.00 | 0.00 |
|     y    | 0.00 |  9.99 | 0.00 |
|     z    | 0.00 |  0.00 | 6.47 |

### 6-311G*
|  ɑ (au³) |   x  |   y   |  z   |
| -------- | ---- | ----- | ---- |
|     x    | 3.77 |  0.00 | 0.00 |
|     y    | 0.00 | 10.30 | 0.00 |
|     z    | 0.00 |  0.00 | 6.93 |

### 6-311G*
|  ɑ (au³) |   x  |   y   |  z   |
| -------- | ---- | ----- | ---- |
|     x    | 8.26 |  0.00 | 0.00 |
|     y    | 0.00 | 11.62 | 0.00 |
|     z    | 0.00 |  0.00 | 9.42 |


# References 

## CCCBDB entries
| Reference  | DOI  | Squib |
| --         | --   | ---   |
| Landolt-Bornstein Vol 1 part 3 p509 (1951)  |  | 1951LB1.3:509 |
| DE Woon, TH Dunning Jr "Gaussian basis sets for use in correlated molecular
calculations. IV. Calculation of static electrical properties" J. Chem. Phys.
100(4) 2975, 1994  | 10.1063/1.466439  | 1994Woo/Dun:2975 |
| TN Olney, NM Cann, G Cooper, CE Brion, Absolute scale determination for
photoabsorption spectra and the calculation of molecular properties using dipole
sum-rules, Chem. Phys. 223 (1997) 59-98  | 10.1016/S0301-0104(97)00145-6  |
1997Oln/Can:59 |
| M Gussoni, R Rui, G Zerbi "Electronic and relaxation contribution to linear
molecular polarizability. An analysis of the experimental values" J. Mol.
Struct. 447 (1998) 163-215  | 10.1016/S0022-2860(97)00292-5  | 1998Gus/Rui:163 |

## Experimental polarizability 
10.1016/S0301-0104(97)00145-6  TN Olney, NM Cann, G Cooper, CE Brion, Absolute
scale determination for photoabsorption spectra and the calculation of molecular
properties using dipole sum-rules, Chem. Phys. 223 (1997) 59-98
alpha = 10.13 Bohr³

## 
NIST Computational Chemistry Comparison and Benchmark Database,
NIST Standard Reference Database Number 101
Release 22, May 2022, Editor: Russell D. Johnson III
http://cccbdb.nist.gov/
