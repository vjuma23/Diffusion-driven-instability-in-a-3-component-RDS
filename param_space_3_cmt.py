"""
Diffusion-Driven Instability (DDI) Parameter Space Generator
Created by: Victor Juma (vjuma23@gmail.com)
Updated: February 14, 2026

DESCRIPTION:
This script identifies and visualizes parameter spaces for Diffusion-Driven 
Instability in three-component reaction-diffusion systems. It maps regions 
where a steady state is stable without diffusion but becomes unstable with 
diffusion, resulting in pattern formation.

HOW TO RUN:
1. Ensure dependencies are installed: pip install numpy matplotlib pandas
2. Run from the terminal specifying the polymer model:
   python param_space_3_cmt.py --polymer 4
3. Optional: Adjust resolution using the --res flag (e.g., --res 100 for fast testing).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def discriminant_f(x6, x4, x2, x0):
    """
    Discriminant of the cubic x6*t^3 + x4*t^2 + x2*t + x0 
    """
    inner = np.maximum(x4**2 - 3*x2*x6, 0)
    return (27*(x6**2)*(x0) + 2*(x4**3) - 2*(inner)**1.5 - 9* x2 * x4* x6)

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description='DDI Parameter Space Generator')
    parser.add_argument('--polymer', type=int, default=0, help='Polymer model ID (0, 1, 2, 3, 4, 44, 444, 5, 55, 6)')
    parser.add_argument('--res', type=int, default=300, help='Grid resolution (default 300)')
    args = parser.parse_args()

    polymer = args.polymer
    nnx = nny = nnz = args.res

    # --- 2. Grid Setup ---
    x1 = np.linspace(1e-7, 2, nnx)
    y1 = np.linspace(1e-7, 2, nny)
    z1 = np.linspace(1e-7, 2, nnz)
    a, b, c = np.meshgrid(x1, y1, z1)

    # --- 3. Model params & Jacobian ---
    us = a + b + c
    vs = b / (us**2)
    ws = c / (us**2)

    fu, fv, fw = -1 + 2 * us * vs + 2 * us * ws, us**2, us**2
    gu, gv, gw = -2 * us * vs, -us**2, 0
    hu, hv, hw = -2 * us * ws, 0, -us**2

    Tr = fu + gv
    D = fu * gv - fv * gu

    TrJF = fu + gv + hw
    TDA = D + Tr * hw
    DJF = D * hw

    b2 = -TrJF
    b1 = TDA - hu * fw
    b0 = hu * fw * gv - DJF
    b2b1b0 = b2 * b1 - b0

    # --- 4. Diffusion Matrices ---
    polymer_models = {
        0: np.array([[1.26, 0, 0], [0, 2.32, 0], [0, 0, 5.07]]),
        1: np.array([[1.26, -0.55, -104], [-0.42, 1.32, 60], [-0.00013, 0.00004, 0.07]]),
        2: np.array([[0.66, 0.08, -12], [0.07, 0.81, 11], [-0.00002, 0, 0.061]]),
        3: np.array([[0.67, 0.06, 36], [0.14, 0.87, -23], [0.00006, 0.0001, 0.064]]),
        4: np.array([[0.57, 6, 8], [-0.012, 1.6, 1.8], [-0.0006, -0.08, 0.56]]),
        44: np.array([[0.57, 6, 2.4], [-0.0804, 1.6, 1.8], [-0.0006, -0.08, 0.56]]),
        444: np.array([[0.57, 6, 0.08], [-0.0804, 1.6, 1.8], [-0.0006, -0.08, 0.56]]),
        5: np.array([[0.5, 5.2, -45], [-0.012, 1.2, -12], [-0.00022, 0.0032, 0.4]]),
        55: np.array([[0.5, 5.2, -45], [-0.012, 10, -12], [-0.00022, 0.0032, 0.4]]),
        6: np.array([[2.5, 0, 1.45], [0, 2.5, 2.7], [0, 0.7, 1.0]]),
        77: np.array([[1, 0.5, 0.5], [0.01, 10, 0], [0.01, 0, 10]]),
    }

    main_folder = "All_polymers_models"
    folder_name = os.path.join(main_folder, f"polymer_{polymer}")
    os.makedirs(folder_name, exist_ok=True)

    cdif = polymer_models.get(polymer, polymer_models[0])
    nddif = cdif / cdif[0, 0]
    d11, d12, d13 = nddif[0, :]
    d21, d22, d23 = nddif[1, :]
    d31, d32, d33 = nddif[2, :]
    CD = np.array([[1, d12, d13], [d21, d22, d23], [d31, d32, d33]])
    
    determCD = np.linalg.det(CD)
    traceCD = np.trace(CD)

    # --- 5. DDI Coefficients (Stability Matrices) ---
    pp1 = d22 + d33 + d22 * d33
    pp2 = d13 * d31 + d23 * d32 + d12 * d21
    s2 = pp1 - pp2
    s1 = (d31*fw + d13*hu) + (d21*fv + d12*gu) - (hw*(1+d22)) - (d22*fu + gv) - ((fu+gv)*d33)
    s0 = D + Tr * hw - hu * fw

    m6 = determCD
    a7, a8 = d21 * fv + d12 * gu, d22 * fu + gv
    m4 = (d13*d31*gv + d23*d32*fu + d22*d31*fw + d13*d22*hu + d12*d21*hw + d33*a7) - \
         (d13*d32*gu + d23*d31*fv + d32*d21*fw + d23*d12*hu + d22*hw + d33*a8)
    m2 = (a8*hw + d33*D) - (hw*a7) + (d32*gu*fw - d22*fw*hu - d31*fw*gv - d13*hu*gv + d23*hu*fv)
    m0 = -DJF + fw * gv * hu

    condm2 = m4 * m4 - 3 * m2 * m6
    condm1 = np.ones_like(condm2)
    mask_m = condm2 > 0
    if np.any(mask_m):
        condm1[mask_m] = discriminant_f(m6, m4[mask_m], m2[mask_m], m0[mask_m])

    cc1 = d23*d32 + d13*d22*d31 + d12*d21*d33
    cc2 = d22*d33 + d12*d23*d31 + d13*d21*d32
    c6 = cc1 - cc2 + traceCD * ((d22*d33 + d22 + d33) - (d12*d21 + d13*d31 + d23*d32))
    
    cb3 = d12*gu + d13*hu + d21*fv + d31*fw
    cb4 = d22*(fu+hw) + d33*(fu+gv) + gv + hw
    c4 = (d12*d23*hu + d13*d32*gu + d21*d32*fw + d22*d33*fu + d23*d31*fv + d22*hw + d33*gv) - \
         (d12*d21*hw + d12*d33*gu + d13*d22*hu + d13*d31*gv + d21*d33*fv + d22*d31*fw + d23*d32*fu) + \
         (traceCD * (cb3 - cb4)) - (TrJF * ((d22*d33 + d22 + d33) - (d12*d21 + d13*d31 + d23*d32)))
    
    c2 = (d12*gu*hw + d13*gv*hu + d21*fv*hw + d31*fw*gv) - \
         (d22*(fu*hw - fw*hu) + d23*fv*hu + d32*fw*gu + d33*D + gv*hw) + \
         (traceCD * (TDA - fw*hu)) - (TrJF * (cb3 - cb4))
    
    c0 = DJF - fw*gv*hu - TrJF*(TDA - fw*hu)

    condc2 = c4 * c4 - 3 * c2 * c6
    condc1 = np.ones_like(condc2)
    mask_c = condc2 > 0
    if np.any(mask_c):
        condc1[mask_c] = discriminant_f(c6, c4[mask_c], c2[mask_c], c0[mask_c])

    # --- 6. Stability Conditions ---
    base_stab = (b2 > 0) & (b1 > 0) & (b0 > 0) & (b2b1b0 > 0)
    ddi_check = (s1 < 0) & (s1**2 - 4 * s0 * s2 > 0)
    
    conditions = {
        0: base_stab, # No diffusion
        1: base_stab & ddi_check,
        2: base_stab & (condm1 < 0) & (m2 < 0),
        22: base_stab & (condm1 < 0) & (condm2 > 0) & (m4 < 0), # Case 2
        3: base_stab & (condc1 < 0) & (c2 < 0),
        33: base_stab & (condc1 < 0) & (condc2 > 0) & (c4 < 0),
        4: base_stab & ddi_check & (condm1 < 0) & (m2 < 0),
        44: base_stab & ddi_check & (condm1 < 0) & (condm2 > 0) & (m4 < 0),
        5: base_stab & ddi_check & (condc1 < 0) & (c2 < 0),
        55: base_stab & ddi_check & (condc1 < 0) & (condc2 > 0) & (c4 < 0),
        6: base_stab & (condm1 < 0) & (m2 < 0) & (condc1 < 0) & (c2 < 0),
        66: base_stab & (condm1 < 0) & (condm2 > 0) & (m4 < 0) & (condc1 < 0) & (c2 < 0),
        7: base_stab & (condm1 < 0) & (m2 < 0) & (condc1 < 0) & (condc2 > 0) & (c4 < 0),
        77: base_stab & (condm1 < 0) & (condm2 > 0) & (m4 < 0) & (condc1 < 0) & (condc2 > 0) & (c4 < 0),
        8: base_stab & ddi_check & (condm1 < 0) & (m2 < 0) & (condc1 < 0) & (c2 < 0),
        88: base_stab & ddi_check & (condm1 < 0) & (condm2 > 0) & (m4 < 0) & (condc1 < 0) & (c2 < 0),
        9: base_stab & ddi_check & (condm1 < 0) & (m2 < 0) & (condc1 < 0) & (condc2 > 0) & (c4 < 0),
        99: base_stab & ddi_check & (condm1 < 0) & (condm2 > 0) & (m4 < 0) & (condc1 < 0) & (condc2 > 0) & (c4 < 0)
    }

    condition_folder_names = {
        0: "without_diffusion", 1: "space_1_basic_ddi",
        2: "space_2_cs1", 22: "space_2_cs2", 3: "space_3_cs1", 33: "space_3_cs2",
        4: "space_4_cs1", 44: "space_4_cs2", 5: "space_5_cs1", 55: "space_5_cs2",
        6: "space_6_cs1", 66: "space_6_cs2", 7: "space_7_cs1", 77: "space_7_cs2",
        8: "space_8_cs1", 88: "space_8_cs2", 9: "space_9_cs1", 99: "space_9_cs2"
    }

    # --- 7. Main Loop ---
    for c_id, mask in conditions.items():
        if not np.any(mask): continue
        print(f"Processing space: {c_id}")
        
        path = os.path.join(folder_name, condition_folder_names[c_id])
        os.makedirs(path, exist_ok=True)
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(a[mask], b[mask], c[mask], color='blue', s=20, alpha=0.3)
        ax.set_xlabel('a', fontsize=15); ax.set_ylabel('b', fontsize=15); ax.set_zlabel('c', fontsize=15)
        
        plt.savefig(os.path.join(path, f'plot_p{polymer}_s{c_id}.jpg'), dpi=300)
        plt.close(fig)

        df = pd.DataFrame({'x coord': a[mask].ravel(), 'y coord': b[mask].ravel(), 'z coord': c[mask].ravel()})
        df.to_csv(os.path.join(path, f'coords_p{polymer}_s{c_id}.csv'), index=False)

    print(f"Workflow Complete for Polymer {polymer}.")

if __name__ == "__main__":
    main()
