"""
DDI Mode Isolation & Stability Analysis
Created by: Victor Juma (vjuma23@gmail.com)
Updated: February 14, 2026

DESCRIPTION:
This script performs mode isolation analysis for three-component reaction-diffusion 
systems. It calculates characteristic polynomials (Q0, Q1, Q2, Q3) and 
eigenvalues across wavenumbers (k^2) and modes (n).

HOW TO RUN:
python mode_isolate.py --polymer 6 --gamma 250
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

def main():
    # --- 1. Argument Parsing & Configuration ---
    parser = argparse.ArgumentParser(description='DDI Mode Isolation Analysis')
    parser.add_argument('--polymer', type=int, default=6, help='Polymer model ID')
    parser.add_argument('--gamma', type=float, default=250.0, help='Scale factor gamma')
    parser.add_argument('--a', type=float, default=0.175, help='Parameter a')
    parser.add_argument('--b', type=float, default=0.088, help='Parameter b')
    parser.add_argument('--c', type=float, default=0.1, help='Parameter c')
    parser.add_argument('--res', type=int, default=2000, help='Plot resolution')
    args = parser.parse_args()

    # Grid Setup
    max_nn = 10
    nn = np.linspace(0, max_nn, args.res)
    ksqd = (nn * np.pi)**2

    # Figure Global Settings
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 16
    })
    LINEWIDTH_CURVES = 3.5
    LINEWIDTH_DASHED = 1.5

    # --- 2. Diffusion Matrices ---
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

    if args.polymer not in polymer_models:
        print(f"Error: Polymer ID {args.polymer} not found.")
        sys.exit()

    output_path = f"mode_isolate_results/poly_{args.polymer}"
    os.makedirs(output_path, exist_ok=True)

    # --- 3. Mathematical Calculations ---
    us = args.a + args.b + args.c
    vs, ws = args.b / (us**2), args.c / (us**2)
    fu, fv, fw = -1 + 2*us*vs + 2*us*ws, us**2, us**2
    gu, gv = -2*us*vs, -us**2
    hu, hw = -2*us*ws, -us**2

    TrJF, D = (fu + gv + hw), (fu * gv - fv * gu)
    TDA, DJF = (D + (fu + gv) * hw), (D * hw)

    cdif = polymer_models[args.polymer]
    nddif = cdif / cdif[0, 0]
    d11, d12, d13 = nddif[0, :]; d21, d22, d23 = nddif[1, :]; d31, d32, d33 = nddif[2, :]
    traceCD, determCD = np.trace(nddif), np.linalg.det(nddif)

    # Polynomials
    y_Q3 = np.ones_like(ksqd)
    y_Q2 = traceCD * ksqd - args.gamma * TrJF

    pp1, pp2 = (d22 + d33 + d22*d33), (d13*d31 + d23*d32 + d12*d21)
    s2 = pp1 - pp2
    s1 = (d31*fw + d13*hu) + (d21*fv + d12*gu) - (hw*(1+d22)) - (d22*fu + gv) - ((fu+gv)*d33)
    s0 = D + (fu+gv)*hw - hu*fw
    y_Q1 = s2 * ksqd**2 + args.gamma * s1 * ksqd + args.gamma**2 * s0

    m6, m0 = determCD, -DJF + fw*gv*hu
    a7, a8 = d21*fv + d12*gu, d22*fu + gv
    m4 = (d13*d31*gv + d23*d32*fu + d22*d31*fw + d13*d22*hu + d12*d21*hw + d33*a7) - \
         (d13*d32*gu + d23*d31*fv + d32*d21*fw + d23*d12*hu + d22*hw + d33*a8)
    m2 = (a8*hw + d33*D) - (hw*a7) + (d32*gu*fw - d22*fw*hu - d31*fw*gv - d13*hu*gv + d23*hu*fv)
    y_Q0 = m6 * ksqd**3 + args.gamma * m4 * ksqd**2 + args.gamma**2 * m2 * ksqd + args.gamma**3 * m0

    # Polynomial Q2Q1 - Q0
    cc1, cc2 = (d23*d32 + d13*d22*d31 + d12*d21*d33), (d22*d33 + d12*d23*d31 + d13*d21*d32)
    c6 = cc1 - cc2 + traceCD * ((d22*d33 + d22 + d33) - (d12*d21 + d13*d31 + d23*d32))
    cb3, cb4 = (d12*gu + d13*hu + d21*fv + d31*fw), (d22*(fu+hw) + d33*(fu+gv) + gv + hw)
    c4 = (d12*d23*hu + d13*d32*gu + d21*d32*fw + d22*d33*fu + d23*d31*fv + d22*hw + d33*gv) - \
         (d12*d21*hw + d12*d33*gu + d13*d22*hu + d13*d31*gv + d21*d33*fv + d22*d31*fw + d23*d32*fu) + \
         (traceCD * (cb3 - cb4)) - (TrJF * ((d22*d33 + d22 + d33) - (d12*d21 + d13*d31 + d23*d32)))
    c2 = (d12*gu*hw + d13*gv*hu + d21*fv*hw + d31*fw*gv) - \
         (d22*(fu*hw - fw*hu) + d23*fv*hu + d32*fw*gu + d33*D + gv*hw) + \
         (traceCD * (TDA - fw*hu)) - (TrJF * (cb3 - cb4))
    c0 = DJF - fw*gv*hu - TrJF*(TDA - fw*hu)
    y_Poly = c6 * ksqd**3 + args.gamma * c4 * ksqd**2 + args.gamma**2 * c2 * ksqd + args.gamma**3 * c0

    # Max Real Parts of Eigenvalues
    max_real = np.array([np.max(np.real(np.roots([1, y_Q2[i], y_Q1[i], y_Q0[i]]))) for i in range(len(ksqd))])

    # --- 4. Plotting function ---
    def generate_plots(x_axis, x_label, suffix, is_k2=False):
        # Figure 1: 1x4 Panel (Q1, Q0, Q2Q1-Q0, Real Part)
        fig, axs = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
        plot_data = [
            (y_Q1, 'y', '$Q_1$', [-1E3, 1E5]),
            (y_Q0, 'm', '$Q_0$', [-2E4, 8E4]),
            (y_Poly, 'b', '$Q_2Q_1 - Q_0$', [-1E5, 1E8]),
            (max_real, 'r', 'Real Part', [-2, 0.5])
        ]
        
        for i, (data, color, ylabel, ylim) in enumerate(plot_data):
            axs[i].plot(x_axis, data, color, linewidth=LINEWIDTH_CURVES)
            axs[i].axhline(0, color='k', linestyle='--', linewidth=LINEWIDTH_DASHED)
            axs[i].set_ylabel(ylabel); axs[i].set_xlabel(x_label); axs[i].set_ylim(ylim); axs[i].grid(True)
            if i < 3:
                axs[i].fill_between(x_axis, ylim[0], ylim[1], where=(data < 0), color=color, alpha=0.2, linewidth=0)
            else:
                axs[i].fill_between(x_axis, ylim[0], ylim[1], where=(data > 0), color='r', alpha=0.2, linewidth=0)
            for m in range(max_nn + 1):
                vline_x = (m * np.pi)**2 if is_k2 else m
                axs[i].axvline(x=vline_x, color='k', linestyle='--', linewidth=0.5)

        #
        fname_main = f"pol_{args.polymer}_Q_gam_{args.gamma}{'_k2' if is_k2 else ''}.jpg"
        plt.savefig(os.path.join(output_path, fname_main), dpi=600, bbox_inches='tight')

        # Figure 2: 1x2 Panel (Q3 and Q2)
        fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        axs2[0].plot(x_axis, y_Q3, 'k', linewidth=LINEWIDTH_CURVES); axs2[0].set_ylabel('$Q_3$'); axs2[0].set_ylim([0, 2])
        axs2[1].plot(x_axis, y_Q2, 'g', linewidth=LINEWIDTH_CURVES); axs2[1].set_ylabel('$Q_2$')
        
        for ax in axs2:
            ax.axhline(0, color='k', linestyle='--', linewidth=LINEWIDTH_DASHED)
            ax.set_xlabel(x_label); ax.grid(True)
            for m in range(max_nn + 1):
                vline_x = (m * np.pi)**2 if is_k2 else m
                ax.axvline(x=vline_x, color='k', linestyle='--', linewidth=0.5)

        #
        fname_q2q3 = f"pol_{args.polymer}_Q2_Q3_gam_{args.gamma}{'_k2' if is_k2 else ''}.jpg"
        plt.savefig(os.path.join(output_path, fname_q2q3), dpi=600, bbox_inches='tight')

    # Generate both sets of plots (mode n and k^2)
    generate_plots(nn, "mode (n)", "mode", is_k2=False)
    generate_plots(ksqd, r"$k^2$", "k2", is_k2=True)
    print(f"Workflow complete. Results saved in: {output_path}")

if __name__ == "__main__":
    main()
