"""
Auditoria: Mean_Experimental_Data vs leitura/plots do plot_tool_V8.
Compara aba a aba (opções 1..N do menu) com o Excel original.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import plot_tool_V8 as pt

FILE = Path(
    r"g:/Meu Drive/LEMI/uncertainties/data_example/example/mean_sf6_v2/"
    "Mean_Experimental_Data_FSC2_SF6_Oil_v2.xlsx"
)

TOL_REL = 1e-4
TOL_ABS = 1e-6


def _col_by_substr(columns, *parts):
    for c in columns:
        if c is None or (isinstance(c, float) and pd.isna(c)):
            continue
        s = str(c).replace("\n", " ").strip()
        if all(p.lower() in s.lower() for p in parts):
            return c
    return None


def read_raw_block(sheet: str):
    hdr = pd.read_excel(FILE, sheet_name=sheet, header=None, nrows=4, usecols="B:Z")
    names = hdr.iloc[2].tolist()
    units_row = hdr.iloc[3].tolist()
    units = {}
    for n, u in zip(names, units_row):
        if pd.notna(n):
            units[str(n).replace("\n", " ").strip()] = (
                "" if pd.isna(u) else str(u).replace("\n", " ").strip()
            )
    df = pd.read_excel(
        FILE, sheet_name=sheet, header=None, skiprows=4, usecols="B:Z", nrows=16
    )
    df.columns = names
    return df, units


def _series(df, col):
    if col is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    block = df[col]
    if isinstance(block, pd.DataFrame):
        block = block.iloc[:, 0]
    return block


def _numeric_raw(ser):
    out = []
    for v in ser:
        if pt._is_empty_measurement_cell(v) or pt._is_blank_excel_value(v):
            out.append(np.nan)
            continue
        if isinstance(v, str) and "#" in v:
            out.append(np.nan)
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, dtype=float)


def _compare_arrays(name, raw, proc, sheet):
    issues = []
    n = min(len(raw), len(proc))
    for i in range(n):
        r, p = raw[i], proc[i]
        if not np.isfinite(r) and not np.isfinite(p):
            continue
        if np.isfinite(r) != np.isfinite(p):
            issues.append(f"  {name} linha {i+1}: raw={r!r} proc={p!r}")
            continue
        if np.isfinite(r) and np.isfinite(p):
            if abs(r - p) > max(TOL_ABS, TOL_REL * max(abs(r), 1.0)):
                issues.append(f"  {name} linha {i+1}: raw={r} proc={p} diff={p-r}")
    if len(issues) > 8:
        return [f"{sheet}: {name} — {len(issues)} divergências (1ªs 5):"] + issues[:5]
    if issues:
        return [f"{sheet}: {name}"] + issues
    return []


def audit_sheet(opt: int, sheet: str) -> list[str]:
    lines = [f"\n{'='*72}", f"Opção {opt}: {sheet}", "=" * 72]
    raw, raw_units = read_raw_block(sheet)
    df_tool, units_tool = pt.read_single_sheet(str(FILE), sheet)
    if df_tool is None:
        lines.append("ERRO: read_single_sheet falhou")
        return lines

    fluid_1, fluid_2, direction, theta, exp_id, _ = pt.extract_info_from_filename(
        sheet
    )
    lines.append(
        f"fluid_1={fluid_1} fluid_2={fluid_2} {direction} theta={theta} deg id={exp_id}"
    )

    # --- Colunas críticas ---
    col_jg = _col_by_substr(raw.columns, "jG") or _col_by_substr(raw.columns, "jg")
    col_jl = _col_by_substr(raw.columns, "jL") or _col_by_substr(raw.columns, "jl")
    col_alpha = _col_by_substr(raw.columns, "α") or _col_by_substr(
        raw.columns, "void", "fraction"
    )
    col_dpf = _col_by_substr(raw.columns, "dpdz", "F") or _col_by_substr(
        raw.columns, "dP", "dz", "F"
    )
    col_dpt = _col_by_substr(raw.columns, "dpdz", "T") or _col_by_substr(
        raw.columns, "dP", "dz", "T"
    )
    col_fp = _col_by_substr(raw.columns, "flow", "pattern")
    col_resg_excel = _col_by_substr(raw.columns, "Re_sg")
    col_mu = _col_by_substr(raw.columns, "Mu_gas")
    col_rho = _col_by_substr(raw.columns, "Rho_gas")

    cm = pt.build_column_mapping(df_tool)
    miss = pt.missing_column_keys(
        cm, ["jG", "jL", "Flow Pattern", "Temp.", "Gauge Pressure"]
    )
    if miss:
        lines.append(f"AVISO colunas ausentes no mapping: {miss}")
    pt.ensure_alpha_in_column_mapping(cm, df_tool.columns)
    for k in ("α", "dp/dz_F", "dp/dz_T"):
        if k not in cm:
            lines.append(f"AVISO: sem coluna canónica {k}")

    # Comparar leitura numérica (após coerce) vs Excel interpretado
    pairs = []
    if col_jg and "jG" in cm:
        pairs.append(("jG", _numeric_raw(_series(raw, col_jg)), pd.to_numeric(
            pt._one_col_series(df_tool, cm["jG"]), errors="coerce"
        ).to_numpy()))
    if col_jl and "jL" in cm:
        pairs.append(("jL", _numeric_raw(_series(raw, col_jl)), pd.to_numeric(
            pt._one_col_series(df_tool, cm["jL"]), errors="coerce"
        ).to_numpy()))
    if col_alpha and "α" in cm:
        pairs.append(("α", _numeric_raw(_series(raw, col_alpha)), pd.to_numeric(
            pt._one_col_series(df_tool, cm["α"]), errors="coerce"
        ).to_numpy()))
    if col_dpf and "dp/dz_F" in cm:
        pairs.append(("dp/dz_F", _numeric_raw(_series(raw, col_dpf)), pd.to_numeric(
            pt._one_col_series(df_tool, cm["dp/dz_F"]), errors="coerce"
        ).to_numpy()))
    if col_dpt and "dp/dz_T" in cm:
        pairs.append(("dp/dz_T", _numeric_raw(_series(raw, col_dpt)), pd.to_numeric(
            pt._one_col_series(df_tool, cm["dp/dz_T"]), errors="coerce"
        ).to_numpy()))

    for name, a, b in pairs:
        lines.extend(_compare_arrays(name, a, b, sheet))

    # Pós-padronização
    df_std = df_tool.copy()
    pt.standardize_liquid_conditions({sheet: df_std})
    pt.compute_Re_sg_column(df_std, cm, fluid_1, units_dict=units_tool)

    if col_resg_excel:
        excel_re = _numeric_raw(_series(raw, col_resg_excel))
        calc_re = pd.to_numeric(df_std["Re_sg"], errors="coerce").to_numpy()
        bad = []
        for i in range(len(excel_re)):
            e, c = excel_re[i], calc_re[i]
            if not np.isfinite(e) and not np.isfinite(c):
                continue
            if not np.isfinite(e) and np.isfinite(c):
                if col_mu and _numeric_raw(_series(raw, col_mu))[i] <= 0:
                    bad.append(
                        f"  linha {i+1}: Excel Re_sg vazio, calc OK (Mu_gas=0 → CoolProp)"
                    )
                else:
                    bad.append(f"  linha {i+1}: Excel sem Re_sg, calc={c:.1f}")
            elif np.isfinite(e) and np.isfinite(c):
                rel = abs(e - c) / max(abs(e), 1.0)
                if rel > 0.05:
                    bad.append(
                        f"  linha {i+1}: Excel Re_sg={e:.0f} calc={c:.0f} rel_err={rel:.1%}"
                    )
        if bad:
            lines.append("Re_sg Excel vs compute_Re_sg_column (>5% ou só num lado):")
            lines.extend(bad[:8])
            if len(bad) > 8:
                lines.append(f"  ... +{len(bad)-8} linhas")
        else:
            lines.append("Re_sg: coerente com Excel (tolerância 5%) ou Mu_gas=0 corrigido")

    # Contagens como nos plots
    df_plot = df_std.copy().reset_index(drop=True)
    jl_col = cm["jL"]
    pt._cluster_measured_jl_legend_labels(df_plot, jl_col)
    jg = pd.to_numeric(pt._one_col_series(df_plot, cm["jG"]), errors="coerce")
    alpha = pd.to_numeric(pt._one_col_series(df_plot, cm["α"]), errors="coerce")
    resg = pd.to_numeric(df_plot["Re_sg"], errors="coerce")
    if "dp/dz_F" in cm:
        dpf = pd.to_numeric(pt._one_col_series(df_plot, cm["dp/dz_F"]), errors="coerce") / 1000
    else:
        dpf = pd.Series(np.nan, index=df_plot.index)
    if "dp/dz_T" in cm:
        dpt = pd.to_numeric(pt._one_col_series(df_plot, cm["dp/dz_T"]), errors="coerce") / 1000
    else:
        dpt = pd.Series(np.nan, index=df_plot.index)

    n_alpha_jg = int((jg.notna() & alpha.notna()).sum())
    n_fric = int((jg.notna() & dpf.notna()).sum())
    n_tot = int((jg.notna() & dpt.notna()).sum())
    if "Re_sl_group" in df_plot.columns:
        re_sl = pd.to_numeric(df_plot["Re_sl_group"], errors="coerce")
        n_alpha_resg = int((resg.notna() & alpha.notna()).sum())
        n_fric_resg = int((resg.notna() & dpf.notna()).sum())
        n_tot_resg = int((resg.notna() & dpt.notna()).sum())
        re_series = sorted(re_sl.dropna().unique())
        lines.append(
            f"Pontos plotáveis: α×jG={n_alpha_jg} F×jG={n_fric} T×jG={n_tot} | "
            f"α×Re_sg={n_alpha_resg} F×Re_sg={n_fric_resg} T×Re_sg={n_tot_resg}"
        )
        lines.append(f"Re_sl_group (séries): {[round(float(x), 1) for x in re_series]}")
        if n_tot_resg < n_tot:
            lines.append(
                f"AVISO: {n_tot - n_tot_resg} ponto(s) total×jG sem Re_sg (série omitida em total_vs_Re_g)"
            )
    else:
        lines.append(
            f"Pontos plotáveis: α×jG={n_alpha_jg} F×jG={n_fric} T×jG={n_tot}"
        )

    jl_groups = sorted(df_plot["_jl_legend_group_mean"].dropna().unique())
    lines.append(f"j_L legend groups: {[float(x) for x in jl_groups]}")

    # Placeholders em Flow Pattern
    if col_fp and "Flow Pattern" in cm:
        fp_raw = [_series(raw, col_fp).iloc[i] for i in range(len(raw))]
        fp_proc = pt._one_col_series(df_plot, cm["Flow Pattern"]).tolist()
        for i, (a, b) in enumerate(zip(fp_raw, fp_proc)):
            ca = pt.canonical_flow_pattern_name(a)
            cb = pt.canonical_flow_pattern_name(b)
            if str(a).strip() != str(b).strip() and ca == cb:
                pass  # ok canonical
            elif str(a).strip() != str(b).strip() and ca != cb:
                lines.append(f"  FP linha {i+1}: raw={a!r} → proc={b!r} canon={cb}")

    # Incertezas vs Excel
    for ukey, raw_hints in (
        ("U_alpha", ("U(alpha)", "U(α)")),
        ("U_dpdz_F", ("U(-dpdz_F", "U(-dP/dz F")),
        ("U_dpdz_T", ("U(-dpdz_T", "U(-dP/dz T")),
    ):
        ucol = pt._find_point_uncertainty_column(df_tool, cm, ukey)
        raw_col = None
        for h in raw_hints:
            raw_col = _col_by_substr(raw.columns, h.replace("(", "").split()[0][:6])
            if raw_col:
                break
        if raw_col is None:
            for c in raw.columns:
                if c and any(h.lower() in str(c).lower() for h in raw_hints):
                    raw_col = c
                    break
        if ucol and raw_col:
            a = _numeric_raw(_series(raw, raw_col))
            b = pd.to_numeric(pt._one_col_series(df_tool, ucol), errors="coerce").to_numpy()
            issues = _compare_arrays(ukey, a, b, sheet)
            if issues:
                lines.extend(issues)
        elif ucol:
            lines.append(f"U: {ukey} mapeado para {ucol!r} (sem coluna raw identificada)")

    # jL_raw deve igualar jL do Excel (antes do cluster)
    if col_jl and "jL" in cm:
        raw_jl = _numeric_raw(_series(raw, col_jl))
        jl_raw = pd.to_numeric(df_std.get("jL_raw"), errors="coerce").to_numpy()
        issues = _compare_arrays("jL_raw vs Excel jL", raw_jl, jl_raw, sheet)
        if issues:
            lines.extend(issues)
        else:
            lines.append("jL_raw: identico ao jL medido no Excel")

    # Temperatura e pressao inalteradas
    for key, hint in (("Temp.", ("temp",)), ("Gauge Pressure", ("gauge", "pressure"))):
        if key not in cm:
            continue
        raw_col = _col_by_substr(raw.columns, *hint)
        if not raw_col:
            continue
        a = _numeric_raw(_series(raw, raw_col))
        b = pd.to_numeric(pt._one_col_series(df_tool, cm[key]), errors="coerce").to_numpy()
        issues = _compare_arrays(key, a, b, sheet)
        if issues:
            lines.extend(issues)

    # Flow patterns unicos (canonico)
    if col_fp and "Flow Pattern" in cm:
        fps = []
        for i in range(len(raw)):
            r = _series(raw, col_fp).iloc[i]
            p = pt._one_col_series(df_plot, cm["Flow Pattern"]).iloc[i]
            if pt._is_blank_excel_value(r) and pt._is_blank_excel_value(p):
                continue
            cr = pt.canonical_flow_pattern_name(r)
            cp = pt.canonical_flow_pattern_name(p)
            if cr != cp:
                fps.append(f"  linha {i+1}: {r!r} vs {p!r}")
        uniq = sorted(
            {
                pt.canonical_flow_pattern_name(x)
                for x in pt._one_col_series(df_plot, cm["Flow Pattern"])
                if not pt._is_blank_excel_value(x)
            }
        )
        lines.append(f"Flow patterns (canonicos): {uniq}")
        if fps:
            lines.append("AVISO flow pattern divergente:")
            lines.extend(fps)

    # Serie de menor j_L recebe barras de incerteza (alpha)
    jl_col = cm.get("jL")
    if jl_col:
        idx = pt._indices_for_lowest_jl_series(
            df_plot,
            jl_col,
            pt._one_col_series(df_plot, cm["jG"]),
            pt._one_col_series(df_plot, cm["α"]),
        )
        lines.append(
            f"Incerteza alpha: {len(idx)} ponto(s) na serie j_L minima (indices {idx})"
        )

    return lines


def main():
    out_path = Path(__file__).parent / "audit_mean_report.txt"
    xl = pd.ExcelFile(FILE)
    sheets = xl.sheet_names
    chunks = [f"Arquivo: {FILE.name}", f"Abas ({len(sheets)}): {sheets}"]
    all_issues = []
    for opt, sheet in enumerate(sheets, 1):
        try:
            rep = audit_sheet(opt, sheet)
            chunks.append("\n".join(rep))
            for line in rep:
                low = line.strip().lower()
                if any(
                    k in low
                    for k in (
                        "erro",
                        "aviso",
                        "diverg",
                        "linha ",
                        "excel re_sg",
                        "sem coluna",
                    )
                ):
                    all_issues.append(f"[{sheet}] {line.strip()}")
        except Exception as e:
            import traceback

            chunks.append(f"\nOpcao {opt} {sheet}: EXCECAO {e}\n{traceback.format_exc()}")
            all_issues.append(f"{sheet}: EXCECAO {e}")

    chunks.append("\n" + "=" * 72)
    chunks.append("RESUMO DA AUDITORIA")
    chunks.append("=" * 72)
    if not all_issues:
        chunks.append("Nenhum problema critico automatico nas 10 abas.")
    else:
        chunks.append(f"{len(all_issues)} alertas:")
        chunks.extend(all_issues)

    out_path.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Relatorio: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
