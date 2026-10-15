#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gsta_gao_collins.py - Analise de Tendencia Granulometrica (GSTA)

Metodo de Gao & Collins (1992, 1994), com as opcoes de vizinhanca e de
barreiras de Poizot & Mear (2010), e teste de significancia por permutacao.

Referencias
  Gao, S. & Collins, M. (1992). Net sediment transport patterns inferred from
      grain-size trends, based upon definition of "transport vectors".
      Sedimentary Geology, 81, 47-60.
  Gao, S. & Collins, M. B. (1994). Analysis of grain size trends, for defining
      sediment transport pathways in marine environments. Journal of Coastal
      Research, 10(1), 70-78.
  Poizot, E. & Mear, Y. (2010). Using a GIS to enhance grain size trend
      analysis. Environmental Modelling & Software, 25, 513-525.
  Le Roux, J. P. & Rojas, E. M. (2007). Sediment transport patterns determined
      from grain size parameters: Overview and state of the art. Sedimentary
      Geology, 202, 473-488.

Procedimento
  1. Coordenadas geograficas sao projetadas (padrao: UTM 36S, EPSG:32736),
     de modo que a distancia caracteristica (Dcr) fica em metros.
  2. Vizinhos de cada estacao: todas as estacoes a uma distancia <= Dcr cujo
     segmento de ligacao nao atravessa uma barreira (terra, ilhas).
  3. Para cada par (i -> j) define-se um vetor unitario de i para j quando
     ocorre um dos casos escolhidos (parametros em phi):
        FB-: j mais fina (Mz maior), mais bem selecionada (sigma menor) e
             mais negativamente assimetrica (Sk menor);
        CB+: j mais grossa (Mz menor), mais bem selecionada e mais
             positivamente assimetrica (Sk maior).
     Os vetores sao somados por estacao (Gao & Collins, 1994, Eq. 1).
  4. Suavizacao opcional: media vetorial da estacao com os vizinhos dentro
     de Dcr (Gao & Collins, 1994, Eq. 2). Le Roux & Rojas (2007) recomendam
     suavizar apenas quando os vetores nao suavizados ja sao significativos.
  5. Significancia: os trios (Mz, sigma, Sk) sao redistribuidos ao acaso
     entre as estacoes (posicoes fixas). Para cada estacao, p = fracao de
     replicas com modulo >= observado. Para o campo inteiro, compara-se o
     modulo da resultante global.

Saidas (prefixo definido por --saida)
  <prefixo>_vetores.csv      uma linha por estacao e cenario
  <prefixo>_vetores.geojson  pontos com azimute e modulo (abrir no QGIS 3)
  <prefixo>_resumo.csv       resultante global e p global por cenario
  Azimute em graus a partir do norte, sentido horario. No QGIS, use um
  simbolo de seta com rotacao definida pelo campo "azimute".

Entrada: CSV com colunas
  estacao, lon, lat, Mz, sigma, Sk   (Mz, sigma e Sk em phi, Folk & Ward)
  Separador ; ou , e decimal . ou , sao detectados automaticamente.

Exemplo
  python gsta_gao_collins.py dados_gsta.csv --dcr 2700 3500 \
      --barreira terra.geojson --permutacoes 999 --saida inhambane

Requisitos: Python 3.8+, numpy, pandas. Opcionais: pyproj (projecao;
sem ele usa projecao equiretangular local), shapely (barreiras).
"""
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

CASOS = ("FB-", "CB+")


# ---------------------------------------------------------------- entrada
def ler_dados(caminho):
    with open(caminho, encoding="utf-8-sig") as f:
        cab = f.readline()
    sep = ";" if cab.count(";") > cab.count(",") else ","
    dec = "," if sep == ";" else "."
    df = pd.read_csv(caminho, sep=sep, decimal=dec, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    falta = {"estacao", "lon", "lat", "Mz", "sigma", "Sk"} - set(df.columns)
    if falta:
        sys.exit(f"Colunas ausentes no CSV: {sorted(falta)}")
    return df.reset_index(drop=True)


def projetar(lon, lat, epsg):
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = tr.transform(np.asarray(lon), np.asarray(lat))
        return np.c_[x, y], f"EPSG:{epsg}"
    except ImportError:
        lat0 = np.deg2rad(np.mean(lat))
        x = (np.asarray(lon) - np.mean(lon)) * 111320.0 * np.cos(lat0)
        y = (np.asarray(lat) - np.mean(lat)) * 110574.0
        return np.c_[x, y], "equiretangular local (pyproj ausente)"


def ler_barreiras(caminho, epsg):
    """GeoJSON (EPSG:4326) com poligonos ou linhas que bloqueiam o transporte."""
    if not caminho:
        return None
    try:
        from shapely.geometry import shape
        from shapely.ops import transform, unary_union
        from pyproj import Transformer
    except ImportError:
        sys.exit("Barreiras exigem shapely e pyproj (pip install shapely pyproj).")
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    with open(caminho, encoding="utf-8") as f:
        gj = json.load(f)
    geoms = [transform(tr.transform, shape(ft["geometry"])) for ft in gj["features"]]
    return unary_union(geoms)


# ---------------------------------------------------------------- metodo
def matriz_vizinhanca(X, dcr, barreira=None):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    V = (D <= dcr) & (D > 0)
    if barreira is not None:
        from shapely.geometry import LineString
        for i, j in zip(*np.where(np.triu(V))):
            seg = LineString([X[i], X[j]])
            if seg.crosses(barreira) or seg.within(barreira):
                V[i, j] = V[j, i] = False
    return D, V


def tendencia(P, casos):
    """T[i, j] = True se o par i -> j satisfaz algum dos casos escolhidos."""
    M, S, K = P[:, 0], P[:, 1], P[:, 2]
    fino = M[None, :] > M[:, None]
    grosso = M[None, :] < M[:, None]
    melhor = S[None, :] < S[:, None]
    neg = K[None, :] < K[:, None]
    pos = K[None, :] > K[:, None]
    T = np.zeros((len(P), len(P)), bool)
    if "FB-" in casos:
        T |= fino & melhor & neg
    if "CB+" in casos:
        T |= grosso & melhor & pos
    return T


def vetores(P, X, D, V, casos, suavizar):
    T = tendencia(P, casos) & V
    with np.errstate(invalid="ignore", divide="ignore"):
        U = (X[None, :, :] - X[:, None, :]) / D[:, :, None]   # unitarios i -> j
    U = np.nan_to_num(U)
    R = (U * T[:, :, None]).sum(1)                            # Eq. 1
    if suavizar:                                              # Eq. 2
        W = V | np.eye(len(X), dtype=bool)
        R = (W[:, :, None] * R[None, :, :]).sum(1) / W.sum(1)[:, None]
    return R, T.sum(1)


def azimute(R):
    return (np.degrees(np.arctan2(R[:, 0], R[:, 1])) + 360.0) % 360.0


def cenario(P, X, D, V, casos, suavizar, nperm, rng):
    R, nval = vetores(P, X, D, V, casos, suavizar)
    mod = np.linalg.norm(R, axis=1)
    G = R.sum(0)
    cont = np.zeros(len(P))
    cont_g = 0
    for _ in range(nperm):
        Rp, _ = vetores(P[rng.permutation(len(P))], X, D, V, casos, suavizar)
        cont += np.linalg.norm(Rp, axis=1) >= mod - 1e-12
        cont_g += np.linalg.norm(Rp.sum(0)) >= np.linalg.norm(G) - 1e-12
    p = (1 + cont) / (nperm + 1)
    p[mod == 0] = np.nan
    pg = (1 + cont_g) / (nperm + 1)
    return R, mod, nval, p, G, pg


# ---------------------------------------------------------------- execucao
def main():
    ap = argparse.ArgumentParser(description="GSTA de Gao & Collins com teste de permutacao.")
    ap.add_argument("csv")
    ap.add_argument("--dcr", type=float, nargs="+", required=True, help="distancia(s) caracteristica(s) em metros")
    ap.add_argument("--epsg", type=int, default=32736, help="projecao metrica (padrao UTM 36S)")
    ap.add_argument("--barreira", help="GeoJSON de terra/ilhas em EPSG:4326")
    ap.add_argument("--suavizar", action="store_true", help="aplica a Eq. 2 de Gao & Collins (1994)")
    ap.add_argument("--permutacoes", type=int, default=999)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--saida", default="gsta")
    a = ap.parse_args()

    df = ler_dados(a.csv)
    P = df[["Mz", "sigma", "Sk"]].to_numpy(float)
    X, proj = projetar(df.lon, df.lat, a.epsg)
    barreira = ler_barreiras(a.barreira, a.epsg)
    rng = np.random.default_rng(a.semente)

    D0 = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    np.fill_diagonal(D0, np.inf)
    nn = D0.min(1)
    print(f"Projecao: {proj}")
    print(f"Distancia ao vizinho mais proximo: media {nn.mean():.0f} m, "
          f"min {nn.min():.0f} m, max {nn.max():.0f} m")

    linhas, resumo, feats = [], [], []
    combos = [CASOS, ("FB-",), ("CB+",)]
    for dcr in a.dcr:
        D, V = matriz_vizinhanca(X, dcr, barreira)
        sem = df.estacao[V.sum(1) == 0].tolist()
        if sem:
            print(f"Aviso (Dcr={dcr:.0f} m): estacoes sem vizinhos: {', '.join(map(str, sem))}")
        for casos in combos:
            nome = f"{'+'.join(casos)}_Dcr{dcr:.0f}{'_suav' if a.suavizar else ''}"
            R, mod, nval, p, G, pg = cenario(P, X, D, V, casos, a.suavizar, a.permutacoes, rng)
            az = azimute(R)
            azg = float(azimute(G[None])[0])
            resumo.append(dict(cenario=nome, casos="+".join(casos), dcr_m=dcr,
                               azimute_global=round(azg, 1), modulo_global=round(float(np.linalg.norm(G)), 3),
                               p_global=round(pg, 4), estacoes_p_menor_005=int(np.nansum(p < 0.05)),
                               estacoes_sem_vetor=int((mod == 0).sum())))
            for k in range(len(df)):
                reg = dict(cenario=nome, estacao=df.estacao[k], lon=df.lon[k], lat=df.lat[k],
                           vizinhos=int(V[k].sum()), pares_validos=int(nval[k]),
                           azimute=None if mod[k] == 0 else round(float(az[k]), 1),
                           modulo=round(float(mod[k]), 3),
                           p=None if np.isnan(p[k]) else round(float(p[k]), 4))
                linhas.append(reg)
                feats.append({"type": "Feature",
                              "geometry": {"type": "Point", "coordinates": [float(df.lon[k]), float(df.lat[k])]},
                              "properties": {kk: vv for kk, vv in reg.items() if kk not in ("lon", "lat")}})
            print(f"{nome:28s} resultante {azg:5.0f} graus  p_global={pg:.3f}  "
                  f"estacoes p<0,05: {int(np.nansum(p < 0.05))}  sem vetor: {int((mod == 0).sum())}")

    pd.DataFrame(linhas).to_csv(f"{a.saida}_vetores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(resumo).to_csv(f"{a.saida}_resumo.csv", index=False, encoding="utf-8-sig")
    with open(f"{a.saida}_vetores.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    print(f"Arquivos gravados com o prefixo '{a.saida}'.")


if __name__ == "__main__":
    main()
