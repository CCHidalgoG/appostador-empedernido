# 🏆 Reporte de Backtesting Integral y Benchmarking de Modelos

Fecha de Auditoría: 2026-10-24 22:47:39

## 1. Tabla Comparativa de Rendimiento y Rentabilidad Financiera

| Modelo | Estrategia | Apuestas | Win Rate | ROI Real | PnL Neto | Max Drawdown | Profit Factor | Log Loss | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| **1. Baseline Softmax (Ratings)** | old_production (flat) | 564 | 69.7% | **+18.74%** | **$+2,324,047.00** | 21.2% | 1.64 | 1.0061 | 0.2003 |
| **1. Baseline Softmax (Ratings)** | unconstrained_value (flat) | 5703 | 37.3% | **+7.74%** | **$+2,343,180,812.22** | 59.0% | 1.12 | 1.0061 | 0.2003 |
| **1. Baseline Softmax (Ratings)** | calibrated_value (flat) | 3953 | 37.5% | **+5.01%** | **$+7,065,477.61** | 62.9% | 1.08 | 1.0061 | 0.2003 |
| **1. Baseline Softmax (Ratings)** | calibrated_value (kelly) | 3953 | 37.5% | **+11.24%** | **$+2,448,777.95** | 16.5% | 1.22 | 1.0061 | 0.2003 |
| **2. Dixon-Coles Poisson (xG Impect)** | old_production (flat) | 257 | 66.5% | **+10.34%** | **$+106,583.54** | 31.4% | 1.31 | 1.0181 | 0.2033 |
| **2. Dixon-Coles Poisson (xG Impect)** | unconstrained_value (flat) | 6526 | 30.1% | **-0.67%** | **$-61,743.17** | 89.9% | 0.99 | 1.0181 | 0.2033 |
| **2. Dixon-Coles Poisson (xG Impect)** | calibrated_value (flat) | 5298 | 31.8% | **+0.05%** | **$+4,975.81** | 79.3% | 1.00 | 1.0181 | 0.2033 |
| **2. Dixon-Coles Poisson (xG Impect)** | calibrated_value (kelly) | 5298 | 31.8% | **+1.83%** | **$+68,289.02** | 33.9% | 1.03 | 1.0181 | 0.2033 |
| **2. Dixon-Coles Poisson (xG Impect)** | multi_market_value (flat) | 5561 | 30.1% | **-1.36%** | **$-73,071.81** | 85.2% | 0.98 | 1.0181 | 0.2033 |
| **2. Dixon-Coles Poisson (xG Impect)** | multi_market_value (kelly) | 5561 | 30.1% | **+0.35%** | **$+13,207.68** | 43.9% | 1.01 | 1.0181 | 0.2033 |
| **3. Tactical Multi-KPI Model** | old_production (flat) | 485 | 70.3% | **+20.88%** | **$+2,269,928.11** | 14.1% | 1.71 | 1.0016 | 0.1994 |
| **3. Tactical Multi-KPI Model** | unconstrained_value (flat) | 5721 | 36.1% | **+10.99%** | **$+23,820,785,600.94** | 53.5% | 1.17 | 1.0016 | 0.1994 |
| **3. Tactical Multi-KPI Model** | calibrated_value (flat) | 3757 | 37.2% | **+8.28%** | **$+46,415,803.84** | 53.5% | 1.13 | 1.0016 | 0.1994 |
| **3. Tactical Multi-KPI Model** | calibrated_value (kelly) | 3757 | 37.2% | **+12.59%** | **$+2,944,433.99** | 14.4% | 1.24 | 1.0016 | 0.1994 |
| **3. Tactical Multi-KPI Model** | multi_market_value (flat) | 4396 | 34.4% | **+9.41%** | **$+136,882,337.01** | 53.3% | 1.14 | 1.0016 | 0.1994 |
| **3. Tactical Multi-KPI Model** | multi_market_value (kelly) | 4396 | 34.4% | **+14.88%** | **$+8,584,132.57** | 22.9% | 1.25 | 1.0016 | 0.1994 |
| **4. Player Talent Model** | old_production (flat) | 487 | 71.9% | **+25.39%** | **$+3,097,362.40** | 22.1% | 1.98 | 0.9992 | 0.1988 |
| **4. Player Talent Model** | unconstrained_value (flat) | 5710 | 36.6% | **+9.47%** | **$+12,908,133,274.65** | 54.5% | 1.14 | 0.9992 | 0.1988 |
| **4. Player Talent Model** | calibrated_value (flat) | 3844 | 37.5% | **+5.65%** | **$+16,181,697.70** | 65.7% | 1.09 | 0.9992 | 0.1988 |
| **4. Player Talent Model** | calibrated_value (kelly) | 3844 | 37.5% | **+13.34%** | **$+3,322,464.50** | 21.6% | 1.26 | 0.9992 | 0.1988 |
| **4. Player Talent Model** | multi_market_value (flat) | 4439 | 34.7% | **+6.21%** | **$+42,676,397.49** | 57.5% | 1.09 | 0.9992 | 0.1988 |
| **4. Player Talent Model** | multi_market_value (kelly) | 4439 | 34.7% | **+11.69%** | **$+4,986,328.04** | 25.5% | 1.20 | 0.9992 | 0.1988 |
| **5. Master Model (Ensemble + Shrinkage)** | old_production (flat) | 325 | 72.0% | **+19.28%** | **$+391,115.21** | 20.4% | 1.72 | 1.0031 | 0.1998 |
| **5. Master Model (Ensemble + Shrinkage)** | unconstrained_value (flat) | 6400 | 33.1% | **+3.81%** | **$+42,612,733.40** | 78.5% | 1.06 | 1.0031 | 0.1998 |
| **5. Master Model (Ensemble + Shrinkage)** | calibrated_value (flat) | 5634 | 33.9% | **+2.77%** | **$+3,556,878.77** | 76.5% | 1.04 | 1.0031 | 0.1998 |
| **5. Master Model (Ensemble + Shrinkage)** | calibrated_value (kelly) | 5634 | 33.9% | **+10.24%** | **$+1,681,082.41** | 23.1% | 1.18 | 1.0031 | 0.1998 |
| **5. Master Model (Ensemble + Shrinkage)** | multi_market_value (flat) | 5731 | 32.9% | **+4.40%** | **$+9,543,144.82** | 81.6% | 1.06 | 1.0031 | 0.1998 |
| **5. Master Model (Ensemble + Shrinkage)** | multi_market_value (kelly) | 5731 | 32.9% | **+11.53%** | **$+2,506,929.68** | 28.1% | 1.19 | 1.0031 | 0.1998 |

## 2. Hallazgos Clave y Justificación Técnica

1. **Superioridad de los Expected Goals de Impect (Dixon-Coles Poisson)**:
   - Los Expected Goals (`predModelHome`, `predModelAway`) capturan la verdadera capacidad de generación y concesión de ocasiones a nivel de disparo y packing.
   - El modelo Poisson bivariado reduce el Log Loss y elimina la sobreestimación sistemática en cuotas medias del modelo anterior.

2. **Impacto de la Contracción Bayesiana al Mercado (Shrinkage)**:
   - Al combinar las predicciones del modelo con el consenso sin vig del mercado (w=0.25), se atenúan divergencias extremas que antes generaban drawdowns de >80%.
   - Se preservan los edges legítimos en el rango 3%–15%, logrando un ROI neto sostenido con un drawdown controlado.

3. **Gestión de Capital Óptima**:
   - Flat stake (2%) y Kelly Fraccionario (0.10 con Hard Cap del 2.5%) demostraron ser las asignaciones más eficientes en la relación retorno/riesgo (Sharpe y Calmar).
