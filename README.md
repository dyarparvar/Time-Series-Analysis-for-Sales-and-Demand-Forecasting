# Time-Series-Analysis-For-Sales-and-Demand-Forecasting
**Darya Yarparvar** | February 2026
______________________________________________________________

## Problem Statement
Small and medium-sized publishers face asymmetric commercial risk: overestimating
demand leads to inventory costs, while underestimating it means missed revenue.
Predicting how long a book will sell well is central to managing this risk, yet remains
difficult in practice, often resulting in overstocking or missed sales. This project uses
Nielsen BookScan data to forecast sales over time, helping publishers make better print
and investment decisions, especially for books with long-term sales potential.

## Dataset
The data were provided by Nielsen BookScan (Table 1).

<img width="1126" height="276" alt="Screenshot 2026-03-04 at 10 11 37" src="https://github.com/user-attachments/assets/82ab9277-1746-4cba-a095-6e60b47bef8c" />

Table 1. Overview of the data (data provided by Nielsen BookScan).

## Data Cleaning
There were no duplicated records. Entries with missing sales volumes were interpreted
as weeks with no sales. To ensure a regular time series, the data were resampled to
equally spaced weekly intervals, with each period ending on Saturday.
Irrelevant features were removed, and the analysis focused on the sales volume
(units_sold) of 61 books that remain actively sold in the market, defined as having
recorded sales beyond 1 July 2024.

## Exploratory Data Analysis
Most of the 61 actively sold books (Table S1) display a classic L-shaped exponential decay.
The initial years represent the active life, characterised by high-volume sales spikes and
high volatility. The later years represent the dormant life, characterised by a stable flat
line with near-zero variance (Fig. 1).

<img width="1122" height="436" alt="Screenshot 2026-03-04 at 10 12 33" src="https://github.com/user-attachments/assets/16abfdcb-e09a-431d-a6e2-a62e5ea1673d" />

Figure 1. A representative actively sold book exhibiting a classic L-shaped exponential decay. The initial years
represent the active life, characterised by high-volume sales spikes and high volatility. During this period the
product is actively purchased, discussed, and potentially discounted. The market is not yet saturated and
the book is likely available in physical bookstores, which encourages impulsive purchases. The later years
represent the dormant life, characterised by a stable flat line with near-zero variance. This happens because
the market has become saturated and the book has moved from the frontlist to backlist. However, even
without a physical stock in shops, sales do not hit true zero due to online shopping. This creates a
low-volume long tail that persists for years after the end of the active life.

Some books exhibit exceptional sales trajectories. The Alchemist has an early period of
high and chaotic sales, followed by decay and then long-term stabilisation at a steady,
low-volatility level, reflecting its transition from a trendy title to a canonical one. The Very
Hungry Caterpillar maintains strong, non-decaying sales with clear seasonal peaks,
driven by a constantly renewing cohort of newborn readers, allowing it to remain an
evergreen title (Fig. 2).

<img width="1128" height="910" alt="Screenshot 2026-03-04 at 10 13 50" src="https://github.com/user-attachments/assets/7332bd20-2f69-42a5-8034-46d4b12dd259" />

Figure 2. Sales trajectories of two books with long-term sales potential, showing stable demand over
multiple years. The Alchemist (ISBN: 9780722532935) shows early chaotic, spike-driven sales followed by
decay and later stabilisation at a persistent, low-volatility baseline, reflecting its transition from trendy to
canonical. The Very Hungry Caterpillar (ISBN: 9780241003008) exhibits sustained sales with strong seasonal
peaks, driven by a continually renewing cohort of newborn readers, allowing it to remain an evergreen title.

<img width="1128" height="1239" alt="Screenshot 2026-03-04 at 10 14 23" src="https://github.com/user-attachments/assets/76c3de69-aaf3-4f3d-8b25-914ec5d0ec90" />

Figure 3. Weekly sales patterns of two standout titles over 12 years. The Alchemist peaks at around 2200
units (2024), with a mean of 528 and median of 508, reflecting a largely stationary trend with predictable
annual gifting spikes and low volatility outside these peaks. The Very Hungry Caterpillar peaks at
approximately 3900 units (late 2015 and late 2023), with a mean of 1349 and median of 1324, showing a
gradually rising trend with growing volatility and an accelerating baseline in recent years (2022-2024). Both
distributions are right-skewed with roughly 1.1% extreme outliers. Occasional zero-sales weeks likely reflect
COVID-19 related temporary retail closures or supply chain disruptions rather than genuine shifts in demand
(Guren, C., McIlroy, T. and Sieck, S., 2021). Both titles recovered quickly and returned to pre-pandemic
trajectories, confirming the disruption was transient, not structural.

For The Alchemist, with seasonal fluctuations that are roughly constant over time, an
additive STL decomposition (Table S1) extracted a strong seasonal component
(strength=0.68) and a moderate trend component (strength=0.49) (Fig. S1). The ADF test
showed that the weekly sales series is stationary (p-value=4.14e-13). Combined with the
observed autocorrelation, this indicates a stable and predictable structure suitable for
modelling with SARIMA.
For The Very Hungry Caterpillar, seasonal peaks increase with the trend and a
multiplicative decomposition would initially seem more appropriate for this weakly
stationary series (ADF test p-value = 0.03). However, STL decomposition showed that the
raw data preserves the underlying structure better, with stronger trend and seasonal
components (0.59 and 0.45) than the log-transformed series (0.28 and 0.12) (Fig. S3,
S9-10), despite the log transformation rendering the series statistically stationary. This
suggests that sales behaviour is closer to additive than multiplicative.
For both titles, the ACF/PACF plots (Fig. S2, S4) and Ljung-Box test on the residuals
indicates that autocorrelation persists and the residuals do not resemble white noise,
suggesting that not all patterns are purely seasonal or trend-driven. STL decomposition
effectively captures the overall structure of the time series but cannot explain all
underlying dynamics.

## Modelling
For time series analysis of The Alchemist and The very Hungry Caterpillar, we used
weekly sales data from 1 January 2012 onwards. A fixed random seed was applied, and
additional measures were taken to ensure reproducibility throughout the analysis.
### SARIMA Forecasting
We applied SARIMA modelling using pmdarima's auto_arima for automated parameter
selection. Auto_arima optimised AR and MA terms through stepwise search minimising
AIC (Table S3).
### XGBoost Forecasting
We used XGBoost with sktime's TransformedTargetForecaster pipeline, which
deseasonalizes the data, removes polynomial trends, and trains on the transformed
residuals using lagged values as features. Forecasts are generated recursively, with grid
search optimising the lookback window length via expanding window cross-validation
(Table S4). COVID-period (World Health Organization, no date) data was linearly
interpolated to avoid distorting the training process with anomalous near-zero sales.
### LSTM Forecasting
We implemented a multi-layer LSTM network for direct multi-step forecasting,
predicting all 32 weeks simultaneously. Keras Tuner optimised network architecture,
optimiser, and learning rate via grid search on chronologically split validation data (Table
S5). Unlike XGBoost, LSTM was applied on raw scaled sequences without explicit
detrending, learning temporal patterns directly from the data.
### Sequential Hybrid Modelling (SARIMA >> LSTM)
We trained a tuned LSTM network on SARIMA residuals to capture patterns the
statistical model missed (Table S6). This two-stage approach combines SARIMA's
strength in capturing linear seasonal patterns with LSTM's ability to learn complex
non-linear residual dependencies. The final forecast is the sum of SARIMA predictions
and LSTM-predicted residuals.
### Weighted Ensemble Modelling (SARIMA + LSTM)
We combined SARIMA and LSTM forecasts using weighted averages, testing weights
from 0 (pure LSTM) to 1 (pure SARIMA). The optimal weight was selected by minimising
validation MAE. This approach leverages the complementary strengths of both models:
SARIMA's interpretability and stable long-term forecasts, and LSTM's flexibility in
capturing non-linear patterns.
### Monthly Data Forecasting:
We aggregated weekly sales to monthly totals (Fig. 4) and applied SARIMA and XGBoost
models to evaluate whether reduced temporal granularity improves forecast accuracy
(Table S7-8).


<img width="1124" height="899" alt="Screenshot 2026-03-04 at 10 15 05" src="https://github.com/user-attachments/assets/5599d438-de03-4bd8-afe0-e06667cc7159" />

Figure 4. Monthly aggregated (blue) and weekly (beige) sales patterns of The Alchemist and The Very
Hungry Caterpillar over 12 years. Monthly aggregation smooths weekly volatility and reduces high-frequency
noise in both series. The Alchemist shows relatively stable baseline sales with irregular spikes, where weekly
data better preserves the timing of short-term changes. The Very Hungry Caterpillar exhibits a strong
long-term upwards trend with clear seasonality, which is retained when monthly aggregated. The
occasional zero-sales months likely reflect temporary closures or supply chain issues due to the COVID-19
pandemic.


## Results
Model performance varies by book characteristics and temporal granularity (Table 2, 3.)

### The Alchemist
The best-performing model is the sequential hybrid approach on weekly data (27.67%
MAPE), achieving marginal improvements over standalone SARIMA (29.75%) (Fig. 5).
Weekly data performed better than monthly aggregation in SARIMA modelling (29.75%
vs 35.57% MAPE), mainly because it provides much more training data, about 623 weekly
observations compared with only 143 monthly ones (Fig. 6).
The very poor performance of monthly XGBoost (59.19% MAPE) further suggests that the
smaller monthly dataset has a low signal-to-noise ratio, with many informative
event-driven spikes smoothed out by aggregation. The poor performance of weekly
XGBoost (43.98% MAPE) also highlights the difficulty of modelling abrupt changes using
tree-based methods at weekly granularity.

<img width="1115" height="481" alt="Screenshot 2026-03-04 at 10 16 01" src="https://github.com/user-attachments/assets/e5860ff7-483d-4eba-9f61-d1577c08c65f" />

Table 2. Comparison of forecasting models for The Alchemist using weekly and monthly aggregated data.
SARIMA was fitted without exogenous variables; the COVID-19 period was linearly interpolated for XGBoost;
and no pre-processing other than scaling was applied for LSTM. Among standalone models, SARIMA
achieved the lowest error on weekly data (MAPE=29.75%), while sequential hybrid modelling (SARIMA
residuals forecasted by LSTM) produced a small improvement (MAPE=27.67%). Monthly aggregation
reduced performance for both SARIMA and XGBoost, indicating that weekly data better capture the sales
dynamics of this book. [Colour scales are matched to allow direct comparison between both titles and
temporal granularities.]

<img width="1043" height="550" alt="Screenshot 2026-03-04 at 10 16 23" src="https://github.com/user-attachments/assets/8bfa5201-58f4-46df-9d20-c6d785620aae" />

Figure 5. Weekly sales forecasting performance for The Alchemist, comparing observed units sold (turquoise)
with sequential SARIMA >> LSTM forecasts (pink). The model captures the stable baseline and long-term
demand stabilisation but struggles with abrupt changes. It strongly underestimates the large demand spike
in late 2023 and consistently overestimates troughs during April-May. These persistent errors indicate
difficulty adapting to high-frequency volatility and the decaying trend. Overall accuracy (MAE=150.33,
MAPE=28.03%) suggests limited predictive power, with the model favouring smooth, average behaviour
rather than short-term spikes and sudden drops.

<img width="964" height="1118" alt="Screenshot 2026-03-04 at 10 16 45" src="https://github.com/user-attachments/assets/bb3dd85c-0179-47b1-b4bc-17d00ac3b8c7" />

Figure 6. Comparison of weekly (top) and monthly (bottom) sales forecasting for The Alchemist using
SARIMA. On weekly data (MAPE=29.75%), the model captures the average level but heavily smooths the
series, missing the large spike in late 2023 and the volatility in early 2024. Monthly aggregation leads to worse
performance (MAPE=35.57%), as the model struggles to adjust to the sharp drop after January. Overall,
SARIMA is conservative at both granularities. It captures basic seasonality but responds slowly to sudden
changes and fails to model abrupt shifts in the sales pattern.


### The Very Hungry Caterpillar
The best overall forecast, multiplicative XGBoost on monthly data, captures the overall
level and medium-term trend but it remains conservative, and underestimates extreme
fluctuations (Fig. 7).
Monthly aggregation improved multiplicative XGBoost performance (16.53% vs 21.13%
MAPE) because this model is more sensitive to erratic weekly noise and the dominant
signal in the data is a strong, persistent upward trend rather than week-to-week
fluctuations (Fig. S4). Aggregation smooths short-term irregularities, emphasises the
underlying trend and seasonality, and reduces the adverse influence of noise on
forecasting performance.
Multiplicative deseasonalisation performed better on monthly data because seasonal
amplitude increases with the level of sales and larger seasonal swings are easier to
capture in the smoother, aggregated series (Fig. 8).
Sequential hybrid modeling achieved the best performance on weekly sales data with
marginal improvements over standalone SARIMA (18.15% vs 18.63% MAPE).


<img width="1113" height="528" alt="Screenshot 2026-03-04 at 10 17 02" src="https://github.com/user-attachments/assets/8120a311-242b-48e0-bc9d-4d7d6ced38b7" />

Table 3. Comparison of forecasting models for The Very Hungry Caterpillar using weekly and monthly
aggregated data. SARIMA was fitted without exogenous variables; the COVID-19 period was linearly
interpolated for XGBoost; and no pre-processing other than scaling was applied for LSTM. All models
performed well on weekly data, with the sequential hybrid model (MAPE=18.15%) and SARIMA
(MAPE=18.63%) giving the lowest errors. Monthly aggregation preserved similar accuracy for SARIMA
(MAPE=20.26%) and improved XGBoost performance under multiplicative deseasonalisation (MAPE=16.53%),
indicating stronger and more persistent monthly patterns. The superior performance of multiplicative
XGBoost on monthly data reflects the growing seasonal amplitude of sales. [Colour scales are matched to
allow direct comparison between both titles and temporal granularities.]

<img width="1123" height="518" alt="Screenshot 2026-03-04 at 10 17 25" src="https://github.com/user-attachments/assets/8be96aae-6c58-4ea2-bf98-1c04cbf45871" />

Figure 7. Weekly sales forecasting performance for The Very Hungry Caterpillar, comparing observed units
sold (turquoise) with sequential SARIMA >> LSTM forecasts (pink). The model captures the overall level and
medium-term trend but substantially smooths short-term volatility. The model underestimates sharp peaks
and overestimates troughs. Large deviations around early January, March, and May highlight difficulties in
modelling shocks at weekly granularity. Overall accuracy (MAE=357.56, MAPE=18.15%) indicates moderate
predictive performance, suggesting that while the hybrid approach learns persistent structure, it struggles
with high-frequency fluctuations and extreme changes inherent in the weekly series.

<img width="1031" height="1116" alt="Screenshot 2026-03-04 at 10 17 46" src="https://github.com/user-attachments/assets/719d454e-2b8d-434e-96ae-2e2b076bf959" />

Figure 8. Comparison of weekly (top) and monthly (bottom) sales forecasting for The Very Hungry Caterpillar
using multiplicative XGBoost. In the weekly series (MAPE=21.13%), the model identifies the baseline but
suffers from significant smoothness. It fails to track the sharp peaks in the weeks leading up to the end of
2023 and March/April. Transitioning to monthly aggregation improves overall relative accuracy
(MAPE=16.53%) by filtering out short-term noise. However, the model still fails to capture the full magnitude
of the April peak. This suggests that XGBoost effectively learns the medium-term trend, but it remains
conservative, and underestimates extreme fluctuations across both granularities.

### SARIMA - Effective and Reliable
The standalone SARIMA model, which is in fact the same model used as the first step of
sequential hybrid modelling, sheds light on the behaviour of each time series (Fig. 9-12).
Across both titles, weekly models capture strong seasonal dependence and persistence
but exhibit pronounced non-normality and heteroskedasticity driven by sharp sales
spikes. Monthly aggregation smooths these extremes, improving residual normality and
reducing kurtosis, while some degree of time-varying variance still remains.

<img width="1126" height="630" alt="Screenshot 2026-03-04 at 10 18 23" src="https://github.com/user-attachments/assets/eb7c9654-5953-4d6e-a253-fa53c49052f5" />

Figure 9. SARIMA(1,1,2)(2,0,[ ],52) model output for weekly sales of The Alchemist (623 observations). The
model captures strong short-term momentum (high ar.L1 coefficient) and clear annual seasonality
(significant seasonal dependence at lags 52 and 104), indicating that weekly sales are strongly influenced by
both recent demand and recurring yearly patterns. The first MA term (high and negative ma.L1) shows a
strong negative response, indicating that a positive shock last week tends to be overcorrected this week. The
Ljung-Box test confirms no remaining autocorrelation in residuals (Q=0.04, p=0.84), while the Jarque-Bera
test rejects normality (JB=1329.94, p=0.00) with high kurtosis (10.08), reflecting occasional extreme spikes in
the sales data. Heteroskedasticity test shows evidence of non-constant variance (H=2.95, p=0.00).

<img width="1124" height="562" alt="Screenshot 2026-03-04 at 10 18 54" src="https://github.com/user-attachments/assets/b8769023-e991-43db-a9ab-424c1fd519e8" />

Figure 10. SARIMA(1,1,1)(0,1,1,12) model output for monthly aggregated sales of The Alchemist (143
observations). The model has lower non-seasonal persistence than weekly data (lower ar.L1 coefficient),
reflecting smoother and more stable sales. The significant seasonal component at lag 12 shows a clear
annual seasonality. Lower ma.L1 reflects that aggregation smooths shocks and errors propagate less strongly.
The Ljung-Box test shows no remaining autocorrelation in residuals (Q=0.06, p=0.81), and the Jarque-Bera
test does not reject normality (JB=4.84, p=0.09), indicating monthly aggregation successfully smooths
extreme weekly spikes into a more normally distributed series. Heteroskedasticity remains present (H=2.69,
p=0.00), though less severe than weekly data, with moderate kurtosis (3.86) suggesting reduced extreme
values compared to weekly observations.

<img width="1125" height="602" alt="Screenshot 2026-03-04 at 10 19 17" src="https://github.com/user-attachments/assets/63ee16de-6d43-49f2-93d2-fcb689abaf96" />

Figure 11. SARIMA(2,1,1)(1,0,[1,2],52) model output for weekly sales of The Very Hungry Caterpillar (623
observations). The model highlights strong short-term momentum (high ar.L1 coefficient) and dominant
annual seasonality (significant seasonal effects at lag 52), consistent with sharp, recurring sales peaks and
abrupt demand shifts. The first MA term (high and negative ma.L1) shows a strong negative response,
indicating that a positive shock last week tends to be overcorrected this week. The Ljung-Box test confirms
no significant residual autocorrelation (Q=0.01, p=0.94), while the Jarque-Bera test strongly rejects normality
(JB=5951.40, p=0.00) with high kurtosis (18.05), indicating frequent large spikes. Heteroskedasticity is present
(H=5.46, p=0.00), suggesting variance changes over time.

<img width="1124" height="576" alt="Screenshot 2026-03-04 at 10 19 34" src="https://github.com/user-attachments/assets/94422a68-7be6-4fee-be59-0e15b34453d0" />

Figure 12. SARIMA(0,1,3)(0,1,[1],12) model output for monthly aggregated sales of The Very Hungry Caterpillar
(143 observations). The dominance of moving-average dynamics suggests that monthly sales dynamics are
primarily driven by short-term shocks. A significant seasonal component at lag 12 indicates a strong annual
seasonality. Lower ma.L1 reflects that aggregation smooths shocks and errors propagate less strongly. The
Ljung-Box test confirms no residual autocorrelation (Q=0.00, p=0.96), indicating great model fit. The
Jarque-Bera test strongly rejects normality (JB=28.89, p=0.00) with negative skew (-0.75) reflecting the
persistent upward trend and occasional large seasonal spikes even in monthly aggregated data.
Heteroskedasticity is present (H=2.74, p=0.00), though less severe than weekly data, with moderate kurtosis
(4.75) suggesting reduced extreme values compared to weekly observations.

### LSTM - Underperforming
For both titles, LSTM underperformed compared to other methods, behaving
conservatively and systematically underestimating peak weeks (Fig. S7). This is likely due
to limited training data, relying solely on the neural network’s advanced learning
capabilities, and skipping data pre-processing or feature engineering.
While the sequential hybrid method is the superior approach for both titles, the
weighted ensemble models' performance is adversely affected by the inclusion of LSTM
(Table2, 3) (Fig. S8). 
 


## Conclusion
The two titles analysed here represent a rare case of long-term forecasting for books
with lasting demand. The Alchemist, now at its stable canonical position, exhibits a
stable, low-volatility baseline with predictable annual gifting spikes. The Very Hungry
Caterpillar’s steady upward trend reflects demographic renewal, sustaining demand
independently of trends or media. This makes the series structurally more forecastable
in the long term, despite higher short-term volatility.
Sales shocks in The Alchemist and The Very Hungry Caterpillar are driven by external
events rather than underlying demand patterns. Peaks reflect media attention
(Wikipedia, 2026), gifting periods, or cultural and family-oriented events (National Trust,
2026), while dips are linked to COVID-19-related retail closures or supply-chain
disruptions (Guren, C., McIlroy, T. and Sieck, S., 2021). Because these drivers lie outside the
sales history, the models captured seasonality and trend well but failed on event-driven
spikes. This is a core limitation of endogenous time-series models, which cannot
anticipate shocks external to the data.
SARIMA emerged as the most reliable standalone model, capturing seasonality and
short-term autocorrelation without requiring complex feature engineering. The
sequential hybrid approach provided marginal but consistent improvements, meaning
that the data is well-described by linear seasonal structure, with only limited additional
complexity that a neural network component can recover.
The impact of temporal granularity varied by title and model, showing that granularity
interacts with both series structure and model assumptions rather than being
universally beneficial.
### Limitations and Recommendations
- The analysis scope should be widened. Studying all titles and grouping them by
sales pattern (early drop-off, long-term stability, or steady renewal) would make
the results more reliable and easier to generalise.

- External data should be integrated. Demand shocks driven by exogenous events
cannot be anticipated from historical sales patterns alone. Adding these signals to
models like SARIMAX or XGBoost would improve forecasts.

- Structural breaks require proper handling. Treating COVID-19 as a binary
exogenous variable or with a simple interpolation is a pragmatic simplification.
Modelling the pandemic period explicitly would make the system more robust to
similar future disruptions.

- Paying closer attention to the nuances of selected methods would improve
accuracy. XGBoost cannot extrapolate beyond its training range, yet it can offer
untapped potential through feature engineering.

- Future work should include probabilistic forecasts. The decision of how many
copies to print is asymmetric as running out of stock costs differently than having
excess stock. With a probability distribution over future demand, a publisher can
make a rational stocking decision based on their own cost structure.


## References
Guren, C., McIlroy, T. and Sieck, S. (2021). ‘COVID-19 and Book Publishing: Impacts and Insights for 2021’, Publishing Research Quarterly, 37(1), pp.1–14. doi:https://doi.org/10.1007/s12109-021-09791-z.

National Trust (2026) The Very Hungry CaterpillarTM trail. Available at: https://www.nationaltrust.org.uk/visit/london/osterley-park-and-house/events/916ad08a-dd87-4b01-a2c8-7f4a323e7790?utm_source=chatgpt.com (Accessed: 01 March 2026).

Wikipedia (2026) The Alchemist (novel). Available at: https://en.wikipedia.org/wiki/The_Alchemist_%28novel%29?utm_source=chatgpt.com
(Accessed: 01 March 2026).

Wikipedia (2026) The Very Hungry Caterpillar Show. Available at: https://en.wikipedia.org/wiki/The_Very_Hungry_Caterpillar_Show?utm_source=chatgpt.com (Accessed: 01 March 2026).

World Health Organization (no date) Coronavirus disease (covid-19) pandemic. Available at: https://www.who.int/europe/emergencies/situations/covid-19 (Accessed: 01 March 2026).  


_____________________________________________________________________________________________________
Word count (main text only, excluding cover, tables, figures, captions, and references): ~ 1650 words
