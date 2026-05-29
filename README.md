# Telecom Customer Churn

[![Data Pipeline](https://img.shields.io/badge/Data_Pipeline-Production--Ready-brightgreen)](https://github.com/omarwael-24/telecom-customer-churn-prediction)
[![Data Stack](https://img.shields.io/badge/Data_Stack-Pandas_%7C_NumPy-blue)](https://github.com/omarwael-24/telecom-customer-churn-prediction)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-darkgreen)](https://github.com/omarwael-24/telecom-customer-churn-prediction)
[![Code Style](https://img.shields.io/badge/Code_Style-PEP8-orange)](https://github.com/omarwael-24/telecom-customer-churn-prediction)

This repository contains the core data engineering and feature refinement pipeline designed to process telecom subscriber logs. The architecture focuses strictly on transformation pipelines, isolating statistical outliers, and engineering behavioral velocity metrics from raw operational data before any downstream consumption.

---

## 1. System Architecture & Data Flow

To ensure production stability, the raw input data is processed through a modular, decoupled pipeline. Below is the technical architecture mapping the lifecycle of telemetry records from ingestion to refined feature states:

### Pipeline Architecture
<img width="1080" height="110" alt="image" src="https://github.com/user-attachments/assets/6a0244df-e806-4e83-88f2-ca59777ae98c" />

### Architectural Core Pillars:
1. **Data Ingestion & Structural Formatting:** Schema stabilization, row filtering, and administrative feature pruning.
2. **Statistical Noise Mitigation:** Robust handling of data anomalies using interquartile range (IQR) bounds and sequential imputation rules.
3. **Behavioral Synthesis (Feature Engineering):** Mathematical transformation of static fields into dynamic consumption velocity indicators.
4. **Deterministic Feature Encoding:** Strategic conversion of high-cardinality strings and ordinal tiers into linear-friendly numeric profiles.

---

## 2. Detailed Technical Phases

### Phase A: Data Cleansing & Robust Filtering
To ensure production stability, the raw input data undergoes strict structural filtering:

* **Administrative Feature Pruning:** Features representing internal database codes or administrative demographic records with zero correlation to usage behaviors (such as `ethnic`, `infobase`, `HHstatin`, `dwllsize`, and `dwlltype`) are programmatically excluded to optimize memory usage and limit dimensionality.
* **Subscriber Baseline Stabilization:** Hard thresholds are set on active sub-counts (`uniqsubs <= 15`) to isolate standard consumer and family accounts from corporate outliers.
* **Extreme Outlier Isolation:** Standard usage and revenue columns frequently exhibit heavy-tailed distributions. The pipeline applies a strict **3.0x IQR (Interquartile Range)** filtering strategy on variables like `rev_Mean`, `mou_Mean`, `eqpdays`, and `change_mou` to isolate severe data corruption or hardware errors while preserving valid operational patterns.
* **Deterministic Imputation Layer:** To avoid data leakage, missing fields are handled sequentially. Numeric metrics are filled via localized operational medians. Any remaining unmapped attributes default to a defensive value of zero, ensuring that the pipeline outputs zero missing values.

### Phase B: Advanced Behavioral Feature Engineering
Static ledger variables only show historic states. To model the real-time degradation of a subscriber relationship, the pipeline constructs four dynamic behavioral indicators:

* **Minutes of Use (MOU) Drop Velocity:** Calculates the rate of consumption decline by measuring the directional change in monthly usage normalized by the customer's historical baseline.
* **Network Failure Index:** A composite quality metric that evaluates network friction by combining dropped voice calls and blocked data sessions relative to the user's active session footprint.
* **Hardware Lifecycle Score:** Maps physical equipment degradation by analyzing the continuous age of the active handset against the lifespan of the account contract.
* **Cost Per Minute (CPM) Rate:** Represents financial value perception by mapping active billing charges directly against actual continuous network utilization time.

### Phase C: Categorical Encoding Tactics
Alphanumeric string variables are stabilized into numerical formats using specific engineering methods based on feature type:

* **One-Hot Encoding:** Low-cardinality indicators (such as `new_cell`, `asl_flag`, `dualband`, and `refurb_new`) are encoded using Pandas `get_dummies` with `drop_first=True` and cast explicitly to standard integers to eliminate multi-collinearity.
* **High-Certainty Target Encoding:** High-cardinality alphanumeric categories—such as the regional Credit Class Code (`crclscod`)—are stripped down to their primary structural risk character and mapped directly to historical statistical target weights. This turns noisy categoricals into continuous financial indicators.
* **Ordinal Subscriber Mapping:** Tiered behavioral brackets (such as account subscriber density ranges) are explicitly binned and ordered using ranked structural integers via an `OrdinalEncoder` setup.

---
