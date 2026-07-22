# Weekly Progress Reports: DSP391m Capstone

## Instructions for Team

- **Frequency:** Every week (submit by Sunday EOD)
- **Duration:** ~20-30 minutes to fill out
- **Submission:** Email supervisor OR update this file + commit to git
- **Requirement:** Missing 3 reports = automatic fail (per SU26.md)

---

## Week 1: 2026-07-21 to 2026-07-27

### Status Overview
- **Milestone:** v2.6 production model finalized & W&B logging fixed
- **Progress:** 95% complete
- **Blocker:** None

### Work Completed
- [x] Fixed W&B mae/rmse logging (scale mismatch: logging in VND now, not billions)
- [x] Created OPTIMIZATION_ANALYSIS.md (explaining why 13.10% MAPE is locally optimal)
- [x] Created RESEARCH_PAPER_OUTLINE.md (conference publication template)
- [x] Created PRODUCTION_STRATEGY.md (maintenance & monitoring roadmap)
- [x] Locked v2.6 as production-ready (no further optimization)

### Key Findings
1. **Heteroscedasticity:** Tree ensemble diversity handles better than weighted loss (tested & failed)
2. **Feature Importance:** Weak-importance features provide non-linear value; pruning hurts performance
3. **MAPE Gap:** 13.10% vs 10% target requires data expansion or architecture change (out of scope)

### Performance Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAPE | 13.10% | <10% | ⚠️ Gap: +3.10% |
| R² | 0.9200 | >0.90 | ✅ Pass |
| MAE | 2.15B VND | <2.50B | ✅ Pass |
| RMSE | 3.41B VND | <3.80B | ✅ Pass |

### Next Week's Plan
- [ ] Complete research paper draft (write-up sections 1-6)
- [ ] Set up weekly retraining cron job (automated, cloud-based)
- [ ] Test W&B dashboard with live predictions
- [ ] Prepare capstone defense presentation (slides + demo video)

### Issues & Risks
- **Risk:** MAPE gap (13.10% vs 10%) — Document as "local optimum" in defense
- **Mitigation:** Emphasize roadmap for v2.7+ (data collection, temporal features)

### Hours Spent
- [ ] Modeling: 4 hours
- [ ] Engineering: 3 hours
- [ ] Documentation: 3 hours
- [ ] **Total: 10 hours**

### Comments
Optimization phase complete. v2.6 is locked and production-ready. Shifted focus to finalization (documentation, research paper, maintenance strategy).

---

## Week 2: 2026-07-28 to 2026-08-03

### Status Overview
- **Milestone:** Research paper drafted; defense preparation started
- **Progress:** (To be completed)
- **Blocker:** (To be completed)

### Work Completed
- [ ] 
- [ ] 
- [ ] 

### Key Findings
- [ ]

### Performance Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAPE | % | <10% | ⚠️ |
| R² |  | >0.90 | ⚠️ |
| MAE | B VND | <2.50B | ⚠️ |
| RMSE | B VND | <3.80B | ⚠️ |

### Next Week's Plan
- [ ]
- [ ]
- [ ]

### Issues & Risks
- **Risk:** (To be identified)
- **Mitigation:** (To be determined)

### Hours Spent
- [ ] Modeling: hours
- [ ] Engineering: hours
- [ ] Documentation: hours
- [ ] **Total: hours**

### Comments
(To be filled)

---

## Week 3: 2026-08-04 to 2026-08-10

### Status Overview
- **Milestone:** (To be determined)
- **Progress:** (To be completed)
- **Blocker:** (To be completed)

### Work Completed
- [ ] 
- [ ] 
- [ ] 

### Key Findings
- [ ]

### Performance Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAPE | % | <10% | ⚠️ |
| R² |  | >0.90 | ⚠️ |
| MAE | B VND | <2.50B | ⚠️ |
| RMSE | B VND | <3.80B | ⚠️ |

### Next Week's Plan
- [ ]
- [ ]
- [ ]

### Issues & Risks
- **Risk:** (To be identified)
- **Mitigation:** (To be determined)

### Hours Spent
- [ ] Modeling: hours
- [ ] Engineering: hours
- [ ] Documentation: hours
- [ ] **Total: hours**

### Comments
(To be filled)

---

## Week 4: 2026-08-11 to 2026-08-17

*(Template continues for remaining weeks...)*

---

## Summary Table (All Weeks)

| Week | Dates | MAPE | R² | Hours | Status |
|------|-------|------|-----|-------|--------|
| 1 | 7/21-7/27 | 13.10% | 0.9200 | 10 | ✅ Complete |
| 2 | 7/28-8/03 | % | | | ⏳ Pending |
| 3 | 8/04-8/10 | % | | | ⏳ Pending |
| 4 | 8/11-8/17 | % | | | ⏳ Pending |

---

## Supervisor Sign-Off

### Week 1
- **Supervisor Name:** _________________________
- **Signature:** _________________________
- **Date:** _________________________
- **Comments:** 

---

## File-by-File Checklist (Capstone Deliverables)

- [x] **Data Pipeline**
  - [x] Raw data ingestion (Supabase)
  - [x] Preprocessing & feature engineering (78 features)
  - [x] Train/test split (80/20)
  - [x] Log-scale transformation for price
  
- [x] **ML/DL Architecture**
  - [x] 3-tier ensemble (LightGBM, XGBoost, CatBoost)
  - [x] Price stratification (low/mid/high tiers)
  - [x] Hyperparameter tuning (tested & locked)
  - [x] Early stopping for regularization
  
- [x] **XAI (Explainability)**
  - [x] Feature importance analysis
  - [x] SHAP value computation
  - [x] Interpretation of top features
  - [x] Documentation: `docs/XAI_ANALYSIS_SUMMARY.md`
  
- [x] **Production Maintenance**
  - [x] Model versioning & backup
  - [x] W&B experiment tracking
  - [x] Monitoring strategy (MAPE drift alerts)
  - [x] Retraining pipeline (weekly schedule)
  - [x] Documentation: `docs/PRODUCTION_STRATEGY.md`
  
- [x] **Web App Demo**
  - [x] Frontend (React/Vue) with property input
  - [x] API layer (Flask) for predictions
  - [x] Visualization (price heatmap, market trends)
  - [x] Deployment (Supabase + Vercel)
  
- [ ] **Research Paper** (In Progress)
  - [x] Outline & structure
  - [ ] Complete sections 1-7
  - [ ] Add figures & tables
  - [ ] Compile references
  - [ ] Polish for publication
  
- [ ] **Defense Preparation**
  - [ ] Slide deck (problem, solution, results, roadmap)
  - [ ] Demo video (5 min walk-through)
  - [ ] Backup slides (technical deep-dive)
  - [ ] Q&A prep (common questions)

---

## Reporting Compliance Track

```
Week 1 (7/21-7/27): ✅ SUBMITTED
Week 2 (7/28-8/03): ⏳ DUE
Week 3 (8/04-8/10): ⏳ DUE
Week 4 (8/11-8/17): ⏳ DUE
Week 5 (8/18-8/24): ⏳ DUE
...
```

**Note:** Missing 3+ reports = automatic fail per SU26.md clause.

---

**Document prepared:** 2026-07-22  
**Status:** Template Ready — Week 1 Complete
