-- Project 1: Customer Churn Prediction & LTV Engine
-- customer_master view
--
-- Purpose: single source of truth for the FastAPI service (Week 3) so the
-- backend doesn't have to join 3 tables on every request. Combines:
--   - telco_churn_feature_engineered  (customer attributes + engineered features)
--   - customer_churn_predictions      (churn_probability, churn_risk)
--   - customer_ltv_segments           (predicted_ltv, customer_segment)
--
-- Run this in pgAdmin's Query Tool against the telco_churn database.

CREATE OR REPLACE VIEW customer_master AS
SELECT
    f.*,
    p.churn_probability,
    p.churn_risk,
    l.predicted_ltv,
    l.customer_segment
FROM telco_churn_feature_engineered f
JOIN customer_churn_predictions p ON f."customerID" = p."customerID"
JOIN customer_ltv_segments l      ON f."customerID" = l."customerID";

-- ---------------------------------------------------------------------
-- Verification queries -- run these after creating the view above
-- ---------------------------------------------------------------------

-- Should return 7043 (matches every other table's row count)
SELECT COUNT(*) FROM customer_master;

-- Spot check a couple of known customers
SELECT "customerID", "Contract", tenure, "MonthlyCharges",
       churn_probability, churn_risk, predicted_ltv, customer_segment
FROM customer_master
WHERE "customerID" IN ('7590-VHVEG', '5575-GNVDE');

-- ---------------------------------------------------------------------
-- Sample queries for the Backend Engineer to build the API around
-- ---------------------------------------------------------------------

-- GET /predict/churn/{customerID} -- single customer lookup
-- SELECT "customerID", churn_probability, churn_risk FROM customer_master WHERE "customerID" = %s;

-- GET /predict/ltv/{customerID}
-- SELECT "customerID", predicted_ltv, customer_segment FROM customer_master WHERE "customerID" = %s;

-- GET /priority-list -- the 166 high value / high risk customers
-- SELECT * FROM customer_master WHERE customer_segment = 'High Value - High Risk';

-- GET /predict/batch -- pass a list of customerIDs
-- SELECT * FROM customer_master WHERE "customerID" = ANY(%s);
