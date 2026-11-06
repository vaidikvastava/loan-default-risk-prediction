-- =====================================================================
-- Enterprise MySQL Database Schema for Loan Default Risk Management
-- Database: loan_risk_db
-- =====================================================================

CREATE DATABASE IF NOT EXISTS loan_risk_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE loan_risk_db;

-- 1. Applicants Master Table
CREATE TABLE IF NOT EXISTS applicants (
    sk_id_curr INT PRIMARY KEY,
    target TINYINT DEFAULT NULL COMMENT '1 = Defaulted, 0 = Repaid, NULL = Pending Application',
    name_contract_type VARCHAR(50),
    code_gender VARCHAR(10),
    flag_own_car CHAR(1),
    flag_own_realty CHAR(1),
    cnt_children INT DEFAULT 0,
    amt_income_total DECIMAL(15, 2) NOT NULL,
    amt_credit DECIMAL(15, 2) NOT NULL,
    amt_annuity DECIMAL(15, 2),
    amt_goods_price DECIMAL(15, 2),
    name_type_suite VARCHAR(50),
    name_income_type VARCHAR(50),
    name_education_type VARCHAR(100),
    name_family_status VARCHAR(50),
    name_housing_type VARCHAR(50),
    days_birth INT,
    days_employed INT,
    days_registration INT,
    days_id_publish INT,
    own_car_age DECIMAL(5, 1),
    region_rating_client INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_target (target),
    INDEX idx_income_credit (amt_income_total, amt_credit)
) ENGINE=InnoDB;

-- 2. External Credit Bureau Ratings
CREATE TABLE IF NOT EXISTS credit_bureau_ratings (
    sk_id_curr INT PRIMARY KEY,
    ext_source_1 DECIMAL(6, 4),
    ext_source_2 DECIMAL(6, 4),
    ext_source_3 DECIMAL(6, 4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (sk_id_curr) REFERENCES applicants(sk_id_curr) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 3. Derived Financial Ratios (Engineered Features)
CREATE TABLE IF NOT EXISTS financial_ratios (
    sk_id_curr INT PRIMARY KEY,
    debt_to_income DECIMAL(8, 4),
    annuity_to_income DECIMAL(8, 4),
    credit_to_goods_ratio DECIMAL(8, 4),
    credit_annuity_ratio DECIMAL(8, 4),
    income_per_family_member DECIMAL(15, 2),
    employed_to_age_ratio DECIMAL(6, 4),
    ext_source_mean DECIMAL(6, 4),
    ext_source_min DECIMAL(6, 4),
    ext_source_max DECIMAL(6, 4),
    ext_source_std DECIMAL(6, 4),
    FOREIGN KEY (sk_id_curr) REFERENCES applicants(sk_id_curr) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 4. Underwriting Predictions & Decision Audit Log
CREATE TABLE IF NOT EXISTS underwriting_decisions (
    decision_id INT AUTO_INCREMENT PRIMARY KEY,
    sk_id_curr INT,
    model_version VARCHAR(50) DEFAULT 'v1.0.0',
    default_probability DECIMAL(5, 4) NOT NULL,
    risk_tier ENUM('LOW RISK', 'MODERATE RISK', 'HIGH RISK') NOT NULL,
    recommendation ENUM('APPROVE', 'MANUAL REVIEW', 'DECLINE') NOT NULL,
    decision_threshold DECIMAL(4, 3) NOT NULL,
    key_drivers JSON,
    underwriter_notes TEXT,
    decision_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_rec (recommendation),
    INDEX idx_risk_tier (risk_tier),
    INDEX idx_timestamp (decision_timestamp),
    FOREIGN KEY (sk_id_curr) REFERENCES applicants(sk_id_curr) ON DELETE SET NULL
) ENGINE=InnoDB;

-- 5. Business KPI Analytical View
CREATE OR REPLACE VIEW view_portfolio_risk_summary AS
SELECT 
    a.name_contract_type,
    a.name_education_type,
    COUNT(a.sk_id_curr) AS total_applicants,
    AVG(a.amt_credit) AS avg_credit_amount,
    AVG(a.amt_income_total) AS avg_income,
    AVG(CASE WHEN a.target IS NOT NULL THEN a.target ELSE NULL END) * 100 AS historical_default_rate_pct,
    AVG(r.debt_to_income) AS avg_debt_to_income,
    AVG(r.ext_source_mean) AS avg_bureau_score
FROM applicants a
LEFT JOIN financial_ratios r ON a.sk_id_curr = r.sk_id_curr
GROUP BY a.name_contract_type, a.name_education_type;
