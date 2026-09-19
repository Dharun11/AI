-- Run once as a MySQL admin (root). Replace the password placeholder first.
-- Purpose: the agent never touches the raw legacy table. It sees a clean read-only view
-- and can only INSERT into its own audit log.

USE supply_chain;

-- 1. Semantic view: legacy column names -> clean business names.
--    SQL SECURITY DEFINER lets the RO user read the view without any grant on the base table.
CREATE OR REPLACE SQL SECURITY DEFINER VIEW vw_active_fleet AS
SELECT
    TS_UTC          AS `Timestamp`,
    V_LAT           AS `Latitude`,
    V_LON           AS `Longitude`,
    IOT_TEMP_VAL_C  AS `Current_Temperature_C`,
    CGO_COND_CD     AS `Cargo_Condition_Code`,
    RISK_CLS_TXT    AS `Risk_Classification`,
    DELAY_PROB_DEC  AS `Delay_Probability`,
    PRT_CNG_LVL     AS `Port_Congestion_Level`,
    RT_RSK_IDX      AS `Route_Risk_Index`
FROM supply_chain_legacy;

-- 2. Append-only audit log of every agent step.
CREATE TABLE IF NOT EXISTS agent_audit_log (
    log_id        INT AUTO_INCREMENT PRIMARY KEY,
    logged_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id    VARCHAR(64),
    node_executed VARCHAR(50),
    tool_name     VARCHAR(100),
    content       LONGTEXT
);

-- 3. Least-privilege agent account.
CREATE USER IF NOT EXISTS 'usr_fde_ro'@'localhost' IDENTIFIED BY 'CHANGE_ME_STRONG_PASSWORD';
GRANT SELECT ON supply_chain.vw_active_fleet TO 'usr_fde_ro'@'localhost';
GRANT INSERT ON supply_chain.agent_audit_log TO 'usr_fde_ro'@'localhost';
FLUSH PRIVILEGES;
