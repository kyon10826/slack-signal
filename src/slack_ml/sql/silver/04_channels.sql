CREATE OR REPLACE TABLE silver.channels AS
SELECT channel_id, channel_name, is_general
FROM bronze.channels
QUALIFY ROW_NUMBER() OVER (PARTITION BY channel_id ORDER BY channel_name) = 1;

CREATE OR REPLACE TABLE silver.channel_joins AS
SELECT
    channel_id,
    user_id,
    MIN(posted_date) AS joined_date
FROM silver.messages
WHERE author_type = 'member'
  AND subtype = 'channel_join'
GROUP BY channel_id, user_id;
