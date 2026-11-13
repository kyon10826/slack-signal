CREATE OR REPLACE TABLE silver.users AS
WITH members AS (
    SELECT *
    FROM bronze.users
    WHERE user_id IS NOT NULL
      AND NOT is_bot
    QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY user_id) = 1
),
first_message AS (
    SELECT
        user_id,
        MIN(ts_epoch) FILTER (WHERE subtype = 'channel_join') AS first_join_epoch,
        MIN(ts_epoch) AS first_message_epoch
    FROM bronze.messages
    GROUP BY user_id
),
first_reaction AS (
    SELECT r.user_id, MIN(m.ts_epoch) AS first_reaction_epoch
    FROM bronze.reactions AS r
    JOIN bronze.messages AS m ON m.channel_id = r.channel_id AND m.ts = r.message_ts
    GROUP BY r.user_id
),
first_seen AS (
    SELECT
        u.*,
        COALESCE(f.first_join_epoch, least(f.first_message_epoch, r.first_reaction_epoch)) AS first_epoch
    FROM members AS u
    LEFT JOIN first_message AS f USING (user_id)
    LEFT JOIN first_reaction AS r USING (user_id)
)
SELECT
    user_id,
    name,
    real_name,
    display_name,
    deleted,
    CAST(to_timestamp(first_epoch) AT TIME ZONE 'Asia/Tokyo' AS DATE) AS register_date
FROM first_seen;
