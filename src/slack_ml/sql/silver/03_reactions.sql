CREATE OR REPLACE TABLE silver.reactions AS
SELECT DISTINCT
    r.channel_id,
    r.message_ts,
    r.reaction,
    r.user_id AS reactor_id,
    m.user_id AS author_id,
    m.author_type AS target_author_type,
    m.posted_at AS reacted_at_approx,
    m.posted_date AS reacted_date_approx
FROM bronze.reactions AS r
JOIN silver.users AS u ON u.user_id = r.user_id
JOIN silver.messages AS m ON m.channel_id = r.channel_id AND m.ts = r.message_ts;
