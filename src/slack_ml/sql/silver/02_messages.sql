CREATE OR REPLACE TABLE silver.messages AS
WITH dedup AS (
    SELECT *
    FROM bronze.messages
    QUALIFY ROW_NUMBER() OVER (PARTITION BY channel_id, ts ORDER BY source_file) = 1
),
classified AS (
    SELECT
        m.*,
        CASE
            WHEN m.user_id IS NULL THEN 'no_user'
            WHEN u.user_id IS NULL THEN 'unknown'
            WHEN u.is_bot THEN 'bot'
            WHEN m.bot_id IS NOT NULL THEN 'member_app'
            ELSE 'member'
        END AS author_type,
        m.thread_ts IS NOT NULL AND m.thread_ts <> m.ts AS is_reply,
        to_timestamp(m.ts_epoch) AT TIME ZONE 'Asia/Tokyo' AS posted_at
    FROM dedup AS m
    LEFT JOIN bronze.users AS u ON u.user_id = m.user_id
)
SELECT
    c.channel_id,
    c.channel_name,
    c.ts,
    c.user_id,
    c.author_type,
    c.subtype,
    c.author_type = 'member'
        AND COALESCE(c.subtype, '') IN ('', 'thread_broadcast', 'file_share', 'me_message') AS is_activity,
    c.posted_at,
    CAST(c.posted_at AS DATE) AS posted_date,
    c.is_reply,
    CASE WHEN c.is_reply THEN c.thread_ts END AS thread_ts,
    CASE WHEN c.is_reply THEN COALESCE(c.parent_user_id, p.user_id) END AS parent_user_id,
    CASE WHEN c.is_reply THEN p.author_type END AS parent_author_type,
    c.text,
    length(c.text) AS text_len,
    c.source_file
FROM classified AS c
LEFT JOIN classified AS p
    ON c.is_reply AND p.channel_id = c.channel_id AND p.ts = c.thread_ts;
