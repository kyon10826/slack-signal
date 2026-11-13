CREATE OR REPLACE TABLE gold.user_events AS
WITH acts AS (
    SELECT * FROM silver.messages WHERE is_activity
)
SELECT
    user_id,
    posted_at AS event_at,
    posted_date AS event_date,
    'post' AS event_type,
    channel_id,
    ts AS ref_ts,
    CAST(NULL AS VARCHAR) AS counterpart_id,
    CAST(NULL AS VARCHAR) AS counterpart_type
FROM acts
WHERE NOT is_reply

UNION ALL
SELECT
    user_id, posted_at, posted_date, 'reply', channel_id, ts, parent_user_id,
    CASE
        WHEN parent_user_id = user_id THEN 'self'
        WHEN parent_author_type = 'member' THEN 'member'
        WHEN parent_author_type IS NOT NULL THEN 'bot'
    END
FROM acts
WHERE is_reply

UNION ALL
SELECT parent_user_id, posted_at, posted_date, 'reply_received', channel_id, thread_ts, user_id, 'member'
FROM acts
WHERE is_reply
  AND parent_author_type = 'member'
  AND parent_user_id <> user_id

UNION ALL
SELECT
    reactor_id, reacted_at_approx, reacted_date_approx, 'reaction_given', channel_id, message_ts, author_id,
    CASE
        WHEN author_id = reactor_id THEN 'self'
        WHEN target_author_type = 'member' THEN 'member'
        ELSE 'bot'
    END
FROM silver.reactions

UNION ALL
SELECT author_id, reacted_at_approx, reacted_date_approx, 'reaction_received', channel_id, message_ts, reactor_id, 'member'
FROM silver.reactions
WHERE target_author_type = 'member'
  AND author_id <> reactor_id;
