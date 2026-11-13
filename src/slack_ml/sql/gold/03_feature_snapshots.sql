CREATE OR REPLACE TABLE gold.feature_snapshots AS
WITH data_range AS (
    SELECT MIN(posted_date) AS first_date, MAX(posted_date) AS last_date
    FROM silver.messages
    WHERE is_activity
),
bounds AS (
    SELECT
        last_date + 1 AS scoring_date,
        last_date + 1 - CAST(7 * (date_diff('day', first_date + 56, last_date + 1) // 7) AS INTEGER)
            AS first_snapshot
    FROM data_range
),
snapshots AS (
    SELECT CAST(t.s AS DATE) AS snapshot_date, b.scoring_date
    FROM bounds AS b,
        generate_series(b.first_snapshot, b.scoring_date, INTERVAL 7 DAY) AS t(s)
),
own AS (
    SELECT *
    FROM gold.user_events
    WHERE event_type IN ('post', 'reply', 'reaction_given')
),
population AS (
    SELECT DISTINCT s.snapshot_date, s.scoring_date, u.user_id, u.register_date
    FROM snapshots AS s
    JOIN own AS o
        ON o.event_date >= s.snapshot_date - 28
        AND o.event_date < s.snapshot_date
    JOIN silver.users AS u
        ON u.user_id = o.user_id
        AND u.register_date <= s.snapshot_date - 7
),
ev AS (
    SELECT
        p.snapshot_date,
        p.user_id,
        e.event_type,
        e.counterpart_type,
        e.counterpart_id,
        e.channel_id,
        e.event_date,
        date_diff('day', e.event_date, p.snapshot_date) AS age
    FROM population AS p
    JOIN gold.user_events AS e
        ON e.user_id = p.user_id
        AND e.event_date < p.snapshot_date
        AND e.event_date >= p.snapshot_date - 56
),
agg AS (
    SELECT
        snapshot_date,
        user_id,
        COUNT(*) FILTER (WHERE event_type = 'post' AND age <= 7) AS posts_7d,
        COUNT(*) FILTER (WHERE event_type = 'post' AND age <= 28) AS posts_28d,
        COUNT(*) FILTER (
            WHERE event_type = 'reply' AND counterpart_type = 'member' AND age <= 28
        ) AS replies_to_members_28d,
        COUNT(*) FILTER (WHERE event_type = 'reply' AND counterpart_type = 'self' AND age <= 28) AS self_replies_28d,
        COUNT(*) FILTER (
            WHERE event_type = 'reaction_given' AND counterpart_type = 'member' AND age <= 28
        ) AS reactions_to_members_28d,
        COUNT(*) FILTER (
            WHERE event_type IN ('reply', 'reaction_given') AND counterpart_type = 'bot' AND age <= 28
        ) AS bot_interactions_28d,
        COUNT(*) FILTER (WHERE event_type IN ('post', 'reply', 'reaction_given') AND age <= 28) AS actions_28d,
        COUNT(*) FILTER (WHERE event_type IN ('post', 'reply', 'reaction_given') AND age > 28) AS actions_prev_28d,
        COUNT(*) FILTER (WHERE event_type = 'reply_received' AND age <= 28) AS replies_received_28d,
        COUNT(*) FILTER (WHERE event_type = 'reaction_received' AND age <= 28) AS reactions_received_28d,
        COUNT(DISTINCT event_date) FILTER (
            WHERE event_type IN ('post', 'reply', 'reaction_given') AND age <= 28
        ) AS active_days_28d,
        COUNT(DISTINCT counterpart_id) FILTER (
            WHERE counterpart_type = 'member' AND age <= 28
        ) AS distinct_partners_28d,
        COUNT(DISTINCT channel_id) FILTER (WHERE event_type IN ('post', 'reply') AND age <= 28) AS channels_28d,
        MIN(age) FILTER (WHERE event_type IN ('post', 'reply')) AS days_since_last_post,
        MIN(age) FILTER (WHERE event_type IN ('post', 'reply', 'reaction_given')) AS days_since_last_action
    FROM ev
    GROUP BY snapshot_date, user_id
),
own_messages AS (
    SELECT p.snapshot_date, p.user_id, m.channel_id, m.ts, m.text_len, m.is_reply
    FROM population AS p
    JOIN silver.messages AS m
        ON m.user_id = p.user_id
        AND m.is_activity
        AND m.posted_date >= p.snapshot_date - 28
        AND m.posted_date < p.snapshot_date
),
message_stats AS (
    SELECT
        o.snapshot_date,
        o.user_id,
        AVG(o.text_len) AS avg_text_len_28d,
        COUNT(*) FILTER (
            WHERE NOT o.is_reply
            AND NOT EXISTS (
                SELECT 1
                FROM silver.messages AS r
                WHERE r.channel_id = o.channel_id
                    AND r.thread_ts = o.ts
                    AND r.is_activity
                    AND r.user_id <> o.user_id
                    AND r.posted_date < o.snapshot_date
            )
        ) AS posts_without_reply_28d
    FROM own_messages AS o
    GROUP BY o.snapshot_date, o.user_id
),
onboarding AS (
    SELECT
        p.snapshot_date,
        p.user_id,
        COUNT(*) FILTER (WHERE e.event_type IN ('post', 'reply')) AS first_week_posts,
        COUNT(*) FILTER (WHERE e.event_type IN ('reply_received', 'reaction_received')) AS first_week_feedback
    FROM population AS p
    JOIN gold.user_events AS e
        ON e.user_id = p.user_id
        AND e.event_date >= p.register_date
        AND e.event_date < p.register_date + 7
    GROUP BY p.snapshot_date, p.user_id
),
labels AS (
    SELECT
        p.snapshot_date,
        p.user_id,
        CASE
            WHEN p.snapshot_date + 28 > p.scoring_date THEN NULL
            WHEN EXISTS (
                SELECT 1
                FROM own AS o
                WHERE o.user_id = p.user_id
                    AND o.event_date >= p.snapshot_date
                    AND o.event_date < p.snapshot_date + 28
            ) THEN 0
            ELSE 1
        END AS churned
    FROM population AS p
)
SELECT
    p.snapshot_date,
    p.user_id,
    date_diff('day', p.register_date, p.snapshot_date) AS tenure_days,
    COALESCE(a.posts_7d, 0) AS posts_7d,
    COALESCE(a.posts_28d, 0) AS posts_28d,
    COALESCE(a.replies_to_members_28d, 0) AS replies_to_members_28d,
    COALESCE(a.self_replies_28d, 0) AS self_replies_28d,
    COALESCE(a.reactions_to_members_28d, 0) AS reactions_to_members_28d,
    COALESCE(a.bot_interactions_28d, 0) AS bot_interactions_28d,
    COALESCE(a.actions_28d, 0) AS actions_28d,
    COALESCE(a.actions_prev_28d, 0) AS actions_prev_28d,
    (COALESCE(a.actions_28d, 0) + 1.0) / (COALESCE(a.actions_prev_28d, 0) + 1.0) AS activity_trend,
    COALESCE(a.replies_received_28d, 0) AS replies_received_28d,
    COALESCE(a.reactions_received_28d, 0) AS reactions_received_28d,
    (COALESCE(a.replies_received_28d, 0) + COALESCE(a.reactions_received_28d, 0))
        / greatest(COALESCE(a.posts_28d, 0), 1) AS feedback_per_post_28d,
    COALESCE(m.posts_without_reply_28d, 0) AS posts_without_reply_28d,
    COALESCE(m.posts_without_reply_28d, 0) / NULLIF(COALESCE(a.posts_28d, 0), 0) AS unreplied_share_28d,
    COALESCE(a.active_days_28d, 0) AS active_days_28d,
    COALESCE(a.distinct_partners_28d, 0) AS distinct_partners_28d,
    COALESCE(a.channels_28d, 0) AS channels_28d,
    COALESCE(a.days_since_last_post, 57) AS days_since_last_post,
    COALESCE(a.days_since_last_action, 57) AS days_since_last_action,
    COALESCE(m.avg_text_len_28d, 0) AS avg_text_len_28d,
    COALESCE(o.first_week_posts, 0) AS first_week_posts,
    COALESCE(o.first_week_feedback, 0) AS first_week_feedback,
    l.churned
FROM population AS p
LEFT JOIN agg AS a USING (snapshot_date, user_id)
LEFT JOIN message_stats AS m USING (snapshot_date, user_id)
LEFT JOIN onboarding AS o USING (snapshot_date, user_id)
LEFT JOIN labels AS l USING (snapshot_date, user_id)
ORDER BY p.snapshot_date, p.user_id;
