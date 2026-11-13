CREATE OR REPLACE TABLE gold.weekly_kpi AS
WITH data_range AS (
    SELECT MIN(posted_date) AS first_date, MAX(posted_date) AS last_date
    FROM silver.messages
    WHERE is_activity
),
weeks AS (
    SELECT CAST(t.w AS DATE) AS week_start
    FROM data_range AS r,
        generate_series(date_trunc('week', r.first_date), date_trunc('week', r.last_date), INTERVAL 7 DAY) AS t(w)
),
own AS (
    SELECT *, CAST(date_trunc('week', event_date) AS DATE) AS week_start
    FROM gold.user_events
    WHERE event_type IN ('post', 'reply', 'reaction_given')
),
weekly_actions AS (
    SELECT
        week_start,
        COUNT(DISTINCT user_id) AS active_members,
        COUNT(DISTINCT user_id) FILTER (WHERE counterpart_type = 'member') AS interacting_members,
        COUNT(*) FILTER (WHERE event_type = 'post') AS posts,
        COUNT(*) FILTER (WHERE event_type = 'reply') AS replies,
        COUNT(*) FILTER (WHERE event_type = 'reply' AND counterpart_type = 'bot') AS replies_to_bot
    FROM own
    GROUP BY week_start
),
active_5plus AS (
    SELECT week_start, COUNT(*) AS active_5plus_28d
    FROM (
        SELECT w.week_start, o.user_id
        FROM weeks AS w
        JOIN own AS o
            ON o.event_type IN ('post', 'reply')
            AND o.event_date BETWEEN w.week_start + 6 - 27 AND w.week_start + 6
        GROUP BY w.week_start, o.user_id
        HAVING COUNT(*) >= 5
    )
    GROUP BY week_start
),
member_posts AS (
    SELECT channel_id, ts, user_id, posted_date, CAST(date_trunc('week', posted_date) AS DATE) AS week_start
    FROM silver.messages
    WHERE is_activity AND NOT is_reply
),
first_reply AS (
    SELECT p.channel_id, p.ts, MIN(r.posted_date) AS first_reply_date
    FROM member_posts AS p
    JOIN silver.messages AS r
        ON r.channel_id = p.channel_id
        AND r.thread_ts = p.ts
        AND r.is_activity
        AND r.user_id <> p.user_id
    GROUP BY p.channel_id, p.ts
),
reply_rate AS (
    SELECT
        p.week_start,
        AVG(CASE WHEN f.first_reply_date <= p.posted_date + 7 THEN 1.0 ELSE 0.0 END) AS reply_rate_7d
    FROM member_posts AS p
    LEFT JOIN first_reply AS f USING (channel_id, ts)
    GROUP BY p.week_start
),
new_members AS (
    SELECT CAST(date_trunc('week', register_date) AS DATE) AS week_start, COUNT(*) AS new_members
    FROM silver.users
    WHERE register_date IS NOT NULL
    GROUP BY 1
)
SELECT
    w.week_start,
    w.week_start < r.first_date OR w.week_start + 6 > r.last_date AS is_partial_week,
    COALESCE(a.active_members, 0) AS active_members,
    COALESCE(a.interacting_members, 0) AS interacting_members,
    CASE WHEN w.week_start + 6 - 27 >= r.first_date THEN COALESCE(f.active_5plus_28d, 0) END AS active_5plus_28d,
    COALESCE(a.posts, 0) AS posts,
    COALESCE(a.replies, 0) AS replies,
    COALESCE(a.replies_to_bot, 0) AS replies_to_bot,
    CASE WHEN w.week_start + 6 + 7 <= r.last_date THEN rr.reply_rate_7d END AS reply_rate_7d,
    COALESCE(n.new_members, 0) AS new_members
FROM weeks AS w
CROSS JOIN data_range AS r
LEFT JOIN weekly_actions AS a ON a.week_start = w.week_start
LEFT JOIN active_5plus AS f ON f.week_start = w.week_start
LEFT JOIN reply_rate AS rr ON rr.week_start = w.week_start
LEFT JOIN new_members AS n ON n.week_start = w.week_start
ORDER BY w.week_start;
