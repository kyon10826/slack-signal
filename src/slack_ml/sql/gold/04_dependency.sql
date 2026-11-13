CREATE OR REPLACE TABLE gold.thread_responses AS
WITH bounds AS (
    SELECT MAX(posted_date) AS last_date FROM silver.messages WHERE is_activity
),
threads AS (
    SELECT m.channel_id, m.ts AS thread_ts, m.user_id AS asker_id
    FROM silver.messages AS m
    CROSS JOIN bounds AS b
    WHERE m.is_activity
      AND NOT m.is_reply
      AND m.posted_date > b.last_date - 90
)
SELECT
    t.channel_id,
    t.thread_ts,
    t.asker_id,
    r.user_id AS responder_id,
    COUNT(*) AS replies
FROM threads AS t
JOIN silver.messages AS r
    ON r.channel_id = t.channel_id AND r.thread_ts = t.thread_ts
WHERE r.is_activity
  AND r.is_reply
  AND r.user_id <> t.asker_id
GROUP BY ALL;

CREATE OR REPLACE TABLE gold.channel_dependency AS
WITH per_thread AS (
    SELECT channel_id, thread_ts, COUNT(*) AS n_responders
    FROM gold.thread_responses
    GROUP BY ALL
),
per_channel AS (
    SELECT channel_id, COUNT(*) AS threads
    FROM per_thread
    GROUP BY ALL
),
per_person AS (
    SELECT
        r.channel_id,
        r.responder_id,
        COUNT(*) AS answered,
        COUNT(*) FILTER (WHERE t.n_responders = 1) AS solo
    FROM gold.thread_responses AS r
    JOIN per_thread AS t USING (channel_id, thread_ts)
    GROUP BY ALL
)
SELECT
    p.channel_id,
    ch.channel_name,
    c.threads,
    COUNT(*) OVER (PARTITION BY p.channel_id) AS responders,
    p.responder_id AS key_person_id,
    p.answered AS key_answered,
    p.solo AS key_solo,
    p.answered / c.threads AS key_share,
    p.solo / c.threads AS solo_share,
    c.threads >= 5 AND p.solo / c.threads > 0.5 AS is_dependent,
    p.solo >= 10 AND p.solo / c.threads <= 0.5 AS is_heavy
FROM per_person AS p
JOIN per_channel AS c USING (channel_id)
LEFT JOIN silver.channels AS ch USING (channel_id)
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY p.channel_id ORDER BY p.solo DESC, p.answered DESC, p.responder_id
) = 1;

CREATE OR REPLACE TABLE gold.person_dependency AS
WITH per_thread AS (
    SELECT channel_id, thread_ts, COUNT(*) AS n_responders
    FROM gold.thread_responses
    GROUP BY ALL
),
per_person AS (
    SELECT
        r.responder_id AS user_id,
        COUNT(DISTINCT r.channel_id) AS channels,
        COUNT(*) AS answered,
        COUNT(*) FILTER (WHERE t.n_responders = 1) AS solo
    FROM gold.thread_responses AS r
    JOIN per_thread AS t USING (channel_id, thread_ts)
    GROUP BY ALL
),
roles AS (
    SELECT
        key_person_id AS user_id,
        COUNT(*) FILTER (WHERE is_dependent) AS dependent_channels,
        COUNT(*) FILTER (WHERE is_heavy) AS heavy_channels
    FROM gold.channel_dependency
    GROUP BY ALL
)
SELECT
    p.user_id,
    p.channels,
    p.answered,
    p.solo,
    p.solo / NULLIF(SUM(p.solo) OVER (), 0) AS solo_share,
    COALESCE(r.dependent_channels, 0) AS dependent_channels,
    COALESCE(r.heavy_channels, 0) AS heavy_channels,
    RANK() OVER (ORDER BY p.solo DESC) AS solo_rank
FROM per_person AS p
LEFT JOIN roles AS r USING (user_id);

CREATE OR REPLACE TABLE gold.backup_candidates AS
WITH bounds AS (
    SELECT MAX(posted_date) AS last_date FROM silver.messages WHERE is_activity
),
dependent AS (
    SELECT channel_id, key_person_id, key_solo FROM gold.channel_dependency WHERE is_dependent
),
active AS (
    SELECT DISTINCT e.user_id
    FROM gold.user_events AS e
    CROSS JOIN bounds AS b
    WHERE e.event_type IN ('post', 'reply', 'reaction_given')
      AND e.event_date > b.last_date - 28
),
present AS (
    SELECT m.channel_id, m.user_id, COUNT(*) AS messages
    FROM silver.messages AS m
    CROSS JOIN bounds AS b
    WHERE m.is_activity AND m.posted_date > b.last_date - 90
    GROUP BY ALL
),
participants AS (
    SELECT channel_id, thread_ts, asker_id AS user_id FROM gold.thread_responses
    UNION
    SELECT channel_id, thread_ts, responder_id FROM gold.thread_responses
),
shared AS (
    SELECT d.channel_id, p.user_id, COUNT(*) AS shared_threads
    FROM dependent AS d
    JOIN participants AS k
        ON k.channel_id = d.channel_id AND k.user_id = d.key_person_id
    JOIN participants AS p
        ON p.channel_id = k.channel_id AND p.thread_ts = k.thread_ts AND p.user_id <> d.key_person_id
    GROUP BY ALL
),
answered AS (
    SELECT channel_id, responder_id AS user_id, COUNT(*) AS answered
    FROM gold.thread_responses
    GROUP BY ALL
),
loaded AS (
    SELECT user_id FROM gold.person_dependency WHERE dependent_channels + heavy_channels > 0
),
pool AS (
    SELECT
        d.channel_id,
        d.key_person_id,
        d.key_solo,
        pr.user_id AS candidate_id,
        COALESCE(a.answered, 0) AS answered,
        COALESCE(s.shared_threads, 0) AS shared_threads,
        pr.messages,
        lo.user_id IS NOT NULL AS is_loaded
    FROM dependent AS d
    JOIN present AS pr ON pr.channel_id = d.channel_id AND pr.user_id <> d.key_person_id
    JOIN active AS ac ON ac.user_id = pr.user_id
    LEFT JOIN answered AS a ON a.channel_id = d.channel_id AND a.user_id = pr.user_id
    LEFT JOIN shared AS s ON s.channel_id = d.channel_id AND s.user_id = pr.user_id
    LEFT JOIN loaded AS lo ON lo.user_id = pr.user_id
),
first_pass AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY channel_id
            ORDER BY is_loaded, answered DESC, shared_threads DESC, messages DESC, candidate_id
        ) AS first_rank
    FROM pool
),
taken AS (
    SELECT DISTINCT f.channel_id, f.candidate_id
    FROM first_pass AS f
    JOIN first_pass AS g
        ON g.candidate_id = f.candidate_id AND g.channel_id <> f.channel_id AND g.first_rank <= 3
    WHERE g.key_solo > f.key_solo
       OR (g.key_solo = f.key_solo AND g.channel_id < f.channel_id)
)
SELECT
    f.channel_id,
    f.key_person_id,
    f.candidate_id,
    f.answered,
    f.shared_threads,
    f.messages,
    f.is_loaded,
    t.candidate_id IS NOT NULL AS is_taken,
    ROW_NUMBER() OVER (
        PARTITION BY f.channel_id
        ORDER BY
            f.is_loaded,
            t.candidate_id IS NOT NULL,
            f.answered DESC,
            f.shared_threads DESC,
            f.messages DESC,
            f.candidate_id
    ) AS candidate_rank
FROM first_pass AS f
LEFT JOIN taken AS t USING (channel_id, candidate_id)
QUALIFY candidate_rank <= 3;
